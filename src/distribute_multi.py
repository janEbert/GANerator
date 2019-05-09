import os
import subprocess
import threading
import time

import machines


def join_paths(paths):
    if type(paths) in (tuple, list):
        return os.path.join(os.path.dirname(__file__), *paths)
    else:
        return os.path.join(os.path.dirname(__file__), paths)


STARTUP_SCRIPT_PATH = tuple(map(join_paths, machines.STARTUP_SCRIPT_PATH))
CLOUD_API_PATH = tuple(map(join_paths, machines.CLOUD_API_PATH))


def process_machine_vars(machine_vars):
    for key, val in machine_vars.items():
        if not type(val) in (tuple, list):
            machine_vars[key] = (val,) * machines.NUM_INSTANCES
            continue
        n = len(val)
        if n < machines.NUM_INSTANCES:
            machine_vars[key] = tuple(val) \
                    + (val[-1],) * (machines.NUM_INSTANCES - n)
    return machine_vars


MACHINE_VARS = process_machine_vars({
        'image_family':           machines.IMAGE_FAMILY,
        'zone':                   machines.ZONE,
        'machine_type':           machines.MACHINE_TYPE,
        'gpu_type':               machines.GPU_TYPE,
        'gpu_count':              machines.GPU_COUNT,
        'instance_name_prefix':   machines.INSTANCE_NAME_PREFIX,
        'ro_disk_name':           machines.RO_DISK_NAME,
        'service_account':        machines.SERVICE_ACCOUNT,

        'start_command':          machines.START_COMMAND,
        'remote_process_command': machines.REMOTE_PROCESS_COMMAND,
        'init_command':           machines.INIT_COMMAND,
        'finish_command':         machines.FINISH_COMMAND,
        'end_command':            machines.END_COMMAND,

        'startup_script_path':    STARTUP_SCRIPT_PATH,
        'cloud_api_path':         CLOUD_API_PATH,
})


def list_format(list_, pattern, replacement, reverse_search=False):
    """
    Replace all occurences of `pattern` in the given list of strings
    with `replacement`.
    """
    if reverse_search:
        indizes = range(len(list_) - 1, -1, -1)
    else:
        indizes = range(len(list_))

    for i in indizes:
        part = list_[i]
        if pattern in part:
            list_[i] = part.replace(pattern, replacement)
            break


def start_cloud_instance(num, debug):
    command              = MACHINE_VARS['start_command'][num]

    image_family         = MACHINE_VARS['image_family'][num]
    zone                 = MACHINE_VARS['zone'][num]
    machine_type         = MACHINE_VARS['machine_type'][num]
    gpu_type             = MACHINE_VARS['gpu_type'][num]
    gpu_count            = MACHINE_VARS['gpu_count'][num]
    instance_name_prefix = MACHINE_VARS['instance_name_prefix'][num]
    ro_disk_name         = MACHINE_VARS['ro_disk_name'][num]
    service_account      = MACHINE_VARS['service_account'][num]

    cloud_api_path       = MACHINE_VARS['cloud_api_path'][num]
    startup_script_path  = MACHINE_VARS['startup_script_path'][num]

    command = command.format(
        instance_name_prefix=instance_name_prefix,
        zone=zone,
        machine_type=machine_type,
        image_family=image_family,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        ro_disk_name=ro_disk_name,
        service_account=service_account,
        suffix=num
    )
    command = command.replace('GANERATOR_CLOUD_BIN',
            '"' + cloud_api_path + '"')
    command = command.replace('GANERATOR_STARTUP', startup_script_path)
    if debug:
        print(command)
    else:
        try:
            subprocess.run(command, check=True, shell=True)
            # Wait and hope that formatting has finished by then.
            time.sleep(180)
        except KeyboardInterrupt:
            stop_cloud_instance(num, debug)
            raise


def start_cloud_init(num, debug):
    init_command = MACHINE_VARS['init_command'][num]
    if init_command:
        init_command = init_command.format(suffix=num)

        instance_name_prefix   = MACHINE_VARS['instance_name_prefix'][num]
        remote_process_command = MACHINE_VARS['remote_process_command'][num]
        cloud_api_path         = MACHINE_VARS['cloud_api_path'][num]

        remote_process_command = remote_process_command.format(
                instance_name_prefix=instance_name_prefix, suffix=num)

        command = remote_process_command.replace('GANERATOR_CLOUD_BIN',
                '"' + cloud_api_path + '"')
        command = command.replace('GANERATOR_COMMAND', init_command)
        if debug:
            time.sleep(1)
            print(command)
        else:
            try:
                subprocess.run(command, check=True, shell=True)
            except KeyboardInterrupt:
                stop_cloud_instance(num, debug)
                raise


def start_cloud_process(command, num, debug):
    instance_name_prefix   = MACHINE_VARS['instance_name_prefix'][num]
    remote_process_command = MACHINE_VARS['remote_process_command'][num]
    cloud_api_path         = MACHINE_VARS['cloud_api_path'][num]

    remote_process_command = remote_process_command.format(
            instance_name_prefix=instance_name_prefix, suffix=num)
    remote_process_command = remote_process_command.replace(
            'GANERATOR_CLOUD_BIN', '"' + cloud_api_path + '"')
    remote_process_command = remote_process_command.replace(
            'GANERATOR_COMMAND', command)
    if debug:
        time.sleep(1)
        print(remote_process_command)
    else:
        try:
            subprocess.run(remote_process_command, check=True, shell=True)
        except KeyboardInterrupt:
            pass


def start_cloud_finish(num, debug):
    finish_command = MACHINE_VARS['finish_command'][num]
    if finish_command:
        finish_command = finish_command.format(suffix=num)

        instance_name_prefix   = MACHINE_VARS['instance_name_prefix'][num]
        remote_process_command = MACHINE_VARS['remote_process_command'][num]
        cloud_api_path         = MACHINE_VARS['cloud_api_path'][num]

        remote_process_command = remote_process_command.format(
                instance_name_prefix=instance_name_prefix, suffix=num)

        command = remote_process_command.replace('GANERATOR_CLOUD_BIN',
                '"' + cloud_api_path + '"')
        command = command.replace('GANERATOR_COMMAND', finish_command)
        if debug:
            time.sleep(1)
            print(command)
        else:
            try:
                subprocess.run(command, check=True, shell=True)
            except KeyboardInterrupt:
                pass


def stop_cloud_instance(num, debug):
    instance_name_prefix   = MACHINE_VARS['instance_name_prefix'][num]
    cloud_api_path = MACHINE_VARS['cloud_api_path'][num]
    command = MACHINE_VARS['end_command'][num].format(
            instance_name_prefix=instance_name_prefix, suffix=num)
    command = command.replace('GANERATOR_CLOUD_BIN',
            '"' + cloud_api_path + '"')
    if debug:
        time.sleep(1)
        print(command)
        time.sleep(1)
    else:
        subprocess.run(command, check=True, shell=True)


def start_experiment(command, num, debug):
    if not debug:
        instance_name_prefix = MACHINE_VARS['instance_name_prefix'][num]
        print('Starting on machine {}-{}'.format(instance_name_prefix, num))
    start_cloud_instance(num, debug)
    start_cloud_init(num, debug)
    start_cloud_process(command, num, debug)
    start_cloud_finish(num, debug)
    stop_cloud_instance(num, debug)
    if not debug:
        print('Experiment on machine {}-{} finished'.format(
                instance_name_prefix, num))
        time.sleep(30)  # wait for the machine to actually be deleted


def start_multi_experiments(command, combinations, num, debug):
    for params in combinations:
        start_experiment(command + ' ' + params, num, debug)


def start_distributed(command, combinations, num, debug):
    thread = threading.Thread(target=start_multi_experiments,
            args=(command, combinations, num, debug))
    thread.start()
    return thread


def start_all_distributed(command, combinations, debug):
    threads = []
    print('Starting {} experiments on {} instances.'.format(len(combinations),
            machines.NUM_INSTANCES))
    for i in range(machines.NUM_INSTANCES):
        instance_params = combinations[i::machines.NUM_INSTANCES]
        # start in parallel: sequential processes on instances
        # start sequentially: processes per instance
        threads.append(start_distributed(command, instance_params, i, debug))
    for thread in threads:
        thread.join()
    print('All experiments finished.\n'
            'Please check to see if all machines have been shutdown properly.')


def run_distributed(command, num, debug):
    start_distributed(command, num, debug)


def run_all_distributed(command, combinations, debug):
    start_all_distributed(command, combinations, debug)

