import os
import subprocess
import threading
import time

import machine

STARTUP_SCRIPT_PATH = os.path.join(os.path.dirname(__file__),
        '../GANerator_GCP_startup.sh')


def start_cloud_instance(num, debug):
    command = machine.START_COMMAND.format(num)
    if debug:
        print(command)
    else:
        command = command.split(' ')
        for i in range(len(command) - 1, -1, -1):
            part = command[i]
            if 'GANERATOR_STARTUP' in part:
                part.replace('GANERATOR_STARTUP', STARTUP_SCRIPT_PATH)
                command[i] = part
                break

        subprocess.run(command, check=True)


def start_cloud_init(num, debug):
    if machine.INIT_COMMAND:
        command = machine.REMOTE_PROCESS_COMMAND.format(
                command=machine.INIT_COMMAND, suffix=num)
        if debug:
            time.sleep(1)
            print(command)
        else:
            subprocess.run(command.split(' '), check=True)


def start_cloud_process(command, num, debug):
    command = machine.REMOTE_PROCESS_COMMAND.format(
            command=command, suffix=num)
    if debug:
        time.sleep(1)
        print(command)
    else:
        subprocess.run(command.split(' '), check=True)


def stop_cloud_instance(num, debug):
    command = machine.END_COMMAND.format(num)
    if debug:
        time.sleep(1)
        print(command)
        time.sleep(1)
    else:
        subprocess.run(command.split(' '), check=True)


def start_experiment(command, num, debug):
    if not debug:
        print('Starting on machine {}-{}'.format(
                machine.INSTANCE_NAME_PREFIX, num))
    start_cloud_instance(num, debug)
    start_cloud_init(num, debug)
    start_cloud_process(command, num, debug)
    stop_cloud_instance(num, debug)
    if not debug:
        print('Experiment on machine {}-{} finished'.format(
                machine.INSTANCE_NAME_PREFIX, num))

def start_distributed(command, num, debug):
    thread = threading.Thread(target=start_experiment, args=(command, num, debug))
    thread.start()
    return thread

def start_all_distributed(command, combinations, debug):
    threads = []
    for i, params in enumerate(combinations, 1):
        threads.append(start_distributed(command + params, i, debug))
    for thread in threads:
        thread.join()
    print('All experiments finished.\n'
            'Please check to see if all machines have been shutdown properly.')


def run_distributed(command, num, debug):
    start_distributed(command, num, debug)


def run_all_distributed(command, combinations, debug):
    start_all_distributed(command, combinations, debug)
