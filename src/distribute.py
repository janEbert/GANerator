"""
Sequential distribution.
"""

import os
import subprocess
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


def start_cloud_finish(num, debug):
    if machine.FINISH_COMMAND:
        command = machine.REMOTE_PROCESS_COMMAND.format(
                command=machine.FINISH_COMMAND, suffix=num)
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
    start_cloud_finish(num, debug)
    stop_cloud_instance(num, debug)
    if not debug:
        print('Experiment on machine {}-{} finished'.format(
                machine.INSTANCE_NAME_PREFIX, num))

def start_distributed(command, num, debug):
    start_experiment(command, num, debug)

def start_all_distributed(command, combinations, debug):
    for i, params in enumerate(combinations, 1):
        start_distributed(command + params, i, debug)
    print('All experiments finished.\n'
            'Please check to see if all machines have been shutdown properly.')


def run_distributed(command, num, debug):
    start_distributed(command, num, debug)


def run_all_distributed(command, combinations, debug):
    start_all_distributed(command, combinations, debug)

