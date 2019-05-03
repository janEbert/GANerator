import os
import subprocess
import threading
import time

from distribute import start_cloud_instance, start_cloud_init, \
                       start_cloud_process, start_cloud_finish, \
                       stop_cloud_instance, start_experiment
import machine


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

