#!/usr/bin/env python3
"""
Execute all experiments in parallel on a number of cloud machines.
See `machines.py` for information and configuration.
"""

import argparse
import itertools
import os
import subprocess

import distribute_multi as distributed

# Relative path from this file to the source file to run.
SRC_PATH = 'GANerator_generated.py'

# Relative path from this file to the generator file.
GEN_PATH = 'ipynb_to_py.py'


def self_cross_product(it, length=2):
    """
    Return all unordered permutations with repeated elements of `it`
    with itself.

    >>> self_cross_product(('happy', 'feet'))
    (
            ('happy', 'happy'),
            ('happy', 'feet'),  # wombo combo
            ('feet', 'happy'),
            ('feet', 'feet')
    )
    """
    return tuple(itertools.product(it, repeat=length))


# Parameters to be tested. Each parameter must have the same name as in
# the source and _must_ be a tuple of options.
# To set a parameter to a single value, use a 1-tuple like `('param',)`.
test_params = {
    'append_time': (True,),
    'save_dir': ('../GANerator_experiments',),
    'save_params': (True,),
    'dataset_root': ('/mnt/disks/ganerator-disk/ffhq',),
    'img_shape': (64, 128),
    'normalization': self_cross_product(('b', 's', 'n', 'v', 'i', 'a')),
}


def process_command(command):
    return command.translate(str.maketrans('','','[](),\'"'))


def dict_combinations(dict_):
    dict_keys = list(dict_)
    combinations = itertools.product(*[range(len(v)) for v in dict_.values()])
    results = []

    for comb in combinations:
        command = ''
        for key, val_ix in zip(dict_, comb):
            command = ' '.join((command, '--' + key, str(dict_[key][val_ix])))
        results.append(process_command(command[1:]))
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
            help='Only echo the commands to execute.')
    parser.add_argument('--py_bin', type=str, default='python3',
            help='Python 3 binary. Check in your cloud instance.')
    parser.add_argument('--no_gen', action='store_true',
            help='Do not generate the source code again.')
    return parser.parse_args()


def generate_src(py_bin):
    subprocess.run([py_bin,
            os.path.join(os.path.dirname(__file__), GEN_PATH)],
            check=True)


def start_experiments(debug=False, py_bin='python3', no_gen=False):
    try:
        subprocess.check_output([py_bin, '--version'])
    except FileNotFoundError:
        py_bin = 'python'
        try:
            subprocess.check_output([py_bin, '--version'])
        except FileNotFoundError as ex:
            print('Python could not be located. Please specify your binary '
            'using the `--py_bin` argument.')
            raise ex
    if not no_gen:
        generate_src(py_bin)

    # since we are going to start on a cloud machine, we know whether it
    # is going to have unix paths or not
    src_file = os.path.dirname(__file__) + '/' + SRC_PATH
    cmd = [py_bin, src_file]
    combinations = dict_combinations(test_params)
    distributed.run_all_distributed(' '.join(cmd), combinations, debug)
    print('Done!')


if __name__ == '__main__':
    start_experiments(**vars(parse_args()))

