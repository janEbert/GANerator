#!/usr/bin/env python3

import argparse
import itertools
import os
import subprocess

# Relative path from this file to the source file to run.
SRC_PATH = 'GANerator_generated.py'

# Relative path from this file to the generator file.
GEN_PATH = 'ipynb_to_py.py'


def self_cross_product(it):
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
    return tuple(itertools.product(it, repeat=2))


# Parameters to be tested. Each parameter must have the same name as in
# the source and _must_ be a tuple of options.
# To set a parameter to a single value, use a 1-tuple like `('param',)`.
test_params = {
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
    parser.add_argument('--distribute', action='store_true',
            help='(TODO!) Distribute training over cloud instances. '
                 'See `machines.py` for more information.')
    parser.add_argument('--debug', action='store_true',
            help='Only echo the commands to execute.')
    parser.add_argument('--py_bin', type=str, default='python3',
            help='Python 3 binary.')
    parser.add_argument('--no_gen', action='store_true',
            help='Do not generate the source code again.')
    return parser.parse_args()


def generate_src(py_bin):
    subprocess.run([py_bin,
            os.path.join(os.path.dirname(__file__), GEN_PATH)],
            check=True)


def start_experiments(distribute=False, debug=False, py_bin='python3',
        no_gen=False):
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

    if debug:
        def run_func(x):
            print(' '.join(x))
    else:
        def run_func(x):
            subprocess.run(x, check=True)
    src_file = os.path.join(os.path.dirname(__file__), SRC_PATH)
    cmd = [py_bin, src_file]
    combinations = dict_combinations(test_params)
    for i, parameters in enumerate(combinations, 1):
        if not debug:
            print("Starting experiment {}/{}: '{}'".format(
                    i, len(combinations), parameters))
        run_func(cmd + parameters.split(' '))
    print('Done!')


if __name__ == '__main__':
    start_experiments(**vars(parse_args()))
