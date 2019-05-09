#!/usr/bin/env python3

from enum import Enum
from pathlib import Path
import sys

import torch

NO_NORM_ERROR = 'ERROR'


# easier to just copy rather than import
class Norm(Enum):
    BATCH           = 'batch'
    VIRTUAL_BATCH   = 'virtualbatch'
    SPECTRAL        = 'spectral'
    INSTANCE        = 'instance'
    AFFINE_INSTANCE = 'affineinstance'
    NONE            = 'none'


def print_norm(path):
    """Print both the file name and norm."""
    norm = load_norm(path)
    if norm != NO_NORM_ERROR:
        if type(norm) is tuple:
            formatted = '{} {}'.format(*norm)
        else:
            formatted = '{} {}'.format(norm, norm)
    else:
        formatted = NO_NORM_ERROR

    print(str(path) + '\t' + formatted)


def load_norm(path):
    return torch.load(path).get('normalization', NO_NORM_ERROR)


def main():
    if len(sys.argv) <= 1:
        print('Please supply the folder or .pt file for which to grab '
              'norm parameters from. Also accepts parent directories.')
        return -1
    processed = 0
    for path in map(Path, sys.argv[1:]):
        if path.suffix != '.pt':
            assert path.is_dir(), 'Only accepts folders or .pt files.'

            for child in path.iterdir():
                if child.is_dir():
                    for gchild in child.iterdir():
                        if gchild.suffix == '.pt':
                            print_norm(gchild)
                            processed += 1

                elif child.suffix == '.pt':
                    print_norm(child)
                    processed += 1
        else:
            print_norm(path)
            processed += 1
    if not processed:
        print('No parameter file found.')


if __name__ == '__main__':
    main()

