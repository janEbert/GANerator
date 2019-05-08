#!/usr/bin/env python3

# Time of creation: 2019-05-07 23:04:42

import argparse
from ast import literal_eval
import datetime
from enum import Enum
from pathlib import Path
from time import time, perf_counter as pcounter, process_time as ptime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._jit_internal import weak_module, weak_script_method
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd as ConvBase
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormBase
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.datasets as dsets
import torchvision.transforms as tfs
import torchvision.utils as tvutils

class Norm(Enum):
    # file name parameter interpolation
    BATCH           = 'batch'
    VIRTUAL_BATCH   = 'virtualbatch'
    SPECTRAL        = 'spectral'
    INSTANCE        = 'instance'
    AFFINE_INSTANCE = 'affineinstance'
    NONE            = 'none'

def main():
    # Parameters


    # Paths are written in UNIX-like notation!
    # So write `C:\Users\user\GANerator` as `C:/Users/user/GANerator` or `~/GANerator`.

    # All parameters that take classes also accept strings of the class.

    # Only parameters in the 'Data and Models' section will be saved and loaded!

    parser = argparse.ArgumentParser()
    # Experiment specific
    # ===================
    parser.add_argument('--num_imgs', help='How many images to generate.',
            default=30000)
    parser.add_argument('--exp_name', nargs='+',
            help="File names for this experiment. If `None` or `''`, `append_time` is always `True`.",
            default=None)
    parser.add_argument('--append_time', nargs='+',
            help="Append the current time to the file names (to prevent overwriting).",
            default=True)
    parser.add_argument('--load_dir', nargs='+',
            help="Directory to load saved files from. If `save_dir` is `None`, this also acts as `save_dir`.",
            default='.')
    parser.add_argument('--save_dir', nargs='+',
            help="Directory to save to. If `None`, use the value of `load_dir`.",
            default='.')

    parser.add_argument('--load_exp', nargs='+',
            help="Load the models and parameters from this experiment (previous `exp_name`). Also insert the optionally appended time (WIP: if this value is otherwise ambiguous). Set the parameters `models_file` or `params_file` below to use file names. If set to `True`, use `exp_name`. If `False` or `None`, do not load.",
            default=False)
    parser.add_argument('--params_file', nargs='+',
            help="Load parameters from this path. Set to `False` to not load. Priority over `load_exp`. Set to `True` to ignore this so it does not override `load_exp`.",
            default=True)
    parser.add_argument('--models_file', nargs='+',
            help="Load models from this path. Set to `False` to not load. Priority over `load_exp`. Set to `True` to ignore this so it does not override `load_exp`.",
            default=True)
    parser.add_argument('--load_weights_only', nargs='+',
            help="Load only the models' weights. To continue training, set this to `False`.",
            default=True)

    parser.add_argument('--save_params', nargs='+',
            help="Save the parameters in the 'Data and Models' section to a file.",
            default=False)
    parser.add_argument('--save_weights_only', nargs='+',
            help="Save only the models' weights. To continue training later, set this to `False`.",
            default=False)
    parser.add_argument('--checkpoint_period', nargs='+',
            help="After how many steps to save a model checkpoint. Set to `0` to only save when finished.",
            default=100)

    parser.add_argument('--num_eval_imgs', nargs='+',
            help="How many images to generate for (temporal) evaluation.",
            default=64)


    # Hardware and Multiprocessing
    # ============================
    parser.add_argument('--num_workers', nargs='+',
            help="Amount of worker threads to create on the CPU. Set to `0` to use CPU count.",
            default=0)
    parser.add_argument('--num_gpus', nargs='+',
            help="Amount of GPUs to use. `None` to use all available ones. Set to `0` to run on CPU only.",
            default=None)
    parser.add_argument('--cuda_device_id', nargs='+',
            help="ID of CUDA device. In most cases, this should be left at `0`.",
            default=0)


    # Reproducibility
    # ===============
    parser.add_argument('--seed', nargs='+',
            help="Random seed if `None`. The used seed will always be saved in `saved_seed`.",
            default=0)
    parser.add_argument('--ensure_reproducibility', nargs='+',
            help="If using cuDNN: Set to `True` to ensure reproducibility in favor of performance.",
            default=False)
    parser.add_argument('--flush_denormals', nargs='+',
            help="Whether to set denormals to zero. Some architectures do not support this.",
            default=True)


    # Data and Models
    # ===============
    # Only parameters in this section will be saved and updated when loading.

    parser.add_argument('--dataset_root', nargs='+',
            help="Path to the root folder of the data set. This value is only loaded if set to `None`!",
            default='~/datasets/ffhq')
    parser.add_argument('--dataset_class', nargs='+',
            help="Set this to the torchvision.datasets class (module `dsets`). This value is only loaded if set to `None`!",
            default=dsets.ImageFolder)
    parser.add_argument('--epochs', nargs='+',
            help="Number of training epochs.",
            default=5)
    parser.add_argument('--batch_size', nargs='+',
            help="Size of each training batch. Strongly depends on other parameters.",
            default=512)
    parser.add_argument('--img_channels', nargs='+',
            help="Number of channels in the input images. Normally 3 for RGB and 1 for grayscale.",
            default=3)
    parser.add_argument('--img_shape', nargs='+',
            help="Shape of the output images (excluding channel dimension). Can be an integer to get squares. At the moment, an image can only be square sized and a power of two.",
            default=64)
    parser.add_argument('--resize', nargs='+',
            help="If `True`, resize images; if `False`, crop (to the center).",
            default=True)

    parser.add_argument('--data_mean', nargs='+',
            help="Data is normalized to this mean (per channel).",
            default=0.0)
    parser.add_argument('--data_std', nargs='+',
            help="Data is normalized to this standard deviation (per channel).",
            default=1.0)
    parser.add_argument('--float_dtype', nargs='+',
            help="Float precision as `torch.dtype`.",
            default=torch.float32)
    parser.add_argument('--g_input', nargs='+',
            help="Size of the generator's random input vectors (`z` vector).",
            default=128)

    # GAN hacks
    parser.add_argument('--g_flip_labels', nargs='+',
            help="Switch labels for the generator's training step.",
            default=False)
    parser.add_argument('--d_noisy_labels_prob', nargs='+',
            help="Probability to switch labels when training the discriminator.",
            default=0.0)
    parser.add_argument('--smooth_labels', nargs='+',
            help="Replace discrete labels with slightly different continuous ones.",
            default=False)


    # Values in this paragraph can be either a single value (e.g. an `int`) or a 2-`tuple` of the same type.
    # If a single value, that value will be applied to both the discriminator and generator network.
    # If a 2-`tuple`, the first value will be applied to the discriminator, the second to the generator.
    parser.add_argument('--features', nargs='+',
            help="Relative size of the network's internal features.",
            default=64)
    parser.add_argument('--optimizer', nargs='+',
            help="Optimizer class. GAN hacks recommends `(optim.SGD, optim.Adam)`.",
            default=optim.Adam)
    parser.add_argument('--lr', nargs='+',
            help="Optimizer learning rate. (Second optimizer argument, so not necessarily learning rate.)",
            default=0.0002)
    parser.add_argument('--optim_param', nargs='+',
            help="Third optimizer argument. (For example, `betas` for `Adam` or `momentum` for `SGD`.)",
            default=((0.5, 0.999),))
    parser.add_argument('--optim_kwargs', nargs='+',
            help="Any further optimizer keyword arguments as a dictionary.",
            default={})
    parser.add_argument('--normalization', nargs='+',
            help="Kind of normalization. Must be a `Norm` or in `('b', 'v', 's', 'i', 'a', 'n')`. Usually, spectral normalization is used in the discriminator while virtual batch normalization is used in the generator.",
            default=Norm.BATCH)
    parser.add_argument('--activation', nargs='+',
            help="Activation between hidden layers. GAN hacks recommends `nn.LeakyReLU`.",
            default=(nn.LeakyReLU, nn.ReLU))
    parser.add_argument('--activation_kwargs', nargs='+',
            help="Activation keyword arguments.",
            default=({'negative_slope': 0.2,'inplace': True}, {'inplace': True}))
    params = vars(parser.parse_args())
    for key, val in params.items():
        if type(val) is list:
            if len(val) == 1:
                params[key] = val[0]
            else:
                params[key] = tuple(val)



    # Process parameters

    num_imgs = int(params['num_imgs'])

    # Model parameters as tuples. If it is a tuple, give the class to return as well.
    # If the class is given as `'eval'`, the parameter is literally evaluated if either
    # the tuple or its content begins with a symbol in '({['.
    tuple_params = (
        ('features', int),
        ('optimizer', 'eval'),
        ('lr', float),
        ('optim_param', 'eval'),
        ('optim_kwargs', 'eval'),
        ('normalization', 'eval'),
        ('activation', 'eval'),
        ('activation_kwargs', 'eval'),
    )

    # Parameters that we do *not* want to save (or load).
    # We list these instead of the model parameters as those should be easier to extend.
    static_params = [
        'exp_name',
        'append_time',
        'load_dir',
        'save_dir',

        'load_exp',
        'params_file',
        'models_file',
        'load_weights_only',

        'save_params',
        'save_weights_only',
        'checkpoint_period',

        'num_workers',
        'num_gpus',
        'cuda_device_id',

        'seed',
        'ensure_reproducibility',
        'flush_denormals',
    ]


    def string_to_class(string):
        if type(string) is str:
            string = string.split('.')
            if len(string) == 1:
                m = __builtins__
            else:
                m = globals()[string[0]]
                for part in string[1:-1]:
                    m = getattr(m, part)
            return getattr(m, string[-1])
        else:
            return string


    def argstring(string):
        """
        Return a string converted to its value as if evaled or itself.

        `string` is converted to:
        - the corresponding boolean if it is `'True'` or `'False'`
        - None if `'None'`
        - nothing and returned as it is otherwise.
        """
        return {'True': True, 'False': False, 'None': None}.get(string, string)


    # Experiment name

    append_time = argstring(params['append_time'])
    exp_name    = argstring(params['exp_name'])
    if not exp_name or append_time:
        if exp_name is not str:
            exp_name = ''
        exp_name = ''.join((exp_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))


    # Load parameters

    load_dir = argstring(params['load_dir'])
    save_dir = argstring(params['save_dir'])
    if save_dir is None:
        save_dir = load_dir

    load_exp = argstring(params['load_exp'])

    params_file = argstring(params['params_file'])
    load_params = params_file and (load_exp or type(params_file) is str)

    dataset_root  = argstring(params['dataset_root'])
    dataset_class = string_to_class(params['dataset_class'])

    # Check whether these parameters are `None`.
    # If yes, check that parameters loading is enabled. Otherwise do not update them.
    if dataset_root is None:
        assert load_params, '`dataset_root` cannot be `None` if not loading parameters.'
    else:
        static_params.append('dataset_root')
    if dataset_class is None:
        assert load_params, '`dataset_class` cannot be `None` if not loading parameters.'
    else:
        static_params.append('dataset_class')


    if params_file and (load_exp or type(params_file) is str):
        if type(params_file) is str:
            params_path = Path(params_file)
        elif type(load_exp) is bool:  #
            params_path = Path('{}/params_{}.pt'.format(load_dir, exp_name))
        else:
            params_path = Path('{}/params_{}.pt'.format(load_dir, load_exp))

        params_path = params_path.expanduser()
        upd_params = torch.load(params_path)
        params.update(upd_params)
        del upd_params
    elif params_file == '':
        print("`params_file` is an empty string (`''`). Parameters were not loaded. "
              'Set to `False` to suppress this warning or to `True` to let `load_exp` handle loading.')


    # Hardware and multiprocessing

    num_gpus       = argstring(params['num_gpus'])
    cuda_device_id = int(params['cuda_device_id'])
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        print('Using {} GPUs.'.format(num_gpus))
    else:
        num_gpus = int(num_gpus)
    use_gpus = num_gpus > 0
    multiple_gpus = num_gpus > 1
    if use_gpus:
        assert torch.cuda.is_available(), 'CUDA is not available. ' \
                'Check what is wrong or set `num_gpus` to `0` to run on CPU.'  # Never check for this again
        device = torch.device('cuda:' + str(cuda_device_id))
    else:
        device = torch.device('cpu')

    num_workers = int(params['num_workers'])
    if not num_workers:
        num_workers = mp.cpu_count()
        print('Using {} worker threads.'.format(num_workers))


    # Load model

    models_file = argstring(params['models_file'])
    models_cp = None
    if models_file and (load_exp or type(models_file) is str):
        if type(models_file) is str:
            models_path = Path(models_file)
        elif type(load_exp) is bool:
            models_path = Path('{}/models_{}.tar'.format(load_dir, exp_name))
        else:
            models_path = Path('{}/models_{}.tar'.format(load_dir, load_exp))
        models_path = models_path.expanduser()
        models_cp = torch.load(models_path, map_location=device)
    elif models_file == '':
        print("`models_file` is an empty string (`''`). Models were not loaded. "
              'Set to `False` to suppress this warning or to `True` to let `load_exp` handle loading.')


    # Reproducibility

    seed = argstring(params['seed'])
    if seed is None:
        seed = np.random.randint(10000)
    else:
        seed = int(seed)
    print('Seed: {}.'.format(seed))
    params['saved_seed'] = seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    ensure_reproducibility = argstring(params['ensure_reproducibility'])
    torch.backends.cudnn.deterministic = ensure_reproducibility
    if ensure_reproducibility:
        torch.backends.cudnn.benchmark = False  # This is the default but do it anyway

    flush_denormals = argstring(params['flush_denormals'])
    set_flush_success = torch.set_flush_denormal(flush_denormals)
    if flush_denormals and not set_flush_success:
        print('Not able to flush denormals. `flush_denormals` set to `False`.')
        flush_denormals = False


    # Dataset root

    dataset_root = Path(dataset_root).expanduser()


    # Floating point precision

    float_dtype = string_to_class(params['float_dtype'])
    if float_dtype is torch.float16:
        print('PyTorch does not support half precision well yet. Be careful and assume errors.')
    torch.set_default_dtype(float_dtype)


    # Parameters we do not need to process

    load_weights_only = argstring(params['load_weights_only'])
    save_weights_only = argstring(params['save_weights_only'])
    checkpoint_period = int(params['checkpoint_period'])
    num_eval_imgs     = int(params['num_eval_imgs'])

    epochs       = int(params['epochs'])
    batch_size   = int(params['batch_size'])
    img_channels = int(params['img_channels'])
    resize       = argstring(params['resize'])

    data_mean   = float(params['data_mean'])
    data_std    = float(params['data_std'])
    g_input     = int(params['g_input'])

    g_flip_labels       = argstring(params['g_flip_labels'])
    d_noisy_labels_prob = float(params['d_noisy_labels_prob'])
    smooth_labels       = argstring(params['smooth_labels'])

    assert 0.0 <= d_noisy_labels_prob <= 1.0, \
            'Invalid probability for `d_noisy_labels`. Must be between 0 and 1 inclusively.'

    # Single or tuple parameters

    def param_as_ntuple(key, n=2, return_type=None):
        if return_type is None:
            def return_func(x): return x
        else:
            return_func = return_type
        val = params[key]
        if return_type == 'eval':
            if type(val) is str and val[0] in '({[':
                val = literal_eval(val)

            def return_func(x):
                if type(x) is str and x[0] in '({[':
                    return literal_eval(str(x))
                else:
                    return x

        if type(val) in (tuple, list):
            assert 0 < len(val) <= n, 'Tuples should have length {} (`{}` is `{}`).'.format(n, key, val)
            if len(val) < n:
                if len(val) > 1:
                    print('`{}` is `{}`. Length is less than {}; '.format(key, val, n)
                          + 'last entry has been repeated to fit length.')
                return tuple(map(return_func, tuple(val) + (val[-1],) * (n - len(val))))
            else:
                return tuple(map(return_func, val))
        return (return_func(val),) * n

    def ispow2(x):
        log2 = np.log2(x)
        return log2 == int(log2)


    img_shape = param_as_ntuple('img_shape', return_type=int)
    assert img_shape[0] == img_shape[1], '`img_shape` must be square (same width and height).'
    assert ispow2(img_shape[0]), '`img_shape` must be a power of two (2^n).'

    d_params = {}
    g_params = {}
    for key in tuple_params:
        if type(key) is tuple:
            key, ret_type = key
            d_params[key], g_params[key] = param_as_ntuple(key, return_type=ret_type)
        else:
            d_params[key], g_params[key] = param_as_ntuple(key)


    # Normalization and class parameters

    for p in d_params, g_params:
        normalization = p['normalization']
        if isinstance(normalization, str) and normalization.lower() in ('b', 'v', 's', 'i', 'a', 'n'):
            normalization = {'b': Norm.BATCH, 'v': Norm.VIRTUAL_BATCH,
                             's': Norm.SPECTRAL, 'i': Norm.INSTANCE,
                             'a': Norm.AFFINE_INSTANCE, 'n': Norm.NONE}[normalization]
        if not isinstance(normalization, Norm):
            try:
                normalization = Norm(normalization)
            except ValueError:
                normalization = string_to_class(normalization)
            finally:
                assert isinstance(normalization, Norm), \
                        "Unknown normalization. Must be a `Norm` or in `('b', 'v', 's', 'i', 'a', 'n')`."
        p['normalization'] = normalization

        p['optimizer'] = string_to_class(p['optimizer'])
        p['activation'] = string_to_class(p['activation'])


    save_models_path_str = '{}/models_{}_{{}}_steps.tar'.format(save_dir, exp_name)


    # Generate example batch

    example_noise = torch.randn(batch_size, g_input, 1, 1, device=device)

    # Model helper methods

    @weak_module
    class VirtualBatchNorm2d(nn.Module):
        def __init__(self, num_features, eps=1e-5, affine=True):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
                self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            self.reset_parameters(True)

        def reset_parameters(self, all=False):
            if self.affine:
                nn.init.uniform_(self.weight)
                nn.init.zeros_(self.bias)
            if all:
                self.in_coef = None
                self.ref_coef = None

        @weak_script_method
        def forward(self, input, ref_batch):
            self._check_input_dim(input)
            if self.in_coef is None:
                self._check_input_dim(ref_batch)
                self.in_coef = 1 / (len(ref_batch) + 1)
                self.ref_coef = 1 - self.in_coef

            mean, std, ref_mean, ref_std = self.calculate_statistics(input, ref_batch)
            return self.normalize(input, mean, std), self.normalize(ref_batch, ref_mean, ref_std)

        @weak_script_method
        def calculate_statistics(self, input, ref_batch):
            in_mean,  in_sqmean  = self.calculate_means(input)
            ref_mean, ref_sqmean = self.calculate_means(ref_batch)

            mean   = self.in_coef * in_mean   + self.ref_coef * ref_mean
            sqmean = self.in_coef * in_sqmean + self.ref_coef * ref_sqmean

            std     = torch.sqrt(sqmean     - mean**2     + self.eps)
            ref_std = torch.sqrt(ref_sqmean - ref_mean**2 + self.eps)
            return mean, std, ref_mean, ref_std

        # TODO could be @staticmethod, but check @weak_script_method first
        @weak_script_method
        def calculate_means(self, batch):
            mean   = torch.mean(batch,    0, keepdim=True)
            sqmean = torch.mean(batch**2, 0, keepdim=True)
            return mean, sqmean

        @weak_script_method
        def normalize(self, batch, mean, std):
            return ((batch - mean) / std) * self.weight + self.bias

        @weak_script_method
        def _check_input_dim(self, input):
            if input.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)'
                                 .format(input.dim()))


    def powers(n, b=2):
        """Yield `n` powers of `b` starting from `b**0`."""
        x = 1
        for i in range(n):
            x_old = x
            x *= b
            yield x_old, x


    def layer_with_norm(layer, norm, features):
        if norm is Norm.BATCH:
            return (layer, nn.BatchNorm2d(features))
        elif norm is Norm.VIRTUAL_BATCH:
            return (layer, VirtualBatchNorm2d(features))
        elif norm is Norm.SPECTRAL:
            return (nn.utils.spectral_norm(layer),)
        elif norm is Norm.INSTANCE:
            return (layer, nn.InstanceNorm2d(features))
        elif norm is Norm.AFFINE_INSTANCE:
            return (layer, nn.InstanceNorm2d(features, affine=True))
        elif norm is Norm.NONE:
            return (layer,)
        else:
            raise ValueError("Unknown normalization `'{}'`".format(norm))


    # Define and initialize generator


    # Generator

    class Generator(nn.Module):
        def __init__(self, normalization, activation, activation_kwargs,
                     img_channels, img_shape, features, g_input, reference_batch=None):
            super().__init__()
            self.layers = self.build_layers(normalization, activation, activation_kwargs, img_channels, img_shape, features, g_input)
            if normalization is not Norm.VIRTUAL_BATCH:
                self.reference_batch = None  # we can test for VBN with this invariant
                self.layers = nn.Sequential(*self.layers)
            elif reference_batch is None:
                raise ValueError('Normalization is virtual batch norm, but '
                        '`reference_batch` is `None` or missing.')
            else:
                self.reference_batch = reference_batch  # never `None`
                self.layers = nn.ModuleList(self.layers)

        @staticmethod
        def build_layers(norm, activation, activation_kwargs, img_channels, img_shape, features, g_input):
            """
            Return a list of the layers for the generator network.

            Example for a 64 x 64 image:
            >>> Generator.build_layers(Norm.BATCH, nn.ReLU, {'inplace': True},
                                       img_channels=3, img_shape=(64, 64), features=64, g_input=128)
            [
                # input size is 128 (given by `g_input`)
                nn.ConvTranspose2d(g_input, features * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features * 8),
                nn.ReLU(True),
                # state size is (features * 8) x 4 x 4
                nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.ReLU(True),
                # state size is (features * 4) x 8 x 8
                nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.ReLU(True),
                # state size is (features * 2) x 16 x 16
                nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                # state size is (features) x 32 x 32
                nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # output size is 3 x 64 x 64 (given by `img_channels` and `img_shape`)
            ]
            """
            j = 2 ** (int(np.log2(img_shape[0])) - 3)
            # input size is (g_input)
            layers = [
                *layer_with_norm(nn.ConvTranspose2d(g_input, features * j, 4, 1, 0, bias=False),
                                 norm, features * j),
                activation(**activation_kwargs)
            ]
            # state size is (features * 2^n) x 4 x 4
            # each further layer halves feature size and doubles image size
            while j > 1:
                i = j
                j //= 2
                layers.extend((
                    *layer_with_norm(nn.ConvTranspose2d(features * i, features * j, 4, 2, 1, bias=False),
                                     norm, features * j),
                    activation(**activation_kwargs)
                ))
            # state size is (features) x (img_shape[0] / 2) x (img_shape[1] / 2)
            layers.extend((
                nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            ))
            # output size is (img_channels) x (img_shape[0]) x (img_shape[1])
            return layers

        @weak_script_method
        def forward(self, input):
            # Separation is for performance reasons
            if self.reference_batch is None:
                return self.layers(input)
            else:
                # VBN
                ref_batch = self.reference_batch
                for layer in self.layers:
                    if not isinstance(layer, VirtualBatchNorm2d):
                        input     = layer(input)
                        ref_batch = layer(ref_batch)
                    else:
                        input, ref_batch = layer(input, ref_batch)
                return input


    # Initialization

    def init_weights(module):
        if isinstance(module, ConvBase):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, BatchNormBase):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


    g_net = Generator(g_params['normalization'], g_params['activation'], g_params['activation_kwargs'],
                      img_channels, img_shape, g_params['features'], g_input,
                      example_noise.to(device, float_dtype)).to(device, float_dtype)

    # Load models' checkpoints

    if models_cp is not None:
        g_net.load_state_dict(models_cp['g_net_state_dict'])

    if multiple_gpus:
        g_net = nn.DataParallel(g_net, list(range(num_gpus)))

    if models_cp is None:
        g_net.apply(init_weights)


    real_label = 1
    fake_label = 0

    # Load optimizers' checkpoints

    if models_cp is not None:
        if not load_weights_only:
            try:
                g_optim_state_dict = models_cp['g_optim_state_dict']
            except KeyError:
                print("One of the optimizers' state dicts was not found; probably because "
                      "only the models' weights were saved. Set `load_weights_only` to `True`.")
            g_optimizer.load_state_dict(g_optim_state_dict)
            g_net.train()
        else:
            g_net.eval()


    def generate_fakes(batch_size, start_count, g_input, g_net, device, float_dtype, zfill_len, save_dir):
        i = start_count
        noise = torch.randn(batch_size, g_input, 1, 1, device=device).to(device, float_dtype)
        with torch.no_grad():
            fakes = g_net(noise).detach().cpu()

        for f in fakes:
            tvutils.save_image(f, Path('{}/{}.png'.format(save_dir, str(i).zfill(zfill_len))))
            i += 1

    # Save images generated on noise.
    zfill_len = len(str(num_imgs))

    for i in range(0, num_imgs, batch_size):
        generate_fakes(batch_size, i, g_input, g_net, device, float_dtype, zfill_len, save_dir)


if __name__ == '__main__':
    main()

