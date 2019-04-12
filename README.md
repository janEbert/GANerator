# GANerator

### Try out, analyze and generate your generators in PyTorch!

An easily usable, performant and extensible cross-platform interface for all
your GAN needs.

Training [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
(GANs) is a tiresome and hard task. This is primarily because GAN training has
not been well understood yet even after a lot of research went into creating,
understanding and implementing new methods for regularizing and improving GAN
training (such as [virtual batch
normalization](https://arxiv.org/abs/1606.03498), [spectral
normalization](https://arxiv.org/abs/1802.05957) or
[boundary equilibria](https://arxiv.org/abs/1703.10717)).

In an effort to improve and accelerate research in GAN training, this
repository collects state of the art approaches and best practices in a modular
fashion to enable users to quickly switch between and try out different methods
to find out what works best for their current situation. For that, I provide a
Jupyter Notebook. A pure Python version can be generated from the notebook
where parameters are supplied through command line arguments making automated
experiments much easier.

## Current Features

Only late versions of Python 3 are supported.

Supported framework:
- PyTorch

Other:
- True and tested normalizations for GANs (including a [virtual batch
  normalization](https://arxiv.org/abs/1606.03498) layer)
- Experimental setup for testing and saving many different parameters
- Many [GAN hacks](https://github.com/soumith/ganhacks)
- Generate Python source code from the notebook
- Cross platform

This repository offers working multiprocessing on multiple GPUs and CPUs, a
collection of smart parameters that are documented and easily usable. In the
end, you have an understandable environment for GAN training working out of the
box.

## Setup

You will need [NumPy](https://www.numpy.org/),
[Matplotlib](https://matplotlib.org/) and [PyTorch](https://pytorch.org/)
(including Torchvision). If you want to use the [Jupyter](https://jupyter.org/)
notebook, install that as well (otherwise generate the source code as described
below).

To save animations of your GAN training, you may want either
[FFmpeg](https://ffmpeg.org/) (mp4) or [Pillow](https://python-pillow.org/)
(gif).

Using [Anaconda](https://www.anaconda.com/), everything except for PyTorch and
FFmpeg can be installed in one line:
```zsh
    conda install numpy matplotlib jupyter pillow
```

I do not offer installation instructions or a setup file due to the differences
between CPU and GPU packages and CUDA versions for PyTorch.
Please look that stuff up yourself in the following links.

Installation instructions for:
- [NumPy](https://www.scipy.org/install.html)
- [Matplotlib](https://matplotlib.org/users/installing.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Jupyter](https://jupyter.org/install)
- [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)

## Generating source code and running experiments

You can generate source code from the Jupyter notebook. Any parameter will be
passable as command line arguments, where the default values are the ones
currently in the notebook.

For the parameters, there are some rules you need to follow to generate source
code. These are listed in `src/ipynb_to_py.py`.

To generate, simply execute the following:
```zsh
    ./src/ipynb_to_py.py
```

If you then want to start the experiments, modify and execute
`src/run_experiments.py`. Edit the `test_params` dictionary in that file and
enter the following in your command line of choice:
```zsh
    ./src/run_experiments.py --debug
```

That only started a dry run. To start the tests for real, omit the `--debug`
argument and execute again to see your computer go to work.

A `--distributed` flag is planned that will start each test on a separately
created cloud machine. That, however, is still a TODO and will only support
one cloud service provider out of the box.

## Experimental dataset

The system was set up using
[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), but for the
experiments, [FFHQ](https://github.com/NVlabs/ffhq-dataset) will be used.

### Coding style

At the moment, in the notebook, I do not adhere to the PEP 8 line size
guideline. I feel that notebooks are in most cases supposed to be in fullscreen
and readability would suffer, especially concerning documentation. Sorry! If
there are enough complaints or I change my mind, I will modify the code
accordingly.

Sorry for not enough documentation, that will change with time.

## TODOs

Methods to implement (always growing):
- feature matching
- minibatch discrimination
- historical averaging
- bounded equilibrium GANs
- some missing stuff from [ganhacks](https://github.com/soumith/ganhacks)

Evaluation methods:
- semi-supervised learning

Other:
- more documentation of methods
- automatic experiments for parameter combinations
- convolutional layer visualization
- distributed computing version
- grid search supporting cloud computing providers
- more datasets
- data other than images (sound, 3D objects, ...)
- arbitrary data (anything else)
- [CelebA-HQ](https://arxiv.org/abs/1710.10196) generation
- perhaps more frameworks
- Julia version

## Papers

This is a collection of the papers used for this project.

- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- [Unsupervised Representation Learning with Deep Convolutional Generative
  Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- [BEGAN: Boundary Equilibrium Generative Adversarial
  Networks](https://arxiv.org/abs/1703.10717)
- [Spectral Normalization for Generative Adversarial
  Networks](https://arxiv.org/abs/1802.05957)
- [StarGAN: Unified Generative Adversarial Networks for Multi-Domain
  Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
- [Progressive Growing of GANs for Improved Quality, Stability, and
  Variation](https://arxiv.org/abs/1710.10196)
- [Conditional Generative Adversarial Networks](https://arxiv.org/abs/1411.1784)
- [Large Scale GAN Training For High Fidelity Natural Image
  Synthesis](https://arxiv.org/abs/1809.11096)

## Other resources

- [PyTorch DCGAN
  Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  (which a lot is based on)
- [ganhacks](https://github.com/soumith/ganhacks)
