# GANerator

### Try out, analyze and generate your generators in PyTorch!

This is still a work in progress. There is a lot left to read and implement for
me.

Training [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
(GANs) is a tiresome and hard task. This is primarily because GAN training has
not been well understood yet even after a lot of research went into creating,
understanding and implementing new methods for regularizing and improving GAN
training (such as [https://arxiv.org/abs/1606.03498](virtual batch
normalization), [spectral normalization](https://arxiv.org/abs/1802.05957) or
[boundary equilibria](https://arxiv.org/abs/1703.10717)).

In an effort to improve and accelerate research in GAN training, this
repository collects state of the art approaches and best practices in a modular
fashion to enable users to quickly switch between and try out different methods
to find out what works best for their current situation. For that, I provide a
pure Python version as well as a Jupyter Notebook which may be more comfortable
depending on your use case.

In the future, this repository is supposed to be the one shot stop for training
and evaluating GANs. I want an easily usable, performant and extensible
interface for all your GAN needs. To achieve this, I will start with support
for only one framework. This may later expand to other frequently used ones,
however, first I want to have a well established and sustainable baseline to
build on with other machine learning frameworks.

As this is a university project in the semester before my Master's thesis,
development will slow down for some time then.

## Current Features

Supported frameworks:
- PyTorch

Methods implemented:
- WIP

This repository offers working multiprocessing on GPUs and CPUs, a collection
of smart parameters that are documented and easily usable. In the end, you have
an understandable environment for GAN training working out of the box.

## Setup

You will need [NumPy](https://www.numpy.org/),
[Matplotlib](https://matplotlib.org/) and [PyTorch](https://pytorch.org/)
(including Torchvision). If you want to use the [Jupyter](https://jupyter.org/)
notebook, install that as well (mandatory at the moment as we only have the
notebook).

I do not offer installation instructions or a setup file due to the differences
between CPU and GPU packages. Please look that stuff up yourself in the
following links.

Installation instructions for:
- [NumPy](https://www.scipy.org/install.html)
- [Matplotlib](https://matplotlib.org/users/installing.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Jupyter](https://jupyter.org/install)

### Coding style

At the moment, in the notebook, I do not adhere to the PEP 8 line size
guideline. I feel that notebooks are in most cases supposed to be in fullscreen
and readability would suffer from splitting every second line — sorry! If there
are enough complaints or I change my mind, I will modify the code accordingly.

Any pure Python code later will be documented in ReStructuredText following PEP
257 and I will limit code and comments to a certain line size.

### (TODO!) Generating CelebA-HQ (TODO!)

Download CelebA, copy the path to it and execute the following in your shell
(after activating the environment):
```shell
python generate_celeba-hq.py --path [PATH_TO_CELEBA]
```
This will generate a new folder `celeba-hq` containing the improved dataset.

## TODOs

Methods to implement (still growing):
- feature matching
- minibatch discrimination
- virtual batch normalization
- historical averaging
- one sided label smoothing
- spectral normalization
- bounded equilibrium GANs
- everything from [ganhacks](https://github.com/soumith/ganhacks)

Evaluation methods:
- inception score
- semi-supervised learning

Other:
- convolutional layer visualization
- distributed computing version (Horovod?)
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
- [Progressive Growing of GANs for Improved Quality, Stability, and
  Variation](https://arxiv.org/abs/1710.10196)

## Other resources

- [ganhacks](https://github.com/soumith/ganhacks)
