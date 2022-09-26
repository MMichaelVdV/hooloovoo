# HOOLOOVOO

Image analysis toolbox based on a deep learning networks.

Code Author: Sam De Meyer (https://github.com/SamDM)

🔹🔷🔹

Contains tools to:
- Manipulate image and tensor objects.
- simplify the setup of an image augmentation pipeline.
- simplify setting up a training pipeline.
- parse settings files in json or yaml format.
- do piece-wise image segmentation for images that don't fit into memory.

## Installation

### pip install

The easiest way to install this package is using pip.
First set up a python environment, preferably using `conda` or a pip `virtualenv`.
Then inside the conda/venv environment execute the following command:

```bash
# fresh install
pip install git+https://github.com/MMichaelVdV/hooloovoo.git
# upgrade to newer version
pip install --upgrade git+https://github.com/MMichaelVdV/hooloovoo.git
```

This will automatically pull in all python dependencies.
Note that a fresh install will take over a gigabyte of disk space,
this is mainly due to the huge torch binaries.

### manual install

Clone this repository,
then make sure the contents of the `hooloovoo` and `hooloovoo_applications` folder at the root of this repository are available on your `PYTHONPATH`,
then manually install all the dependencies listed in `setup.py` of this repository.
 
## Usage

Once installed, the `hooloovoo` package can be used as a library.

To run one of the applications, invoke the `hooloovoo` command inside the conda/venv invironment.
