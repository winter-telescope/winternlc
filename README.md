# WINTER corrections

A package for implementing nonlinearity corrections for WINTER.
- Current implementation is with a rational function with 8 parameters.
- Also has the ability to generate/fit polynomial and other rational functions.

TODO: add more to the bad pixel masking. 
- Currently it only masks pixels which fail the rational fit or are tied high.
- To add: dead pixel and highly nonlinear pixels to the mask.

## Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

## Download corrections files
The corrections files are too large for GIT. You can instead download them from zenodo with the following command.

```
python winter_corrections/get_corrections.py
```

The file `winter_corrections/config.py` specifices which version and zenodo URL to grab. The current recommended versions are as follows:
- v0.1: original corrections files from June 2024 with six operational sensors. 
- v1.1: latest correction files from September 2024 with five operational sensors.

## Get Started
```
winter_corrections/example.py
```
