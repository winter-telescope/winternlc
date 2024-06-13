# My Package

A package for implementing nonlinearity corrections for WINTER.
- Current implementation is with a rational function with 8 parameters.
- Also has the ability to generate/fit polynomial and other rational functions.

TODO: add more to the bad pixel masking. 
- Currently it only masks pixels which fail the rational fit or are tied high.
- To add: dead pixel and highly nonlinear pixels to the mask. 

## Installation

```bash
pip install .
