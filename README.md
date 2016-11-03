# Bayesian Online Compressed Sensing
Online Algorithm for Bayesian Compressed Sensing Reconstruction
by
Paulo V. Rossi (University of Sao Paulo) and
Yoshiyuki Kabashima (Tokyo Institute of Technology)

http://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.022137

This Python code runs the algorithm described in 'Bayesian Online Compressed Sensing' (DOI:https://doi.org/10.1103/PhysRevE.94.022137).
It provides an efficient and accurate way to reconstruct a sparse signal in a bayesian online fashion.

To use it, install the required packages at requirements.txt through

```pip install -r requirements.txt```
  
and run the script with

```python online_CS.py```

All simulation parameters can be set in the script itself in the functions `simulation` and `prior`.
If everything goes smooth, it will plot the reconstructed vector and an estimate of the error.
