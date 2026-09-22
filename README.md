implementation in JAX of TCSI (arXiv:2609.24501)

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m tcsi.demo
```

Sensing matrices follow `Y_i = e_k (e_g kron e_l)^T (I kron U_s)`, so `d* = d1*d2*d3`. The PDF step-size line is split, so the step is `1 / op_norm` of the scaled Gram, then backtracking. Row clipping is the Algorithm 3 ball of radius `sqrt(3 mu r / N)` after the rank-r retraction.

What does not match the paper: no 30-trial phase transitions, no COSTCO, no TEC maps. Checks are single synthetic runs at d=12. With mu=10 the clip does not fire on those problems.
