Riemannian gradient descent for Tucker tensor completion given a mode-1 subspace estimate ([arXiv:2609.24501](https://arxiv.org/abs/2609.24501)).

```
pip install -e .
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m tcsi.demo
```

Sensing matrices follow the Theorem 2 identity `Y_i = e_k (e_g kron e_l)^T (I kron U_s)`, so `d* = d1*d2*d3` and `d* E[y Y] = M*` when the subspace is exact. The step size is `1 / op_norm` of the scaled Gram, then backtracking, because the PDF step-size line is split. Row clipping is the Algorithm 3 ball of radius `sqrt(3 mu r / N)` after the rank-r retraction, not the entrywise cap written in Algorithm 1.

## What does not match the paper
- no 30-trial phase transitions, no COSTCO, no TEC maps
- checks are single synthetic runs at d=12, not the section 5 figures
- with mu=10 the clip does not fire on those problems
