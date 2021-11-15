# Description
If you call the model in `Process` from `multiprocessing`, the process is terminated. `try\except` does not catch the error.
# Environment
- Python 3.9.4
- jax: 0.2.24
- rocm-libs: 4.2.0.40200-21
