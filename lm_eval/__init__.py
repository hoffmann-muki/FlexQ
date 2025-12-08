import os

# Expose the bundled `algorithm/lm_eval` package at top-level as `lm_eval`
# so code that does `from lm_eval import ...` continues to work when running
# the repository as a package (e.g. `PYTHONPATH=. python -m algorithm.main`).
ROOT = os.path.dirname(os.path.dirname(__file__))
ALG_LM_EVAL_PATH = os.path.join(ROOT, "algorithm", "lm_eval")

if os.path.isdir(ALG_LM_EVAL_PATH):
    # Tell the import system to search the algorithm/lm_eval folder when
    # `import lm_eval` or `from lm_eval import ...` is used.
    __path__[:] = [ALG_LM_EVAL_PATH]
else:
    # Fallback: leave default behavior which will raise ImportError later.
    __path__ = __path__
