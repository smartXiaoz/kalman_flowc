# Contributing

1. Keep paper-fixed parameters and the scoring protocol unchanged unless the
   change is explicitly documented as a new experiment contract.
2. Run the deterministic test suite with plugin auto-loading disabled:
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.
3. Use a fresh experiment directory for new runs. Never commit `runs/`, live
   `state/`, credentials, or generated logs.
4. Record configuration, seed, budget, code hashes, and the exact scoring
   command for every reported result.
