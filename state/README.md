# Runtime state

The supervisor creates loop state, locks, PID files, planner history, and
retry markers in this directory. These files are machine-local and ignored by
Git; do not commit credentials or live process state.
