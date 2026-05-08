#!/usr/bin/env bash
# Gate 5: run the unit-test suite at JULIA_NUM_THREADS = 4 and 8 and
# print a one-line summary of each. Exits non-zero if either run fails.
#
# Run from the SMC2FC_functional/ directory.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PKG_DIR" || exit 1

declare -i overall_status=0

for nt in 4 8; do
    echo "==============================================================================="
    echo "Gate 5: running Pkg.test() under JULIA_NUM_THREADS=$nt"
    echo "==============================================================================="
    out=$(JULIA_NUM_THREADS=$nt julia --project=. -e \
            'using Pkg; Pkg.test()' 2>&1)
    status=$?
    summary=$(echo "$out" | grep -E "^Test Summary|^SMC2FC_functional" | tail -3)
    if [ $status -eq 0 ]; then
        echo "  RESULT: PASS  (JULIA_NUM_THREADS=$nt)"
    else
        echo "  RESULT: FAIL  (JULIA_NUM_THREADS=$nt, exit=$status)"
        overall_status=1
    fi
    echo "$summary"
    echo
done

if [ $overall_status -eq 0 ]; then
    echo "Gate 5: ALL THREAD COUNTS PASSED"
else
    echo "Gate 5: AT LEAST ONE THREAD COUNT FAILED"
fi
exit $overall_status
