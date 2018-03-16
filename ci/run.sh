#!/usr/bin/env bash

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
export RUST_TEST_THREADS=1
#export RUST_BACKTRACE=1
#export RUST_TEST_NOCAPTURE=1

FEATURES="strict,${FEATURES}"

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "FEATURES=${FEATURES}"
echo "OBJDUMP=${OBJDUMP}"

cargo_test() {
    # Test cfg(test) tests:
    cmd="cargo test --target=${TARGET} --features ${FEATURES} $1"
    cmd="$cmd -p coresimd -p stdsimd --manifest-path crates/stdsimd/Cargo.toml"
    cmd="$cmd -- $2"
    $cmd

    # Build other coresimd test subsets:
    TEST_SUBSETS=(test_intr test_v16 test_v32 test_v64 test_v128 test_v256 test_v512)
    for TEST_SUBSET in ${TEST_SUBSETS[@]}; do
        # Build test sub-set:
        cmd="cargo rustc --lib --profile test --target=${TARGET} --features ${FEATURES} $1"
        cmd="$cmd -p coresimd --manifest-path crates/coresimd/Cargo.toml"
        cmd="$cmd -- $2 --cfg ${TEST_SUBSET}"
        $cmd
    done

    # Run other coresimd test subsets:
    for TEST_SUBSET in ${TEST_SUBSETS[@]}; do
        BUILD_TYPE="debug"
        if [[ $1 = *"--release"* ]]; then
            BUILD_TYPE="release"
        fi

        DEPS=target/${TARGET}/${BUILD_TYPE}/deps

        # Loop over all files in the deps dir that have no extension:
        for filename in $(find ${DEPS}/* -maxdepth 0 -type f ! -name "*.*"); do
            ${filename}
        done
    done
}

cargo_test
cargo_test "--release"
