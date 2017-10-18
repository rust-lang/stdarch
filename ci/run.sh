#!/bin/sh

set -ex

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
export RUST_TEST_THREADS=1

cargo test --target $TARGET
cargo test --release --target $TARGET
