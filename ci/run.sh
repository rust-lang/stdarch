#!/bin/sh

set -ex

RUSTFLAGS="-C target-cpu=native" cargo test --target $TARGET
RUSTFLAGS="-C target-cpu=native" cargo test --release --target $TARGET
