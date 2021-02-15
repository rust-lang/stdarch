FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  xz-utils \
  clang

#ENV WASMER_DIR="/.wasmer"

#RUN curl https://get.wasmer.io -sSfL | sh

#ENV PATH="$PATH:/.wasmer:/.wasmer/bin"

#ENV CARGO_TARGET_WASM32_WASI_RUNNER="wasmer run \
#  --llvm \
#  --enable-simd \
#  --mapdir .::/checkout/target/wasm32-wasi/release/deps \
#  --"

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/dev/wasmtime-dev-x86_64-linux.tar.xz | tar xJf -
ENV PATH=$PATH:/wasmtime-dev-x86_64-linux

ENV CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime \
  --enable-simd \
  --mapdir .::/checkout/target/wasm32-wasi/release/deps \
  --"
