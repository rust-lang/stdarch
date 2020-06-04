FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  clang \
  cmake \
  curl \
  git \
  libc6-dev \
  make \
  python \
  python3 \
  xz-utils

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/dev/wasmtime-dev-x86_64-linux.tar.xz | tar xJf -
ENV PATH=$PATH:/wasmtime-dev-x86_64-linux

ENV CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime \
  --enable-simd \
  --mapdir .::/checkout/target/wasm32-wasi/release/deps \
  --"
