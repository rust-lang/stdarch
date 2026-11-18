FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-riscv64-linux-gnu libc6-dev-riscv64-cross \
        llvm wget

RUN wget https://ci-mirrors.rust-lang.org/llvm/llvm-22.1.4-x86_64.tar.gz -O llvm.tar.xz
RUN mkdir llvm
RUN tar -xvf llvm.tar.xz --strip-components=1 -C llvm

ENV CLANG_PATH="/llvm/bin/clang"
ENV GCC_PATH=riscv64-linux-gnu-gcc

ENV CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 -cpu max -L /usr/riscv64-linux-gnu" \
    OBJDUMP=llvm-objdump
