FROM ubuntu:17.10

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ca-certificates \
    libc6-dev \
    gcc-powerpc64-linux-gnu \
    libc6-dev-ppc64-cross \
    qemu-user  \
    qemu-system-ppc \
    make \
    file \
    g++ \
    g++-powerpc64-linux-gnu \
    git \
    python \
    curl \
    bzip2 \
    perl \
    wget \
    autoconf \
    gettext

ENV CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_LINKER=powerpc64-linux-gnu-gcc \
    CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_RUNNER="qemu-ppc64 -L /usr/powerpc64-linux-gnu" \
    CC=powerpc64-linux-gnu-gcc \
    OBJDUMP=powerpc64-linux-gnu-objdump
