FROM ubuntu:17.10
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  ca-certificates \
  libc6-dev \
  gcc-arm-linux-gnueabihf \
  libc6-dev-armhf-cross \
  qemu-user \
  make \
  file \
  g++ \
  g++-arm-linux-gnueabihf \
  git \
  python \
  curl \
  bzip2 \
  perl \
  wget \
  autoconf \
  gettext

ENV CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER=arm-linux-gnueabihf-gcc \
    CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUNNER="qemu-arm -L /usr/arm-linux-gnueabihf" \
    OBJDUMP=arm-linux-gnueabihf-objdump
