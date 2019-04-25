FROM ubuntu:18.04

RUN apt-get update && apt-get install -y gnupg
RUN echo "deb [trusted=yes] http://ftp.de.debian.org/debian buster main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7638D0442B90D010
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  ca-certificates \
  libc6-dev \
  gcc-mipsisa64r6-linux-gnuabi64 \
  libc6-dev-mips32-mips64r6-cross \
  qemu-user \
  make \
  file

ENV CARGO_TARGET_MIPSISA64R6_UNKNOWN_LINUX_GNUABI64_LINKER=mipsisa64r6-linux-gnuabi64-gcc \
    CARGO_TARGET_MIPSISA64R6_UNKNOWN_LINUX_GNUABI64_RUNNER="qemu-mips64 -L /usr/mipsisa64r6-linux-gnuabi64" \
    CC_mipsisa64r6_unknown_linux_gnuabi64=mipsisa64r6-linux-gnuabi64-gcc \
    OBJDUMP=mipsisa64r6-linux-gnuabi64-objdump
