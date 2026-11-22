FROM ubuntu:25.10

# gcc-mips-linux-gnu not available by default in 25.10
RUN echo "deb http://archive.ubuntu.com/ubuntu noble main universe" >> /etc/apt/sources.list.d/noble-copies.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-mips-linux-gnu libc6-dev-mips-cross \
        qemu-system-mips \
        qemu-user \
        make \
        file

ENV CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_LINKER=mips-linux-gnu-gcc \
    CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_RUNNER="qemu-mips -L /usr/mips-linux-gnu" \
    OBJDUMP=mips-linux-gnu-objdump