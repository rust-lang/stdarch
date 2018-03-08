FROM ubuntu:17.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-mips64el-linux-gnuabi64 libc6-dev-mips64el-cross \
        qemu-system-mips64el file make

ENV CARGO_TARGET_MIPS64EL_UNKNOWN_LINUX_GNUABI64_LINKER=mips64el-linux-gnuabi64-gcc \
    CARGO_TARGET_MIPS64EL_UNKNOWN_LINUX_GNUABI64_RUNNER="qemu-mips64el -cpu I6400 -L /usr/mips64el-linux-gnuabi64" \
    CC_mips64el_unknown_linux_gnuabi64=mips64el-linux-gnuabi64-gcc \
    OBJDUMP=mips64el-linux-gnuabi64-objdump
