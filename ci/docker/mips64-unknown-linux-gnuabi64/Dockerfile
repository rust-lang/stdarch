FROM debian:testing

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-mips64-linux-gnuabi64 \
        libc6-dev-mips64r6-cross \
        qemu-system-mips64 file make

RUN wget http://codescape-mips-sdk.imgtec.com/components/toolchain/2015.01-7/Codescape.GNU.Tools.Package.2015.01-7.for.MIPS.IMG.Linux.CentOS-5.x86_64.tar.gz
RUN tar xf Codescape.GNU.Tools.Package.2015.01-7.for.MIPS.IMG.Linux.CentOS-5.x86_64.tar.gz

ENV TCPATH=path/to/mips-img-linux-gnu/2015.01-7
ENV PREFIX=${TCPATH}/bin/mips-img-linux-gnu-
ENV CC=${PREFIX}gcc
ENV CXX=${PREFIX}g++
ENV AR=${PREFIX}ar
ENV RANLIB=${PREFIX}ranlib
ENV LINK=$CXX
ENV OBJD=${PREFIX}objdump
ENV GDB=${PREFIX}gdb
ENV RDE=${PREFIX}readelf

ENV CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_LINKER=mips64-linux-gnuabi64-gcc \
    CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_RUNNER="qemu-mips64 -cpu I6400 -L /usr/mips64-linux-gnuabi64" \
    CC_mips64_unknown_linux_gnuabi64=mips64-linux-gnuabi64-gcc \
    CFLAGS_mips64_unknown_linux_gnuabi64="-march=mips64r6 -I/usr/mipsisa64r6-linux-gnuabi64/include" \
    OBJDUMP=mips64-linux-gnuabi64-objdump
