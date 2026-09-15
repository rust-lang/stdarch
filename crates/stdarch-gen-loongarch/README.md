# LoongArch LSX/LASX intrinsic code generator

A small tool that allows to quickly generate intrinsics for the LoongArch LSX/LASX architectures.

The specification for the intrinsics can be found in `lsx.spec` or `lasx.spec`.

To run and re-generate the code run the following from the root of the `stdarch` crate.

LSX:
```
# Generate bindings
OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lsxintrin.h
OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lsx.spec
```

LASX:
```
# Generate bindings
OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasxintrin.h
OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasx.spec
```
