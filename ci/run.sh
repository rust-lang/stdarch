#!/usr/bin/env sh

set -ex

: "${TARGET?The TARGET environment variable must be set.}"

# Tests are all super fast anyway, and they fault often enough on travis that
# having only one thread increases debuggability to be worth it.
export RUST_TEST_THREADS=1
#export RUST_BACKTRACE=full
#export RUST_TEST_NOCAPTURE=1

RUSTFLAGS="${RUSTFLAGS} --cfg stdsimd_strict"

case ${TARGET} in
    # On 32-bit use a static relocation model which avoids some extra
    # instructions when dealing with static data, notably allowing some
    # instruction assertion checks to pass below the 20 instruction limit. If
    # this is the default, dynamic, then too many instructions are generated
    # when we assert the instruction for a function and it causes tests to fail.
    #
    # It's not clear why `-Z plt=yes` is required here. Probably a bug in LLVM.
    # If you can remove it and CI passes, please feel free to do so!
    i686-* | i586-*)
        export RUSTFLAGS="${RUSTFLAGS} -C relocation-model=static -Z plt=yes"
        ;;
esac

echo "RUSTFLAGS=${RUSTFLAGS}"
echo "FEATURES=${FEATURES}"
echo "OBJDUMP=${OBJDUMP}"
echo "STDSIMD_DISABLE_ASSERT_INSTR=${STDSIMD_DISABLE_ASSERT_INSTR}"
echo "STDSIMD_TEST_EVERYTHING=${STDSIMD_TEST_EVERYTHING}"

cargo_test() {
    subcmd="test"
    if [ "${NORUN}" = "1" ]; then
        export subcmd="build"
    fi
    cmd="cargo ${subcmd} --target=${TARGET} ${1}"
    if [ "${NOSTD}" = "1" ]; then
        cmd="$cmd -p coresimd"
    else
        cmd="$cmd -p coresimd -p stdsimd"
    fi
    cmd="$cmd -- $2"
    $cmd
}

#cargo_test
#cargo_test "--release"

# Test targets compiled with extra features.
case ${TARGET} in
    x86*)
        export STDSIMD_DISABLE_ASSERT_INSTR=1
        export RUSTFLAGS="${RUSTFLAGS} -C target-feature=+avx"
        #cargo_test "--release"
        ;;
    wasm32-unknown-unknown*)
        # export RUSTFLAGS="${RUSTFLAGS} -C target-feature=+simd128"
        #cargo_test "--release --features=wasm_simd128"
        ;;
    *)
        ;;
esac

if [ "${NOLIBSTDBUILD}" = "1" ]; then
    echo "Whether libstd builds with this stdsimd is not tested!"
else
    echo "Testing that libcore and libstd build with this stdsimd..."
    stdsimd="$(pwd)"

    case ${TARGET} in
        *apple*)
            export RUSTC_DIR=~/rustc
            ;;
        *windows*)
            export RUSTC_DIR=~/rustc
            ;;
        *)
            export RUSTC_DIR=/rustc
            ;;
    esac

    git clone --depth 1 https://github.com/rust-lang/rust.git "${RUSTC_DIR}"
    cd "${RUSTC_DIR}"
    git submodule sync
    ./x.py clean

    cp -rv "${stdsimd}"/* src/stdsimd/

    (
        cd src/stdsimd
        git config --global user.email "travis@travis.com"
        git config --global user.name "Travis CI"
        git add -u -v
        git commit -m "modifications"
    )

    ./x.py check src/libcore --stage 1 --target "${TARGET}"
    ./x.py check src/libstd --stage 1 --target "${TARGET}"
fi
