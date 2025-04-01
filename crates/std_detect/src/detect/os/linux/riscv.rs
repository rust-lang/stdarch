//! Run-time feature detection for RISC-V on Linux.
//!
//! On RISC-V, detection using auxv only supports single-letter extensions.
//! So, we use riscv_hwprobe that supports multi-letter extensions if available.
//! <https://www.kernel.org/doc/html/latest/arch/riscv/hwprobe.html>

use core::ptr;

use super::auxvec;
use crate::detect::{Feature, bit, cache};

// See <https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/hwprobe.h>
// for riscv_hwprobe struct and RISCV_HWPROBE_* constants.

#[repr(C)]
struct riscv_hwprobe {
    key: i64,
    value: u64,
}

#[allow(non_upper_case_globals)]
const __NR_riscv_hwprobe: libc::c_long = 258;

const RISCV_HWPROBE_KEY_BASE_BEHAVIOR: i64 = 3;
const RISCV_HWPROBE_BASE_BEHAVIOR_IMA: u64 = 1 << 0;

const RISCV_HWPROBE_KEY_IMA_EXT_0: i64 = 4;
const RISCV_HWPROBE_IMA_FD: u64 = 1 << 0;
const RISCV_HWPROBE_IMA_C: u64 = 1 << 1;
const RISCV_HWPROBE_IMA_V: u64 = 1 << 2;
const RISCV_HWPROBE_EXT_ZBA: u64 = 1 << 3;
const RISCV_HWPROBE_EXT_ZBB: u64 = 1 << 4;
const RISCV_HWPROBE_EXT_ZBS: u64 = 1 << 5;
// const RISCV_HWPROBE_EXT_ZICBOZ: u64 = 1 << 6;
const RISCV_HWPROBE_EXT_ZBC: u64 = 1 << 7;
const RISCV_HWPROBE_EXT_ZBKB: u64 = 1 << 8;
const RISCV_HWPROBE_EXT_ZBKC: u64 = 1 << 9;
const RISCV_HWPROBE_EXT_ZBKX: u64 = 1 << 10;
const RISCV_HWPROBE_EXT_ZKND: u64 = 1 << 11;
const RISCV_HWPROBE_EXT_ZKNE: u64 = 1 << 12;
const RISCV_HWPROBE_EXT_ZKNH: u64 = 1 << 13;
const RISCV_HWPROBE_EXT_ZKSED: u64 = 1 << 14;
const RISCV_HWPROBE_EXT_ZKSH: u64 = 1 << 15;
const RISCV_HWPROBE_EXT_ZKT: u64 = 1 << 16;
const RISCV_HWPROBE_EXT_ZVBB: u64 = 1 << 17;
const RISCV_HWPROBE_EXT_ZVBC: u64 = 1 << 18;
const RISCV_HWPROBE_EXT_ZVKB: u64 = 1 << 19;
const RISCV_HWPROBE_EXT_ZVKG: u64 = 1 << 20;
const RISCV_HWPROBE_EXT_ZVKNED: u64 = 1 << 21;
const RISCV_HWPROBE_EXT_ZVKNHA: u64 = 1 << 22;
const RISCV_HWPROBE_EXT_ZVKNHB: u64 = 1 << 23;
const RISCV_HWPROBE_EXT_ZVKSED: u64 = 1 << 24;
const RISCV_HWPROBE_EXT_ZVKSH: u64 = 1 << 25;
const RISCV_HWPROBE_EXT_ZVKT: u64 = 1 << 26;
const RISCV_HWPROBE_EXT_ZFH: u64 = 1 << 27;
const RISCV_HWPROBE_EXT_ZFHMIN: u64 = 1 << 28;
// const RISCV_HWPROBE_EXT_ZIHINTNTL: u64 = 1 << 29;
const RISCV_HWPROBE_EXT_ZVFH: u64 = 1 << 30;
const RISCV_HWPROBE_EXT_ZVFHMIN: u64 = 1 << 31;
// const RISCV_HWPROBE_EXT_ZFA: u64 = 1 << 32;
const RISCV_HWPROBE_EXT_ZTSO: u64 = 1 << 33;
const RISCV_HWPROBE_EXT_ZACAS: u64 = 1 << 34;
// const RISCV_HWPROBE_EXT_ZICOND: u64 = 1 << 35;
const RISCV_HWPROBE_EXT_ZIHINTPAUSE: u64 = 1 << 36;
const RISCV_HWPROBE_EXT_ZVE32X: u64 = 1 << 37;
const RISCV_HWPROBE_EXT_ZVE32F: u64 = 1 << 38;
const RISCV_HWPROBE_EXT_ZVE64X: u64 = 1 << 39;
const RISCV_HWPROBE_EXT_ZVE64F: u64 = 1 << 40;
const RISCV_HWPROBE_EXT_ZVE64D: u64 = 1 << 41;
// const RISCV_HWPROBE_EXT_ZIMOP: u64 = 1 << 42;
// const RISCV_HWPROBE_EXT_ZCA: u64 = 1 << 43;
// const RISCV_HWPROBE_EXT_ZCB: u64 = 1 << 44;
// const RISCV_HWPROBE_EXT_ZCD: u64 = 1 << 45;
// const RISCV_HWPROBE_EXT_ZCF: u64 = 1 << 46;
// const RISCV_HWPROBE_EXT_ZCMOP: u64 = 1 << 47;
const RISCV_HWPROBE_EXT_ZAWRS: u64 = 1 << 48;
// const RISCV_HWPROBE_EXT_SUPM: u64 = 1 << 49;

const RISCV_HWPROBE_KEY_CPUPERF_0: i64 = 5;
const RISCV_HWPROBE_MISALIGNED_FAST: u64 = 3;
const RISCV_HWPROBE_MISALIGNED_MASK: u64 = 7;

const RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF: i64 = 10;
const RISCV_HWPROBE_MISALIGNED_VECTOR_FAST: u64 = 3;

// syscall returns an unsupported error if riscv_hwprobe is not supported,
// so we can safely use this function on older versions of Linux.
fn _riscv_hwprobe(out: &mut [riscv_hwprobe]) -> bool {
    unsafe fn __riscv_hwprobe(
        pairs: *mut riscv_hwprobe,
        pair_count: libc::size_t,
        cpu_set_size: libc::size_t,
        cpus: *mut libc::c_ulong,
        flags: libc::c_uint,
    ) -> libc::c_long {
        unsafe {
            libc::syscall(
                __NR_riscv_hwprobe,
                pairs,
                pair_count,
                cpu_set_size,
                cpus,
                flags,
            )
        }
    }

    let len = out.len();
    unsafe { __riscv_hwprobe(out.as_mut_ptr(), len, 0, ptr::null_mut(), 0) == 0 }
}

/// Read list of supported features from riscv_hwprobe or the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();

    let mut out = [
        riscv_hwprobe {
            key: RISCV_HWPROBE_KEY_BASE_BEHAVIOR,
            value: 0,
        },
        riscv_hwprobe {
            key: RISCV_HWPROBE_KEY_IMA_EXT_0,
            value: 0,
        },
        riscv_hwprobe {
            key: RISCV_HWPROBE_KEY_CPUPERF_0,
            value: 0,
        },
        riscv_hwprobe {
            key: RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF,
            value: 0,
        },
    ];
    if _riscv_hwprobe(&mut out) {
        let mut enable_feature = |feature, enable| {
            if enable {
                value.set(feature as u32);
            }
        };
        if out[0].key != -1 {
            let base_behavior = out[0].value;
            let ima = base_behavior & RISCV_HWPROBE_BASE_BEHAVIOR_IMA != 0;
            // If future RV128I is supported, implement with `enable_feature` here
            #[cfg(target_arch = "riscv32")]
            enable_feature(Feature::rv32i, ima);
            #[cfg(target_arch = "riscv64")]
            enable_feature(Feature::rv64i, ima);
            enable_feature(Feature::m, ima);
            enable_feature(Feature::a, ima);
        }
        if out[1].key != -1 {
            let ima_ext_0 = out[1].value;
            let fd = ima_ext_0 & RISCV_HWPROBE_IMA_FD != 0;
            enable_feature(Feature::f, fd);
            enable_feature(Feature::d, fd);
            enable_feature(Feature::zicsr, fd); // implied by f
            enable_feature(Feature::c, ima_ext_0 & RISCV_HWPROBE_IMA_C != 0);
            // enable_feature(Feature::zicboz, ima_ext_0 & RISCV_HWPROBE_EXT_ZICBOZ != 0);
            enable_feature(Feature::zfh, ima_ext_0 & RISCV_HWPROBE_EXT_ZFH != 0);
            enable_feature(Feature::zfhmin, ima_ext_0 & RISCV_HWPROBE_EXT_ZFHMIN != 0);
            // enable_feature(Feature::zihintntl, ima_ext_0 & RISCV_HWPROBE_EXT_ZIHINTNTL != 0);
            // enable_feature(Feature::zfa, ima_ext_0 & RISCV_HWPROBE_EXT_ZFA != 0);
            enable_feature(Feature::ztso, ima_ext_0 & RISCV_HWPROBE_EXT_ZTSO != 0);
            enable_feature(Feature::zacas, ima_ext_0 & RISCV_HWPROBE_EXT_ZACAS != 0);
            // enable_feature(Feature::zicond, ima_ext_0 & RISCV_HWPROBE_EXT_ZICOND != 0);
            enable_feature(
                Feature::zihintpause,
                ima_ext_0 & RISCV_HWPROBE_EXT_ZIHINTPAUSE != 0,
            );
            // enable_feature(Feature::zimop, ima_ext_0 & RISCV_HWPROBE_EXT_ZIMOP != 0);
            // enable_feature(Feature::zca, ima_ext_0 & RISCV_HWPROBE_EXT_ZCA != 0);
            // enable_feature(Feature::zcb, ima_ext_0 & RISCV_HWPROBE_EXT_ZCB != 0);
            // enable_feature(Feature::zcd, ima_ext_0 & RISCV_HWPROBE_EXT_ZCD != 0);
            // enable_feature(Feature::zcf, ima_ext_0 & RISCV_HWPROBE_EXT_ZCF != 0);
            // enable_feature(Feature::zcmop, ima_ext_0 & RISCV_HWPROBE_EXT_ZCMOP != 0);
            enable_feature(Feature::zawrs, ima_ext_0 & RISCV_HWPROBE_EXT_ZAWRS != 0);
            // enable_feature(Feature::supm, ima_ext_0 & RISCV_HWPROBE_EXT_SUPM != 0);
            // Bit-Manipulation ISA extensions
            enable_feature(Feature::zba, ima_ext_0 & RISCV_HWPROBE_EXT_ZBA != 0);
            enable_feature(Feature::zbb, ima_ext_0 & RISCV_HWPROBE_EXT_ZBB != 0);
            enable_feature(Feature::zbs, ima_ext_0 & RISCV_HWPROBE_EXT_ZBS != 0);
            enable_feature(Feature::zbc, ima_ext_0 & RISCV_HWPROBE_EXT_ZBC != 0);
            // Scalar Crypto ISA extensions
            let zbkb = ima_ext_0 & RISCV_HWPROBE_EXT_ZBKB != 0;
            enable_feature(Feature::zbkb, zbkb);
            let zbkc = ima_ext_0 & RISCV_HWPROBE_EXT_ZBKC != 0;
            enable_feature(Feature::zbkc, zbkc);
            let zbkx = ima_ext_0 & RISCV_HWPROBE_EXT_ZBKX != 0;
            enable_feature(Feature::zbkx, zbkx);
            let zknd = ima_ext_0 & RISCV_HWPROBE_EXT_ZKND != 0;
            enable_feature(Feature::zknd, zknd);
            let zkne = ima_ext_0 & RISCV_HWPROBE_EXT_ZKNE != 0;
            enable_feature(Feature::zkne, zkne);
            let zknh = ima_ext_0 & RISCV_HWPROBE_EXT_ZKNH != 0;
            enable_feature(Feature::zknh, zknh);
            let zksed = ima_ext_0 & RISCV_HWPROBE_EXT_ZKSED != 0;
            enable_feature(Feature::zksed, zksed);
            let zksh = ima_ext_0 & RISCV_HWPROBE_EXT_ZKSH != 0;
            enable_feature(Feature::zksh, zksh);
            let zkt = ima_ext_0 & RISCV_HWPROBE_EXT_ZKT != 0;
            enable_feature(Feature::zkt, zkt);
            let zkn = zbkb & zbkc & zbkx & zkne & zknd & zknh;
            enable_feature(Feature::zkn, zkn);
            // enable_feature(Feature::zk, zkn & zkr & zkt);
            enable_feature(Feature::zks, zbkb & zbkc & zbkx & zksed & zksh);
            // Standard Vector Extensions
            enable_feature(Feature::v, ima_ext_0 & RISCV_HWPROBE_IMA_V != 0);
            enable_feature(Feature::zvfh, ima_ext_0 & RISCV_HWPROBE_EXT_ZVFH != 0);
            enable_feature(Feature::zvfhmin, ima_ext_0 & RISCV_HWPROBE_EXT_ZVFHMIN != 0);
            enable_feature(Feature::zve32x, ima_ext_0 & RISCV_HWPROBE_EXT_ZVE32X != 0);
            enable_feature(Feature::zve32f, ima_ext_0 & RISCV_HWPROBE_EXT_ZVE32F != 0);
            enable_feature(Feature::zve64x, ima_ext_0 & RISCV_HWPROBE_EXT_ZVE64X != 0);
            enable_feature(Feature::zve64f, ima_ext_0 & RISCV_HWPROBE_EXT_ZVE64F != 0);
            enable_feature(Feature::zve64d, ima_ext_0 & RISCV_HWPROBE_EXT_ZVE64D != 0);
            // Vector Cryptography and Bit-manipulation Extensions
            let zvbb = ima_ext_0 & RISCV_HWPROBE_EXT_ZVBB != 0;
            enable_feature(Feature::zvbb, zvbb);
            let zvbc = ima_ext_0 & RISCV_HWPROBE_EXT_ZVBC != 0;
            enable_feature(Feature::zvbc, zvbc);
            let zvkb = zvbb || ima_ext_0 & RISCV_HWPROBE_EXT_ZVKB != 0;
            enable_feature(Feature::zvkb, zvkb);
            let zvkg = ima_ext_0 & RISCV_HWPROBE_EXT_ZVKG != 0;
            enable_feature(Feature::zvkg, zvkg);
            let zvkned = ima_ext_0 & RISCV_HWPROBE_EXT_ZVKNED != 0;
            enable_feature(Feature::zvkned, zvkned);
            enable_feature(Feature::zvknha, ima_ext_0 & RISCV_HWPROBE_EXT_ZVKNHA != 0);
            let zvknhb = ima_ext_0 & RISCV_HWPROBE_EXT_ZVKNHB != 0;
            enable_feature(Feature::zvknhb, zvknhb);
            let zvksed = ima_ext_0 & RISCV_HWPROBE_EXT_ZVKSED != 0;
            enable_feature(Feature::zvksed, zvksed);
            let zvksh = ima_ext_0 & RISCV_HWPROBE_EXT_ZVKSH != 0;
            enable_feature(Feature::zvksh, zvksh);
            let zvkt = ima_ext_0 & RISCV_HWPROBE_EXT_ZVKT != 0;
            enable_feature(Feature::zvkt, zvkt);
            let zvkn = zvkned & zvknhb & zvkb & zvkt;
            enable_feature(Feature::zvkn, zvkn);
            enable_feature(Feature::zvknc, zvkn & zvbc);
            enable_feature(Feature::zvkng, zvkn & zvkg);
            let zvks = zvksed & zvksh & zvkb & zvkt;
            enable_feature(Feature::zvks, zvks);
            enable_feature(Feature::zvksc, zvks & zvbc);
            enable_feature(Feature::zvksg, zvks & zvkg);
        }
        if out[2].key != -1 {
            enable_feature(
                Feature::unaligned_scalar_mem,
                out[2].value & RISCV_HWPROBE_MISALIGNED_MASK == RISCV_HWPROBE_MISALIGNED_FAST,
            );
        }
        if out[3].key != -1 {
            enable_feature(
                Feature::unaligned_vector_mem,
                out[3].value == RISCV_HWPROBE_MISALIGNED_VECTOR_FAST,
            );
        }
        // FIXME: should be enough with hwprobe only, but our code below checks e
        // unavailable in neither uapi/asm/hwprobe.h nor uapi/asm/hwcap.h.
        // https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/hwcap.h
        // return value;
    }

    // FIXME: As said in the above FIXME, we currently alway checks auxv too.
    // // riscv_hwprobe requires Linux 6.4, so we fallback to auxv-based detection on
    // // old Linux kernel.

    let enable_feature = |value: &mut cache::Initializer, feature, enable| {
        if enable {
            value.set(feature as u32);
        }
    };
    let enable_features = |value: &mut cache::Initializer, feature_slice: &[Feature], enable| {
        if enable {
            for feature in feature_slice {
                value.set(*feature as u32);
            }
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/riscv/include/asm/hwcap.h
    //
    // Note that there is no need to check b'v' - b'a' here for the case where riscv_hwprobe is unsupported,
    // since both RISCV_HWPROBE_IMA_V and COMPAT_HWCAP_ISA_V are only supported on Linux 6.5+.
    // https://github.com/torvalds/linux/commit/162e4df137c1fea6557fda3e4cdf5dc6ca6d5510
    // https://github.com/torvalds/linux/commit/dc6667a4e7e36f283bcd0264a0be55adae4d6f86
    let auxv = auxvec::auxv().expect("read auxvec"); // should not fail on RISC-V platform
    #[allow(clippy::eq_op)]
    enable_feature(
        &mut value,
        Feature::a,
        bit::test(auxv.hwcap, (b'a' - b'a').into()),
    );
    enable_feature(
        &mut value,
        Feature::c,
        bit::test(auxv.hwcap, (b'c' - b'a').into()),
    );
    enable_features(
        &mut value,
        &[Feature::d, Feature::f, Feature::zicsr],
        bit::test(auxv.hwcap, (b'd' - b'a').into()),
    );
    enable_features(
        &mut value,
        &[Feature::f, Feature::zicsr],
        bit::test(auxv.hwcap, (b'f' - b'a').into()),
    );
    let has_i = bit::test(auxv.hwcap, (b'i' - b'a').into());
    // If future RV128I is supported, implement with `enable_feature` here
    // Checking target_pointer_width instead of target_arch is incorrect since
    // there are RV64ILP32* ABIs.
    #[cfg(target_arch = "riscv64")]
    enable_feature(&mut value, Feature::rv64i, has_i);
    #[cfg(target_arch = "riscv32")]
    enable_feature(&mut value, Feature::rv32i, has_i);
    // FIXME: e is not exposed in any of asm/hwcap.h, uapi/asm/hwcap.h, uapi/asm/hwprobe.h
    #[cfg(target_arch = "riscv32")]
    enable_feature(
        &mut value,
        Feature::rv32e,
        bit::test(auxv.hwcap, (b'e' - b'a').into()),
    );
    enable_feature(
        &mut value,
        Feature::m,
        bit::test(auxv.hwcap, (b'm' - b'a').into()),
    );

    value
}
