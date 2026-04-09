//! LoongArch64 SIMD helpers

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_andn<T: Copy>(a: T, b: T) -> T {
    let not: T = crate::intrinsics::simd::simd_splat(u8::MAX);
    let not: T = crate::intrinsics::simd::simd_xor(a, not);
    crate::intrinsics::simd::simd_and(not, b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_fms<T: Copy>(a: T, b: T, c: T) -> T {
    let c: T = crate::intrinsics::simd::simd_neg(c);
    crate::intrinsics::simd::simd_fma(a, b, c)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_madd<T: Copy>(a: T, b: T, c: T) -> T {
    let mul: T = crate::intrinsics::simd::simd_mul(b, c);
    crate::intrinsics::simd::simd_add(mul, a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_msub<T: Copy>(a: T, b: T, c: T) -> T {
    let mul: T = crate::intrinsics::simd::simd_mul(b, c);
    crate::intrinsics::simd::simd_sub(a, mul)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_nfma<T: Copy>(a: T, b: T, c: T) -> T {
    let fma: T = crate::intrinsics::simd::simd_fma(a, b, c);
    crate::intrinsics::simd::simd_neg(fma)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_nfms<T: Copy>(a: T, b: T, c: T) -> T {
    let fma: T = simd_fms(a, b, c);
    crate::intrinsics::simd::simd_neg(fma)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_nor<T: Copy>(a: T, b: T) -> T {
    let or: T = crate::intrinsics::simd::simd_or(a, b);
    let not: T = crate::intrinsics::simd::simd_splat(u8::MAX);
    crate::intrinsics::simd::simd_xor(or, not)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simd_orn<T: Copy>(a: T, b: T) -> T {
    let not: T = crate::intrinsics::simd::simd_splat(u8::MAX);
    let not: T = crate::intrinsics::simd::simd_xor(b, not);
    crate::intrinsics::simd::simd_or(a, not)
}

macro_rules! impl_vv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let r: $ity = $op(a);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vv;

macro_rules! impl_gv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty, $gty:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $gty) -> $oty {
            unsafe {
                let r: $ity = $op(a as $ety);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_gv;

macro_rules! impl_sv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(0)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>() -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let r: $ity = $op(IMM as $ety);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_sv;

macro_rules! impl_vvv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty, b: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vvv;

macro_rules! impl_vvv_s {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty, b: $oty) -> $oty {
            unsafe {
                let m: $ity = simd_splat((size_of::<$ety>() * 8 - 1) as $ety);
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let b: $ity = simd_and(b, m);
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vvv_s;

macro_rules! impl_vuv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, (size_of::<$ety>() * 8).ilog2());
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simd_splat(IMM as $ety);
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simd_splat(IMM as $ety);
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vuv;

macro_rules! impl_vug {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty, $gty:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $gty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: $ety = $op(a, IMM);
                r as $gty
            }
        }
    };
}

pub(super) use impl_vug;

macro_rules! impl_vsv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>(a: $oty) -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simd_splat(IMM as $ety);
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vsv;

macro_rules! impl_vvvv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $oty, b: $oty, c: $oty) -> $oty {
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = transmute(b);
                let c: $ity = transmute(c);
                let r: $ity = $op(a, b, c);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vvvv;

macro_rules! impl_vugv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ty, $ety:ty, $gty:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty, b: $gty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: $ity = $op(a, IMM, b as $ety);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vugv;
