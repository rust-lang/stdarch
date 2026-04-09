//! LoongArch64 SIMD helpers

pub(super) const trait SimdL: Sized {
    type Elem;

    unsafe fn splat(v: i64) -> Self;
}

macro_rules! impl_simdl {
    ($v:ident, $e:ty) => {
        #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
        impl const SimdL for crate::core_arch::simd::$v {
            type Elem = $e;

            #[inline(always)]
            unsafe fn splat(v: i64) -> Self {
                crate::intrinsics::simd::simd_splat(v as Self::Elem)
            }
        }
    };
}

impl_simdl!(i8x16, i8);
impl_simdl!(i8x32, i8);
impl_simdl!(u8x16, u8);
impl_simdl!(u8x32, u8);
impl_simdl!(i16x8, i16);
impl_simdl!(i16x16, i16);
impl_simdl!(u16x8, u16);
impl_simdl!(u16x16, u16);
impl_simdl!(i32x4, i32);
impl_simdl!(i32x8, i32);
impl_simdl!(u32x4, u32);
impl_simdl!(u32x8, u32);
impl_simdl!(i64x2, i64);
impl_simdl!(i64x4, i64);
impl_simdl!(u64x2, u64);
impl_simdl!(u64x4, u64);

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_andn<T: Copy + const SimdL>(a: T, b: T) -> T {
    crate::intrinsics::simd::simd_and(simdl_not(a), b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_fms<T: Copy>(a: T, b: T, c: T) -> T {
    let c: T = crate::intrinsics::simd::simd_neg(c);
    crate::intrinsics::simd::simd_fma(a, b, c)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_madd<T: Copy>(a: T, b: T, c: T) -> T {
    let mul: T = crate::intrinsics::simd::simd_mul(b, c);
    crate::intrinsics::simd::simd_add(mul, a)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_msub<T: Copy>(a: T, b: T, c: T) -> T {
    let mul: T = crate::intrinsics::simd::simd_mul(b, c);
    crate::intrinsics::simd::simd_sub(a, mul)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_nfma<T: Copy>(a: T, b: T, c: T) -> T {
    let fma: T = crate::intrinsics::simd::simd_fma(a, b, c);
    crate::intrinsics::simd::simd_neg(fma)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_nfms<T: Copy>(a: T, b: T, c: T) -> T {
    let fma: T = simdl_fms(a, b, c);
    crate::intrinsics::simd::simd_neg(fma)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_nor<T: Copy + const SimdL>(a: T, b: T) -> T {
    let or: T = crate::intrinsics::simd::simd_or(a, b);
    simdl_not(or)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_not<T: Copy + const SimdL>(a: T) -> T {
    let not: T = simdl_splat(!0);
    crate::intrinsics::simd::simd_xor(a, not)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_orn<T: Copy + const SimdL>(a: T, b: T) -> T {
    crate::intrinsics::simd::simd_or(a, simdl_not(b))
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_shl<T: Copy + const SimdL>(a: T, b: T) -> T {
    let m: T = simdl_splat((size_of::<T::Elem>() * 8 - 1) as i64);
    let b: T = crate::intrinsics::simd::simd_and(b, m);
    crate::intrinsics::simd::simd_shl(a, b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_shr<T: Copy + const SimdL>(a: T, b: T) -> T {
    let m: T = simdl_splat((size_of::<T::Elem>() * 8 - 1) as i64);
    let b: T = crate::intrinsics::simd::simd_and(b, m);
    crate::intrinsics::simd::simd_shr(a, b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(super) const unsafe fn simdl_splat<T: Copy + const SimdL>(a: i64) -> T {
    T::splat(a)
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
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name(a: $gty) -> $oty {
            unsafe {
                let r: $ity = $op(a.into());
                transmute(r)
            }
        }
    };
}

pub(super) use impl_gv;

macro_rules! impl_sv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(0)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>() -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let r: $ity = $op(IMM.into());
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

macro_rules! impl_vuv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, (size_of::<<$ity as SimdL>::Elem>() * 8).ilog2());
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simdl_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simdl_splat(IMM.into());
                let r: $ity = $op(a, b);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vuv;

macro_rules! impl_vug {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty) -> $gty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: <$ity as SimdL>::Elem = $op(a, IMM);
                r as $gty
            }
        }
    };
}

pub(super) use impl_vug;

macro_rules! impl_vsv {
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: i32>(a: $oty) -> $oty {
            static_assert_simm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let b: $ity = simdl_splat(IMM.into());
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
    ($ft:literal, $name:ident, $op:ident, $oty:ty, $ity:ident, $gty:ty, $ibs:expr) => {
        #[inline(always)]
        #[target_feature(enable = $ft)]
        #[rustc_legacy_const_generics(1)]
        #[unstable(feature = "stdarch_loongarch", issue = "117427")]
        pub fn $name<const IMM: u32>(a: $oty, b: $gty) -> $oty {
            static_assert_uimm_bits!(IMM, $ibs);
            unsafe {
                let a: $ity = transmute(a);
                let r: $ity = $op(a, IMM, b as <$ity as SimdL>::Elem);
                transmute(r)
            }
        }
    };
}

pub(super) use impl_vugv;
