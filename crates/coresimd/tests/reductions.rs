#![feature(cfg_target_feature, stdsimd, target_feature)]

#[macro_use]
extern crate stdsimd;

use stdsimd::simd::*;

macro_rules! invoke_arch {
    ($macro:ident, $feature_macro:ident, $id:ident, $elem_ty:ident,
     [$($feature:tt),*]) => {
        $($macro!($feature, $feature_macro, $id, $elem_ty);)*
    }
}

macro_rules! invoke_vectors {
    ($macro:ident, [$(($id:ident, $elem_ty:ident)),*]) => {
        $(
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            invoke_arch!($macro, is_x86_feature_detected, $id, $elem_ty,
                        ["sse", "sse2", "sse3", "ssse3", "sse4.1",
                         "sse4.2", "sse4a", "avx2", "avx2", "avx512f"]);
            #[cfg(target_arch = "aarch64")]
            invoke_arch!($macro, is_aarch64_feature_detected, $id, $elem_ty,
                        ["neon"]);
            #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))]
            invoke_arch!($macro, is_arm_feature_detected, $id, $elem_ty,
                         ["neon"]);
            #[cfg(target_arch = "powerpc")]
            invoke_arch!($macro, is_powerpc_feature_detected, $id, $elem_ty, ["altivec"]);
            #[cfg(target_arch = "powerpc64")]
            invoke_arch!($macro, is_powerpc64_feature_detected, $id, $elem_ty, ["altivec"]);
        )*
    }
}

macro_rules! finvoke {
    ($macro:ident) => {
        invoke_vectors!(
            $macro,
            [(f32x2, f32), (f32x4, f32), (f32x8, f32), (f32x16, f32),
             (f64x2, f64), (f64x4, f64), (f64x8, f64)]
        );
    }
}

macro_rules! iinvoke {
    ($macro:ident) => {
        invoke_vectors!(
            $macro,
            [(i8x2, i8), (i8x4, i8), (i8x8, i8), (i8x16, i8), (i8x32, i8), (i8x64, i8),
             (i16x2, i16), (i16x4, i16), (i16x8, i16), (i16x16, i16), (i16x32, i16),
             (i32x2, i32), (i32x4, i32), (i32x8, i32), (i32x16, i32),
             (i64x2, i64), (i64x4, i64), (i64x8, i64),
             (u8x2, u8), (u8x4, u8), (u8x8, u8), (u8x16, u8), (u8x32, u8), (u8x64, u8),
             (u16x2, u16), (u16x4, u16), (u16x8, u16), (u16x16, u16), (u16x32, u16),
             (u32x2, u32), (u32x4, u32), (u32x8, u32), (u32x16, u32),
             (u64x2, u64), (u64x4, u64), (u64x8, u64)]
        );
    }
}

macro_rules! min_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let n0 = ::std::$elem_ty::NAN;

                assert_eq!(n0.min(-3.0), -3.0);
                assert_eq!((-3.0 as $elem_ty).min(n0), -3.0);

                let v0 = $id::splat(-3.0);

                for i in 0..$id::lanes() {
                    let v = v0.replace(i, n0);
                    if i != $id::lanes() - 1 {
                        assert_eq!(v.min(), -3.0);
                        let mut v = v;
                        for j in 0..i {
                            v = v.replace(j, n0);
                            assert_eq!(v.min(), -3.0);
                        }
                    } else {
                        // not necessarily n0:
                        assert!(v.min().is_nan());
                        let mut v = v;
                        for j in 0..i {
                            v = v.replace(j, n0);
                            assert!(v.min().is_nan());
                        }
                    }
                }

                let vn = $id::splat(n0);
                assert!(vn.min().is_nan());
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn min_nan() {
    finvoke!(min_nan_test);
}

macro_rules! max_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let n0 = ::std::$elem_ty::NAN;

                assert_eq!(n0.max(-3.0), -3.0);
                assert_eq!((-3.0 as $elem_ty).max(n0), -3.0);

                let v0 = $id::splat(-3.0);

                for i in 0..$id::lanes() {
                    let v = v0.replace(i, n0);
                    if i != $id::lanes() - 1 {
                        assert_eq!(v.max(), -3.0);
                        let mut v = v;
                        for j in 0..i {
                            v = v.replace(j, n0);
                            assert_eq!(v.max(), -3.0);
                        }
                    } else {
                        // not necessarily n0:
                        assert!(v.max().is_nan());
                        let mut v = v;
                        for j in 0..i {
                            v = v.replace(j, n0);
                            assert!(v.max().is_nan());
                        }
                    }
                }

                let vn = $id::splat(n0);
                assert!(vn.max().is_nan());
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn max_nan() {
    finvoke!(max_nan_test);
}

macro_rules! sum_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let n0 = ::std::$elem_ty::NAN;

                let v0 = $id::splat(-3.0);

                for i in 0..$id::lanes() {
                    let v = v0.replace(i, n0);
                    assert!(v.sum().is_nan());
                }
                let v = $id::splat(n0);
                assert!(v.sum().is_nan());
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn sum_nan() {
    finvoke!(sum_nan_test);
}

macro_rules! product_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let n0 = ::std::$elem_ty::NAN;

                let v0 = $id::splat(-3.0);

                for i in 0..$id::lanes() {
                    let v = v0.replace(i, n0);
                    assert!(v.product().is_nan());
                }
                let v = $id::splat(n0);
                assert!(v.product().is_nan());
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn product_nan() {
    finvoke!(product_nan_test);
}

trait AsInt {
    type Int;
    fn as_int(self) -> Self::Int;
    fn from_int(Self::Int) -> Self;
}

macro_rules! as_int {
    ($float:ident, $int:ident) => {
        impl AsInt for $float {
            type Int = $int;
            fn as_int(self)  -> $int {
                unsafe { ::std::mem::transmute(self) }
            }
            fn from_int(x: $int) -> $float {
                unsafe { ::std::mem::transmute(x) }
            }
        }
    }
}

as_int!(f32, u32);
as_int!(f64, u64);

trait TreeReduceSum {
    type R;
    fn tree_reduce_sum(self) -> Self::R;
}

macro_rules! tree_reduce_sum_f {
    ($elem_ty:ident) => {
        impl<'a> TreeReduceSum for &'a [$elem_ty] {
            type R = $elem_ty;
            fn tree_reduce_sum(self) -> $elem_ty {
                if self.len() == 2 {
                    self[0] + self[1]
                } else {
                    let mid = self.len() / 2;
                    let (left, right) = self.split_at(mid);
                    Self::tree_reduce_sum(left) + Self::tree_reduce_sum(right)

                }
            }
        }
    }
}
tree_reduce_sum_f!(f32);
tree_reduce_sum_f!(f64);

macro_rules! sum_roundoff_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let mut start = std::$elem_ty::EPSILON;
                let mut sum = 0. as $elem_ty;

                let mut v = $id::splat(0. as $elem_ty);
                for i in 0..$id::lanes() {
                    let c = if i % 2 == 0 { 1e3 } else { -1. };
                    start *= 3.14 * c;
                    sum += start;
                    // println!("{} | start: {}", stringify!($id), start);
                    v = v.replace(i, start);
                }
                let vsum = v.sum();
                println!("{} | lsum: {}", stringify!($id), sum);
                println!("{} | vsum: {}", stringify!($id), vsum);
                let r = vsum.as_int() == sum.as_int();
                // This is false in general; the intrinsic performs a
                // tree-reduce:
                println!("{} | equal: {}", stringify!($id), r);

                let mut a = [0. as $elem_ty; $id::lanes()];
                v.store_unaligned(&mut a);

                let tsum = a.tree_reduce_sum();
                println!("{} | tsum: {}", stringify!($id), tsum);

                // tolerate 1 ULP difference:
                if vsum.as_int() > tsum.as_int() {
                    assert!(vsum.as_int() - tsum.as_int() < 2);
                } else {
                    assert!(tsum.as_int() - vsum.as_int() < 2);
                }
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn sum_roundoff_test() {
    finvoke!(sum_roundoff_test);
}

trait TreeReduceMul {
    type R;
    fn tree_reduce_mul(self) -> Self::R;
}

macro_rules! tree_reduce_mul_f {
    ($elem_ty:ident) => {
        impl<'a> TreeReduceMul for &'a [$elem_ty] {
            type R = $elem_ty;
            fn tree_reduce_mul(self) -> $elem_ty {
                if self.len() == 2 {
                    self[0] * self[1]
                } else {
                    let mid = self.len() / 2;
                    let (left, right) = self.split_at(mid);
                    Self::tree_reduce_mul(left) * Self::tree_reduce_mul(right)

                }
            }
        }
    }
}

tree_reduce_mul_f!(f32);
tree_reduce_mul_f!(f64);

macro_rules! mul_roundoff_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let mut start = std::$elem_ty::EPSILON;
                let mut mul = 1. as $elem_ty;

                let mut v = $id::splat(1. as $elem_ty);
                for i in 0..$id::lanes() {
                    let c = if i % 2 == 0 { 1e3 } else { -1. };
                    start *= 3.14 * c;
                    mul *= start;
                    println!("{} | start: {}", stringify!($id), start);
                    v = v.replace(i, start);
                }
                let vmul = v.product();
                println!("{} | lmul: {}", stringify!($id), mul);
                println!("{} | vmul: {}", stringify!($id), vmul);
                let r = vmul.as_int() == mul.as_int();
                // This is false in general; the intrinsic performs a
                // tree-reduce:
                println!("{} | equal: {}", stringify!($id), r);

                let mut a = [0. as $elem_ty; $id::lanes()];
                v.store_unaligned(&mut a);

                let tmul = a.tree_reduce_mul();
                println!("{} | tmul: {}", stringify!($id), tmul);

                // tolerate 1 ULP difference:
                if vmul.as_int() > tmul.as_int() {
                    assert!(vmul.as_int() - tmul.as_int() < 2);
                } else {
                    assert!(tmul.as_int() - vmul.as_int() < 2);
                }
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn mul_roundoff_test() {
    finvoke!(mul_roundoff_test);
}

macro_rules! sum_overflow_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let start = $elem_ty::max_value() - ($id::lanes() as $elem_ty / 2);

                let v = $id::splat(start as $elem_ty);
                let vsum = v.sum();

                let mut sum = start;
                for _ in 1..$id::lanes() {
                    sum = sum.wrapping_add(start);
                }
                assert_eq!(sum, vsum);
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn sum_overflow_test() {
    iinvoke!(sum_overflow_test);
}

macro_rules! mul_overflow_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let start = $elem_ty::max_value() - ($id::lanes() as $elem_ty / 2);

                let v = $id::splat(start as $elem_ty);
                let vmul = v.product();

                let mut mul = start;
                for _ in 1..$id::lanes() {
                    mul = mul.wrapping_mul(start);
                }
                assert_eq!(mul, vmul);
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn mul_overflow_test() {
    iinvoke!(mul_overflow_test);
}
