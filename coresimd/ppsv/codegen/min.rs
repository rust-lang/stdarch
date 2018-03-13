//! Code generation for the min reduction.
use coresimd::simd::*;

/// Reduction: horizontal max of the vector elements.
#[cfg_attr(feature = "cargo-clippy", allow(stutter))]
pub trait ReduceMin {
    /// Result type of the reduction.
    type Acc;
    /// Computes the horizontal max of the vector elements.
    fn reduce_min(self) -> Self::Acc;
}

macro_rules! red_min {
    ($id:ident, $elem_ty:ident) => {
        impl ReduceMin for $id {
            type Acc = $elem_ty;
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            fn reduce_min(self) -> Self::Acc {
                unsafe {
                    use coresimd::simd_llvm::simd_reduce_min;
                    simd_reduce_min(self)
                }
            }
            // FIXME: broken on AArch64
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_imports)]
            #[inline]
            fn reduce_min(self) -> Self::Acc {
                use num::Float;
                use cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.min(self.extract(i));
                }
                x
            }
        }
    };
}
red_min!(i8x2, i8);
red_min!(u8x2, u8);
red_min!(i16x2, i16);
red_min!(u16x2, u16);
red_min!(i32x2, i32);
red_min!(u32x2, u32);
red_min!(i64x2, i64);
red_min!(u64x2, u64);
red_min!(i8x4, i8);
red_min!(u8x4, u8);
red_min!(i16x4, i16);
red_min!(u16x4, u16);
red_min!(i32x4, i32);
red_min!(u32x4, u32);
red_min!(i64x4, i64);
red_min!(u64x4, u64);
red_min!(i8x8, i8);
red_min!(u8x8, u8);
red_min!(i16x8, i16);
red_min!(u16x8, u16);
red_min!(i32x8, i32);
red_min!(u32x8, u32);
red_min!(i64x8, i64);
red_min!(u64x8, u64);
red_min!(i8x16, i8);
red_min!(u8x16, u8);
red_min!(i16x16, i16);
red_min!(u16x16, u16);
red_min!(i32x16, i32);
red_min!(u32x16, u32);
red_min!(i8x32, i8);
red_min!(u8x32, u8);
red_min!(i16x32, i16);
red_min!(u16x32, u16);
red_min!(i8x64, i8);
red_min!(u8x64, u8);

red_min!(f32x2, f32);
red_min!(f64x2, f64);
red_min!(f32x4, f32);
red_min!(f64x4, f64);
red_min!(f32x8, f32);
red_min!(f64x8, f64);
red_min!(f32x16, f32);

#[cfg(test)]
mod tests {
    use super::ReduceMin;
    use coresimd::simd::*;

    // note: these are tested in the portable vector API tests

    #[test]
    fn reduce_min_i32x4() {
        let v = i32x4::new(1, 2, -1, 3);
        assert_eq!(v.reduce_min(), -1_i32);
    }
    #[test]
    fn reduce_min_u32x4() {
        let v = u32x4::new(4, 2, 7, 3);
        assert_eq!(v.reduce_min(), 2_u32);
    }
    #[test]
    fn reduce_min_f32x4() {
        let v = f32x4::new(4., 2., -1., 3.);
        assert_eq!(v.reduce_min(), -1.);
    }
}
