//! Code generation for the sum reduction.
use coresimd::simd::*;

/// Reduction: horizontal sum of the vector elements.
pub trait ReduceAdd {
    /// Result type of the reduction.
    type Acc;
    /// Computes the horizontal sum of the vector elements.
    fn reduce_add(self) -> Self::Acc;
}

macro_rules! red_add {
    ($id:ident, $elem_ty:ident) => {
        impl ReduceAdd for $id {
            type Acc = $elem_ty;
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            fn reduce_add(self) -> Self::Acc {
                unsafe {
                    use coresimd::simd_llvm::simd_reduce_add;
                    simd_reduce_add(self)
                }
            }
            // FIXME: broken in AArch64
            #[cfg(target_arch = "aarch64")]
            #[inline]
            fn reduce_add(self) -> Self::Acc {
                let mut x = self.extract(0) as Self::Acc;
                for i in 1..$id::lanes() {
                    x += self.extract(i) as Self::Acc;
                }
                x
            }
        }
    };
}
red_add!(i8x2, i8);
red_add!(u8x2, u8);
red_add!(i16x2, i16);
red_add!(u16x2, u16);
red_add!(i32x2, i32);
red_add!(u32x2, u32);
red_add!(i64x2, i64);
red_add!(u64x2, u64);
red_add!(i8x4, i8);
red_add!(u8x4, u8);
red_add!(i16x4, i16);
red_add!(u16x4, u16);
red_add!(i32x4, i32);
red_add!(u32x4, u32);
red_add!(i64x4, i64);
red_add!(u64x4, u64);
red_add!(i8x8, i8);
red_add!(u8x8, u8);
red_add!(i16x8, i16);
red_add!(u16x8, u16);
red_add!(i32x8, i32);
red_add!(u32x8, u32);
red_add!(i64x8, i64);
red_add!(u64x8, u64);
red_add!(i8x16, i8);
red_add!(u8x16, u8);
red_add!(i16x16, i16);
red_add!(u16x16, u16);
red_add!(i32x16, i32);
red_add!(u32x16, u32);
red_add!(i8x32, i8);
red_add!(u8x32, u8);
red_add!(i16x32, i16);
red_add!(u16x32, u16);
red_add!(i8x64, i8);
red_add!(u8x64, u8);
red_add!(f32x2, f32);
red_add!(f64x2, f64);
red_add!(f32x4, f32);
red_add!(f64x4, f64);
red_add!(f32x8, f32);
red_add!(f64x8, f64);
red_add!(f32x16, f32);

#[cfg(test)]
mod tests {
    use super::ReduceAdd;
    use coresimd::simd::*;

    // note: these are tested in the portable vector API tests

    #[test]
    fn reduce_add_i32x4() {
        let v = i32x4::splat(1);
        assert_eq!(v.reduce_add(), 4_i32);
    }
    #[test]
    fn reduce_add_u32x4() {
        let v = u32x4::splat(1);
        assert_eq!(v.reduce_add(), 4_u32);
    }
    #[test]
    fn reduce_add_f32x4() {
        let v = f32x4::splat(1.);
        assert_eq!(v.reduce_add(), 4.);
    }
}
