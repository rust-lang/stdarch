//! Code generation for the product reduction.
use coresimd::simd::*;

/// Reduction: horizontal product of the vector elements.
pub trait ReduceMul {
    /// Result type of the reduction.
    type Acc;
    /// Computes the horizontal product of the vector elements.
    fn reduce_mul(self) -> Self::Acc;
}

macro_rules! red_mul {
    ($id:ident, $elem_ty:ident) => {
        impl ReduceMul for $id {
            type Acc = $elem_ty;
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            fn reduce_mul(self) -> Self::Acc {
                unsafe {
                    use coresimd::simd_llvm::simd_reduce_mul;
                    simd_reduce_mul(self)
                }
            }
            // FIXME: broken in AArch64
            #[cfg(target_arch = "aarch64")]
            #[inline]
            fn reduce_mul(self) -> Self::Acc {
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x *= self.extract(i);
                }
                x
            }
        }
    };
}
red_mul!(i8x2, i8);
red_mul!(u8x2, u8);
red_mul!(i16x2, i16);
red_mul!(u16x2, u16);
red_mul!(i32x2, i32);
red_mul!(u32x2, u32);
red_mul!(i64x2, i64);
red_mul!(u64x2, u64);
red_mul!(i8x4, i8);
red_mul!(u8x4, u8);
red_mul!(i16x4, i16);
red_mul!(u16x4, u16);
red_mul!(i32x4, i32);
red_mul!(u32x4, u32);
red_mul!(i64x4, i64);
red_mul!(u64x4, u64);
red_mul!(i8x8, i8);
red_mul!(u8x8, u8);
red_mul!(i16x8, i16);
red_mul!(u16x8, u16);
red_mul!(i32x8, i32);
red_mul!(u32x8, u32);
red_mul!(i64x8, i64);
red_mul!(u64x8, u64);
red_mul!(i8x16, i8);
red_mul!(u8x16, u8);
red_mul!(i16x16, i16);
red_mul!(u16x16, u16);
red_mul!(i32x16, i32);
red_mul!(u32x16, u32);
red_mul!(i8x32, i8);
red_mul!(u8x32, u8);
red_mul!(i16x32, i16);
red_mul!(u16x32, u16);
red_mul!(i8x64, i8);
red_mul!(u8x64, u8);
red_mul!(f32x2, f32);
red_mul!(f64x2, f64);
red_mul!(f32x4, f32);
red_mul!(f64x4, f64);
red_mul!(f32x8, f32);
red_mul!(f64x8, f64);
red_mul!(f32x16, f32);

#[cfg(test)]
mod tests {
    use super::ReduceMul;
    use coresimd::simd::*;

    // note: these are tested in the portable vector API tests

    #[test]
    fn reduce_mul_i32x4() {
        let v = i32x4::splat(2);
        assert_eq!(v.reduce_mul(), 16_i32);
    }
    #[test]
    fn reduce_mul_u32x4() {
        let v = u32x4::splat(2);
        assert_eq!(v.reduce_mul(), 16_u32);
    }
    #[test]
    fn reduce_mul_f32x4() {
        let v = f32x4::splat(2.);
        assert_eq!(v.reduce_mul(), 16.);
    }
}
