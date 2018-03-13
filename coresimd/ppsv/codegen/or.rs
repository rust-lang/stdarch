//! Code generation for the or reduction.
use coresimd::simd::*;

/// Reduction: horizontal bitwise or of the vector elements.
#[cfg_attr(feature = "cargo-clippy", allow(stutter))]
pub trait ReduceOr {
    /// Result of the reduction.
    type Acc;
    /// Computes the horizontal bitwise or of the vector elements.
    fn reduce_or(self) -> Self::Acc;
}

macro_rules! red_or {
    ($id:ident, $elem_ty:ident) => {
        impl ReduceOr for $id {
            type Acc = $elem_ty;
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            fn reduce_or(self) -> Self::Acc {
                unsafe {
                    use coresimd::simd_llvm::simd_reduce_or;
                    simd_reduce_or(self)
                }
            }
            // FIXME: broken in AArch64
            #[cfg(target_arch = "aarch64")]
            #[inline]
            fn reduce_or(self) -> Self::Acc {
                let mut x = self.extract(0) as Self::Acc;
                for i in 1..$id::lanes() {
                    x |= self.extract(i) as Self::Acc;
                }
                x
            }
        }
    };
}
red_or!(i8x2, i8);
red_or!(u8x2, u8);
red_or!(i16x2, i16);
red_or!(u16x2, u16);
red_or!(i32x2, i32);
red_or!(u32x2, u32);
red_or!(i64x2, i64);
red_or!(u64x2, u64);
red_or!(i8x4, i8);
red_or!(u8x4, u8);
red_or!(i16x4, i16);
red_or!(u16x4, u16);
red_or!(i32x4, i32);
red_or!(u32x4, u32);
red_or!(i64x4, i64);
red_or!(u64x4, u64);
red_or!(i8x8, i8);
red_or!(u8x8, u8);
red_or!(i16x8, i16);
red_or!(u16x8, u16);
red_or!(i32x8, i32);
red_or!(u32x8, u32);
red_or!(i64x8, i64);
red_or!(u64x8, u64);
red_or!(i8x16, i8);
red_or!(u8x16, u8);
red_or!(i16x16, i16);
red_or!(u16x16, u16);
red_or!(i32x16, i32);
red_or!(u32x16, u32);
red_or!(i8x32, i8);
red_or!(u8x32, u8);
red_or!(i16x32, i16);
red_or!(u16x32, u16);
red_or!(i8x64, i8);
red_or!(u8x64, u8);

red_or!(b8x2, i8);
red_or!(b8x4, i8);
red_or!(b8x8, i8);
red_or!(b8x16, i8);
red_or!(b8x32, i8);
red_or!(b8x64, i8);

#[cfg(test)]
mod tests {
    use super::ReduceOr;
    use coresimd::simd::*;

    // note: these are tested in the portable vector API tests

    #[test]
    fn reduce_or_i32x4() {
        let v = i32x4::splat(1);
        assert_eq!(v.reduce_or(), 1_i32);
        let v = i32x4::new(1, 1, 0, 1);
        assert_eq!(v.reduce_or(), 1_i32);
    }
}
