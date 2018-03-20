//! Shuffle vectors
#![allow(unused)]

macro_rules! impl_shuffle {
    ($_e:expr) => {
        use super::Simd;

        /// This trait is not public.
        ///
        /// It is only used to constrain the return
        /// type of the vector shuffles.
        pub trait Shuffle<A> {
            /// The result type of the shuffle.
            type Output;
        }

        // These implementations for the native
        // types allow constraining the the shuffles for
        // each portable vector type by just implementing
        // the Simd trait once for them.

        impl Shuffle<[u32; 2]> for i8 {
            type Output = super::super::i8x2;
        }
        impl Shuffle<[u32; 4]> for i8 {
            type Output = super::super::i8x4;
        }
        impl Shuffle<[u32; 8]> for i8 {
            type Output = super::super::i8x8;
        }
        impl Shuffle<[u32; 16]> for i8 {
            type Output = super::super::i8x16;
        }
        impl Shuffle<[u32; 32]> for i8 {
            type Output = super::super::i8x32;
        }
        impl Shuffle<[u32; 64]> for i8 {
            type Output = super::super::i8x64;
        }
        impl Shuffle<[u32; 2]> for u8 {
            type Output = super::super::u8x2;
        }
        impl Shuffle<[u32; 4]> for u8 {
            type Output = super::super::u8x4;
        }
        impl Shuffle<[u32; 8]> for u8 {
            type Output = super::super::u8x8;
        }
        impl Shuffle<[u32; 16]> for u8 {
            type Output = super::super::u8x16;
        }
        impl Shuffle<[u32; 32]> for u8 {
            type Output = super::super::u8x32;
        }
        impl Shuffle<[u32; 64]> for u8 {
            type Output = super::super::u8x64;
        }

        impl Shuffle<[u32; 2]> for i16 {
            type Output = super::super::i16x2;
        }
        impl Shuffle<[u32; 4]> for i16 {
            type Output = super::super::i16x4;
        }
        impl Shuffle<[u32; 8]> for i16 {
            type Output = super::super::i16x8;
        }
        impl Shuffle<[u32; 16]> for i16 {
            type Output = super::super::i16x16;
        }
        impl Shuffle<[u32; 32]> for i16 {
            type Output = super::super::i16x32;
        }
        impl Shuffle<[u32; 2]> for u16 {
            type Output = super::super::u16x2;
        }
        impl Shuffle<[u32; 4]> for u16 {
            type Output = super::super::u16x4;
        }
        impl Shuffle<[u32; 8]> for u16 {
            type Output = super::super::u16x8;
        }
        impl Shuffle<[u32; 16]> for u16 {
            type Output = super::super::u16x16;
        }
        impl Shuffle<[u32; 32]> for u16 {
            type Output = super::super::u16x32;
        }

        impl Shuffle<[u32; 2]> for i32 {
            type Output = super::super::i32x2;
        }
        impl Shuffle<[u32; 4]> for i32 {
            type Output = super::super::i32x4;
        }
        impl Shuffle<[u32; 8]> for i32 {
            type Output = super::super::i32x8;
        }
        impl Shuffle<[u32; 16]> for i32 {
            type Output = super::super::i32x16;
        }
        impl Shuffle<[u32; 2]> for u32 {
            type Output = super::super::u32x2;
        }
        impl Shuffle<[u32; 4]> for u32 {
            type Output = super::super::u32x4;
        }
        impl Shuffle<[u32; 8]> for u32 {
            type Output = super::super::u32x8;
        }
        impl Shuffle<[u32; 16]> for u32 {
            type Output = super::super::u32x16;
        }
        impl Shuffle<[u32; 2]> for f32 {
            type Output = super::super::f32x2;
        }
        impl Shuffle<[u32; 4]> for f32 {
            type Output = super::super::f32x4;
        }
        impl Shuffle<[u32; 8]> for f32 {
            type Output = super::super::f32x8;
        }
        impl Shuffle<[u32; 16]> for f32 {
            type Output = super::super::f32x16;
        }

        impl Shuffle<[u32; 2]> for i64 {
            type Output = super::super::i64x2;
        }
        impl Shuffle<[u32; 4]> for i64 {
            type Output = super::super::i64x4;
        }
        impl Shuffle<[u32; 8]> for i64 {
            type Output = super::super::i64x8;
        }
        impl Shuffle<[u32; 2]> for u64 {
            type Output = super::super::u64x2;
        }
        impl Shuffle<[u32; 4]> for u64 {
            type Output = super::super::u64x4;
        }
        impl Shuffle<[u32; 8]> for u64 {
            type Output = super::super::u64x8;
        }
        impl Shuffle<[u32; 2]> for f64 {
            type Output = super::super::f64x2;
        }
        impl Shuffle<[u32; 4]> for f64 {
            type Output = super::super::f64x4;
        }
        impl Shuffle<[u32; 8]> for f64 {
            type Output = super::super::f64x8;
        }

        /// The shuffle intrinsics are reimported here.
        ///
        /// At typeck both input vector types are required to be equal and thus
        /// have the same length, the arrays of indices are required to have the
        /// correct lengths, and the result type is constrained by the `where`
        /// clauses below such that only the correct result types can type check.
        ///
        /// FIXME: The only way to produce a monomorphization-time error here is
        /// to pass the intrinsic an element index that is out-of-bounds. Fixing
        /// this probably requires checking that the indices are in-bounds in
        /// MIR typeck.
        mod intrinsics {
            use super::{Simd, Shuffle};
            extern "platform-intrinsic" {
                pub fn simd_shuffle2<T: Simd, U>(a: T, b: T, indices: [u32; 2]) -> U
                    where <T as Simd>::Element: Shuffle<[u32; 2], Output = U>;

                pub fn simd_shuffle4<T: Simd, U>(a: T, b: T, indices: [u32; 4]) -> U
                    where <T as Simd>::Element: Shuffle<[u32; 4], Output = U>;

                pub fn simd_shuffle8<T: Simd, U>(a: T, b: T, indices: [u32; 8]) -> U
                    where <T as Simd>::Element: Shuffle<[u32; 8], Output = U>;

                pub fn simd_shuffle16<T: Simd, U>(a: T, b: T, indices: [u32; 16]) -> U
                    where <T as Simd>::Element: Shuffle<[u32; 16], Output = U>;

                pub fn simd_shuffle32<T: Simd, U>(a: T, b: T, indices: [u32; 32]) -> U
                    where <T as Simd>::Element: Shuffle<[u32; 32], Output = U>;

                pub fn simd_shuffle64<T: Simd, U>(a: T, b: T, indices: [u32; 64]) -> U
                    where <T as Simd>::Element: Shuffle<[u32; 64], Output = U>;
            }
        }

        pub use self::intrinsics::simd_shuffle2 as __shuffle_vector2;
        pub use self::intrinsics::simd_shuffle4 as __shuffle_vector4;
        pub use self::intrinsics::simd_shuffle8 as __shuffle_vector8;
        pub use self::intrinsics::simd_shuffle16 as __shuffle_vector16;
        pub use self::intrinsics::simd_shuffle32 as __shuffle_vector32;
        pub use self::intrinsics::simd_shuffle64 as __shuffle_vector64;
    }
}

vector_impl!([impl_shuffle, 0]);

/// Shuffles vector elements.
///
/// This macro returns a new vector that contains a shuffle of the elements in
/// one or two input vectors:
///
/// * `shuffle!(vec, [indices...])`: one-vector version
/// * `shuffle!(vec0, vec1, [indices...])`: two-vector version
///
/// In the two-vector version both `vec0` and `vec1` must have the same type.
/// The element type of the resulting vector is the element type of the input
/// vector.
///
/// The number of `indices` must be a power-of-two in range `[0, 64)` smaller
/// than two times the number of lanes in the input vector. The length of the
/// resulting vector equals the number of indices provided.
///
/// Given a vector with `N` lanes, the indices in range `[0, N)` refer to the
/// `N` elements in the vector. In the two-vector version, the indices in range
/// `[N, 2*N)` refer to elements in the second vector.
///
/// # Examples
///
/// ```
/// # #![feature(stdsimd)]
/// # #[macro_use] extern crate coresimd;
/// # use coresimd::simd::*;
/// # fn main() {
/// // Shuffle allows reordering the elements of a vector:
/// let x = i32x4::new(1, 2, 3, 4);
/// let r = shuffle!(x, [2, 1, 3, 0]);
/// assert_eq!(r, i32x4::new(3, 2, 4, 1));
///
/// // The resulting vector can be smaller than the input:
/// let r = shuffle!(x, [1, 3]);
/// assert_eq!(r, i32x2::new(2, 4));
///
/// // Equal:
/// let r = shuffle!(x, [1, 3, 2, 0]);
/// assert_eq!(r, i32x4::new(2, 4, 3, 1));
///
/// // Or larger:
/// let r = shuffle!(x, [1, 3, 2, 2, 1, 3, 2, 2]);
/// assert_eq!(r, i32x8::new(2, 4, 3, 3, 2, 4, 3, 3));
/// // At most 2 * the number of lanes in the input vector.
///
/// // It also allows reordering elements of two vectors:
/// let y = i32x4::new(5, 6, 7, 8);
/// let r = shuffle!(x, y, [4, 0, 5, 1]);
/// assert_eq!(r, i32x4::new(5, 1, 6, 2));
/// // And this can be used to construct larger or smaller
/// // vectors as well.
/// # }
/// ```
#[macro_export]
macro_rules! shuffle {
    ($vec0:expr, $vec1:expr, [$l0:expr, $l1:expr]) => {
        unsafe {
            $crate::simd::__shuffle_vector2(
                $vec0, $vec1,
                [$l0, $l1]
            )
        }
    };
    ($vec0:expr, $vec1:expr, [$l0:expr, $l1:expr, $l2:expr, $l3:expr]) => {
        unsafe {
            $crate::simd::__shuffle_vector4(
                $vec0, $vec1,
                [$l0, $l1, $l2, $l3]
            )
        }
    };
    ($vec0:expr, $vec1:expr,
     [$l0:expr, $l1:expr, $l2:expr, $l3:expr,
      $l4:expr, $l5:expr, $l6:expr, $l7:expr]) => {
        unsafe {
            $crate::simd::__shuffle_vector8(
                $vec0, $vec1,
                [$l0, $l1, $l2, $l3,
                 $l4, $l5, $l6, $l7]
            )
        }
    };
    ($vec0:expr, $vec1:expr,
     [$l0:expr, $l1:expr, $l2:expr, $l3:expr,
      $l4:expr, $l5:expr, $l6:expr, $l7:expr,
      $l8:expr, $l9:expr, $l10:expr, $l11:expr,
      $l12:expr, $l13:expr, $l14:expr, $l15:expr]) => {
        unsafe {
            $crate::simd::__shuffle_vector16(
                $vec0, $vec1,
                [$l0, $l1, $l2, $l3,
                 $l4, $l5, $l6, $l7,
                 $l8, $l9, $l10, $l11,
                 $l12, $l13, $l14, $l15]
            )
        }
    };
    ($vec0:expr, $vec1:expr,
     [$l0:expr, $l1:expr, $l2:expr, $l3:expr,
      $l4:expr, $l5:expr, $l6:expr, $l7:expr,
      $l8:expr, $l9:expr, $l10:expr, $l11:expr,
      $l12:expr, $l13:expr, $l14:expr, $l15:expr,
      $l16:expr, $l17:expr, $l18:expr, $l19:expr,
      $l20:expr, $l21:expr, $l22:expr, $l23:expr,
      $l24:expr, $l25:expr, $l26:expr, $l27:expr,
      $l28:expr, $l29:expr, $l30:expr, $l31:expr]) => {
        unsafe {
            $crate::simd::__shuffle_vector32(
                $vec0, $vec1,
                [$l0, $l1, $l2, $l3,
                 $l4, $l5, $l6, $l7,
                 $l8, $l9, $l10, $l11,
                 $l12, $l13, $l14, $l15,
                 $l16, $l17, $l18, $l19,
                 $l20, $l21, $l22, $l23,
                 $l24, $l25, $l26, $l27,
                 $l28, $l29, $l30, $l31]
            )
        }
    };
    ($vec0:expr, $vec1:expr,
     [$l0:expr, $l1:expr, $l2:expr, $l3:expr,
      $l4:expr, $l5:expr, $l6:expr, $l7:expr,
      $l8:expr, $l9:expr, $l10:expr, $l11:expr,
      $l12:expr, $l13:expr, $l14:expr, $l15:expr,
      $l16:expr, $l17:expr, $l18:expr, $l19:expr,
      $l20:expr, $l21:expr, $l22:expr, $l23:expr,
      $l24:expr, $l25:expr, $l26:expr, $l27:expr,
      $l28:expr, $l29:expr, $l30:expr, $l31:expr,
      $l32:expr, $l33:expr, $l34:expr, $l35:expr,
      $l36:expr, $l37:expr, $l38:expr, $l39:expr,
      $l40:expr, $l41:expr, $l42:expr, $l43:expr,
      $l44:expr, $l45:expr, $l46:expr, $l47:expr,
      $l48:expr, $l49:expr, $l50:expr, $l51:expr,
      $l52:expr, $l53:expr, $l54:expr, $l55:expr,
      $l56:expr, $l57:expr, $l58:expr, $l59:expr,
      $l60:expr, $l61:expr, $l62:expr, $l63:expr]) => {
        unsafe {
            $crate::simd::__shuffle_vector64(
                $vec0, $vec1,
                [$l0, $l1, $l2, $l3,
                 $l4, $l5, $l6, $l7,
                 $l8, $l9, $l10, $l11,
                 $l12, $l13, $l14, $l15,
                 $l16, $l17, $l18, $l19,
                 $l20, $l21, $l22, $l23,
                 $l24, $l25, $l26, $l27,
                 $l28, $l29, $l30, $l31,
                 $l32, $l33, $l34, $l35,
                 $l36, $l37, $l38, $l39,
                 $l40, $l41, $l42, $l43,
                 $l44, $l45, $l46, $l47,
                 $l48, $l49, $l50, $l51,
                 $l52, $l53, $l54, $l55,
                 $l56, $l57, $l58, $l59,
                 $l60, $l61, $l62, $l63]
            )
        }
    };
    ($vec:expr, [$($l:expr),*]) => {
        shuffle!($vec, $vec, [$($l),*])
    }
}
