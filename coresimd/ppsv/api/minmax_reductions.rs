//! Implements portable arithmetic vector reductions.
#![allow(unused)]

macro_rules! impl_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Largest vector value.
            ///
            /// # Floating-point behvior
            ///
            /// If the vector contains only `NaN` values,
            /// the result is a `NaN`.
            ///
            /// Otherwise, if the vector contains `NaN` values, either the
            /// largest element of the vector or a `NaN` is returned.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn max(self) -> $elem_ty {
                use ::coresimd::simd_llvm::simd_reduce_max;
                unsafe {
                    simd_reduce_max(self)
                }
            }

            /// Largest vector value.
            ///
            /// # Floating-point behvior
            ///
            /// If the vector contains only `NaN` values,
            /// the result is a `NaN`.
            ///
            /// Otherwise, if the vector contains `NaN` values, either the
            /// largest element of the vector or a `NaN` is returned.
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_imports)]
            #[inline]
            pub fn max(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                use ::num::Float;
                use ::cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.max(self.extract(i));
                }
                x
            }

            /// Smallest vector value.
            ///
            /// # Floating-point behvior
            ///
            /// If the vector contains only `NaN` values,
            /// the result is a `NaN`.
            ///
            /// Otherwise, if the vector contains `NaN` values, either the
            /// smallest element of the vector or a `NaN` is returned.
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            pub fn min(self) -> $elem_ty {
                use ::coresimd::simd_llvm::simd_reduce_min;
                unsafe {
                    simd_reduce_min(self)
                }
            }

            /// # Floating-point behvior
            ///
            /// If the vector contains only `NaN` values,
            /// the result is a `NaN`.
            ///
            /// Otherwise, if the vector contains `NaN` values, either the
            /// smallest element of the vector or a `NaN` is returned.
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_imports)]
            #[inline]
            pub fn min(self) -> $elem_ty {
                // FIXME: broken on AArch64
                // https://bugs.llvm.org/show_bug.cgi?id=36796
                use ::num::Float;
                use ::cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.min(self.extract(i));
                }
                x
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_minmax_reductions {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn max() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.max(), 0 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.max(), 1 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.max(), 2 as $elem_ty);
        }

        #[test]
        fn min() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.min(), 0 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.min(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.min(), 1 as $elem_ty);
            let v = $id::splat(2 as $elem_ty);
            let v = v.replace(1, 1 as $elem_ty);
            assert_eq!(v.min(), 1 as $elem_ty);
        }
    }
}
