//! Implements `Default` for vector types.

macro_rules! impl_default {
    ($id:ident, $elem_ty:ident) => {
        impl Default for $id {
            #[inline]
            fn default() -> Self {
                Self::splat($elem_ty::default())
            }
        }
    }
}

#[cfg(any(test_v16, test_v32, test_v64, test_v128, test_v256, test_v512))]
macro_rules! test_default {
    ($id:ident, $elem_ty:ident) => {
        #[test]
        fn default() {
            use ::coresimd::simd::*;
            use std::default::Default;
            let a = $id::default();
            for i in 0..$id::lanes() {
                assert_eq!(a.extract(i), $elem_ty::default());
            }
        }
    }
}
