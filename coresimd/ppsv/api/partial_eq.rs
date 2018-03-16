//! Implements `PartialEq` for vector types.

macro_rules! impl_partial_eq {
    ($id:ident) => {
        impl PartialEq<$id> for $id {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                $id::eq(*self, *other).all()
            }
            #[inline]
            fn ne(&self, other: &Self) -> bool {
                $id::ne(*self, *other).all()
            }
        }
    }
}

#[cfg(any(test_v16, test_v32, test_v64, test_v128, test_v256, test_v512))]
macro_rules! test_partial_eq {
    ($id:ident, $true:expr, $false:expr) => {
        #[test]
        fn partial_eq() {
            use ::coresimd::simd::*;

            let a = $id::splat($false);
            let b = $id::splat($true);

            assert!(a != b);
            assert!(!(a == b));
            assert!(a == a);
            assert!(!(a != a));
        }
    }
}
