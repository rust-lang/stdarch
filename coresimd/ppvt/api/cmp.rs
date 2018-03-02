//! Lane-wise vector comparisons returning boolean vectors.

macro_rules! impl_cmp {
    ($id:ident, $bool_ty:ident) => {
        impl $id {
            /// Lane-wise equality comparison.
            #[inline(always)]
            pub fn eq(self, other: $id) -> $bool_ty {
                unsafe { simd_eq(self, other) }
            }

            /// Lane-wise inequality comparison.
            #[inline(always)]
            pub fn ne(self, other: $id) -> $bool_ty {
                unsafe { simd_ne(self, other) }
            }

            /// Lane-wise less-than comparison.
            #[inline(always)]
            pub fn lt(self, other: $id) -> $bool_ty {
                unsafe { simd_lt(self, other) }
            }

            /// Lane-wise less-than-or-equals comparison.
            #[inline(always)]
            pub fn le(self, other: $id) -> $bool_ty {
                unsafe { simd_le(self, other) }
            }

            /// Lane-wise greater-than comparison.
            #[inline(always)]
            pub fn gt(self, other: $id) -> $bool_ty {
                unsafe { simd_gt(self, other) }
            }

            /// Lane-wise greater-than-or-equals comparison.
            #[inline(always)]
            pub fn ge(self, other: $id) -> $bool_ty {
                unsafe { simd_ge(self, other) }
            }
        }
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! test_cmp {
    ($id:ident, $elem_ty:ident, $bool_ty:ident) => {
        #[test]
        fn cmp() {
            use ::coresimd::simd::*;

            let a = $id::splat(0 as $elem_ty);
            let b = $id::splat(1 as $elem_ty);

            let r = a.lt(b);
            let e = $bool_ty::splat(true);
            assert!(r == e);
            let r = a.le(b);
            assert!(r == e);

            let e = $bool_ty::splat(false);
            let r = a.gt(b);
            assert!(r == e);
            let r = a.ge(b);
            assert!(r == e);
            let r = a.eq(b);
            assert!(r == e);

            let mut a = a;
            let mut b = b;
            let mut e = e;
            for i in 0..$id::lanes() {
                if i % 2 == 0 {
                    a = a.replace(i, 0 as $elem_ty);
                    b = b.replace(i, 1 as $elem_ty);
                    e = e.replace(i, true);
                } else {
                    a = a.replace(i, 1 as $elem_ty);
                    b = b.replace(i, 0 as $elem_ty);
                    e = e.replace(i, false);
                }
            }
            let r = a.lt(b);
            assert!(r == e);
        }
    }
}
