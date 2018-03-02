//! Implements portable arithmetic vector reductions.

macro_rules! impl_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Lane-wise addition of the vector elements.
            #[inline(always)]
            pub fn add(self) -> $elem_ty {
                let mut r = self.extract(0);
                for i in 1..$id::lanes() {
                    r += self.extract(i);
                }
                r
            }
            /// Lane-wise substraction of the vector elements.
            #[inline(always)]
            pub fn sub(self) -> $elem_ty {
                let mut r = self.extract(0);
                for i in 1..$id::lanes() {
                    r -= self.extract(i);
                }
                r
            }
            /// Lane-wise multiplication of the vector elements.
            #[inline(always)]
            pub fn mul(self) -> $elem_ty {
                let mut r = self.extract(0);
                for i in 1..$id::lanes() {
                    r *= self.extract(i);
                }
                r
            }
            /// Lane-wise division of the vector elements.
            #[inline(always)]
            pub fn div(self) -> $elem_ty {
                let mut r = self.extract(0);
                for i in 1..$id::lanes() {
                    r /= self.extract(i);
                }
                r
            }
            /// Lane-wise remainder of the vector elements.
            #[inline(always)]
            pub fn rem(self) -> $elem_ty {
                let mut r = self.extract(0);
                for i in 1..$id::lanes() {
                    r %= self.extract(i);
                }
                r
            }
        }
    }
}

#[cfg(test)]
macro_rules! test_arithmetic_reductions {
    ($id:ident, $elem_ty:ident) => {

        fn alternating(x: usize) -> ::coresimd::simd::$id {
            use ::coresimd::simd::$id;
            let mut v = $id::splat(1 as $elem_ty);
            for i in 0..$id::lanes() {
                if i % x == 0 {
                    v = v.replace(i, 2 as $elem_ty);
                }
            }
            v
        }

        #[test]
        fn add() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.add(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.add(), $id::lanes() as $elem_ty);
            let v = alternating(2);
            eprintln!("{:?}", v);
            assert_eq!(v.add(), ($id::lanes() / 2 + $id::lanes()) as $elem_ty);
        }
        #[test]
        fn sub() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.sub(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            let v = v.replace(0, $id::lanes() as $elem_ty);
            assert_eq!(v.sub(), 1 as $elem_ty);
        }

        #[test]
        fn mul() {
            use ::coresimd::simd::$id;
            let v = $id::splat(0 as $elem_ty);
            assert_eq!(v.mul(), 0 as $elem_ty);
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.mul(), 1 as $elem_ty);
            let f = match $id::lanes() {
                64 => 16,
                32 => 8,
                16 => 4,
                _ => 2,
            };
            let v = alternating(f);
            eprintln!("{:?}", v);
            assert_eq!(v.mul(), (2_usize.pow(($id::lanes() / f) as u32) as $elem_ty));
        }
        #[test]
        fn div() {
            use ::coresimd::simd::$id;
            let v = $id::splat(1 as $elem_ty);
            assert_eq!(v.div(), 1 as $elem_ty);
            let f = match $id::lanes() {
                64 => 16,
                32 => 8,
                16 => 4,
                _ => 2,
            };
            let v = alternating(f);
            let v = v.replace(0, 2_usize.pow(($id::lanes() / f) as u32) as $elem_ty);
            eprintln!("{:?}", v);
            assert_eq!(v.div(), 2 as $elem_ty);
        }

        #[test]
        fn rem() {
            use ::coresimd::simd::$id;
            let v = $id::splat(2 as $elem_ty);
            let v = v.replace(0, 3 as $elem_ty);
            assert_eq!(v.rem(), 1 as $elem_ty);
            let v = $id::splat(3 as $elem_ty);
            let v = v.replace(0, 2 as $elem_ty);
            assert_eq!(v.rem(), 2 as $elem_ty);
        }
    }
}
