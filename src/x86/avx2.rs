use v256::*;

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi32(a: i32x8) -> i32x8 {
    unsafe { pabsd(a) }
}

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi16(a: i16x16) -> i16x16 {
    unsafe { pabsw(a) }
}

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_abs_epi8(a: i8x32) -> i8x32 {
    unsafe { pabsb(a) }
}

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi64(a: i64x4, b: i64x4) -> i64x4 {
    a + b
}

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi32(a: i32x8, b: i32x8) -> i32x8 {
    a + b
}

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi16(a: i16x16, b: i16x16) -> i16x16 {
    a + b
}

#[inline(always)]
#[target_feature = "+avx2"]
pub fn _mm256_add_epi8(a: i8x32, b: i8x32) -> i8x32 {
    a + b
}


#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx2.pabs.b"]
    fn pabsb(a: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx2.pabs.w"]
    fn pabsw(a: i16x16) -> i16x16;
    #[link_name = "llvm.x86.avx2.pabs.d"]
    fn pabsd(a: i32x8) -> i32x8;    
    #[link_name = "llvm.x86.avx2.padds_b"]
    fn paddsb(a:i8x32,b:i8x32) -> i8x32;


}


#[cfg(test)]
mod tests {
    use v256::*;
    use x86::avx2;
    use std;

    #[test]
    #[target_feature = "+avx2"]
    fn _mm_256_abs_epi32() {
        let a = i32x8::new(0, 1, -1, std::i32::MAX, 
                           std::i32::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi32(a);
        let e = i32x8::new(0, 1, 1, std::i32::MAX, 
                           (std::i32::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm_256_abs_epi16() {
        let a = i16x16::new(0, 1, -1, 2, 
                            -2, 3, -3, 4, 
                            -4, 5, -5, std::i16::MAX, 
                            std::i16::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi16(a);
        let e = i16x16::new(0, 1, 1, 2, 
                            2, 3, 3, 4, 
                            4, 5, 5, std::i16::MAX, 
                            (std::i16::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn _mm_256_abs_epi8() {
        let a = i8x32::new(0, 1, -1, 2, 
                           -2, 3, -3, 4, 
                           -4, 5, -5, std::i8::MAX, 
                           std::i8::MIN + 1, 100, -100, -32, 
                           0, 1, -1, 2,
                           -2, 3, -3, 4,
                           -4, 5, -5, std::i8::MAX, 
                           std::i8::MIN + 1, 100, -100, -32);
        let r = avx2::_mm256_abs_epi8(a);
        let e = i8x32::new(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, std::i8::MAX, (std::i8::MIN + 1).abs(), 100, 100, 32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, std::i8::MAX, (std::i8::MIN + 1).abs(), 100, 100, 32);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn __mm_256_add_eip64() {
        let a = i64x4::new(-10, 0, 100, 1_000_000_000);
        let b = i64x4::new(-1, 0, 1, 2);
        let r = avx2::_mm256_add_epi64(a, b);
        let e = i64x4::new(-11, 0, 101, 1_000_000_002);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn __mm_256_add_eip32() {
        let a = i32x8::new(-1, 0, 1, 2, 3, 4, 5, 6);
        let b = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = avx2::_mm256_add_epi32(a, b);
        let e = i32x8::new(0, 2, 4, 6, 8, 10, 12, 14);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn __mm_256_add_eip16() {
        let a = i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 
                            8, 9, 10, 11, 12, 13, 14, 15);
        let b = i16x16::new(0, 1, 2, 3, 4, 5, 6, 7, 
                            8, 9, 10, 11, 12, 13, 14, 15);
        let r = avx2::_mm256_add_epi16(a, b);
        let e = i16x16::new(0, 2, 4, 6, 8, 10, 12, 14,
                            16, 18, 20, 22, 24, 26, 28, 30);
        assert_eq!(r, e);
    }

    #[test]
    #[target_feature = "+avx2"]
    fn __mm_256_add_eip8() {
        let a = i8x32::new(0, 1, 2, 3, 4, 5, 6, 7, 
                           8, 9, 10, 11, 12, 13, 14, 15, 
                           16, 17, 18, 19, 20, 21, 22, 23, 
                           24, 25, 26, 27, 28, 29, 30, 31);
        let b = i8x32::new(0, 1, 2, 3, 4, 5, 6, 7, 
                           8, 9, 10, 11, 12, 13, 14, 15, 
                           16, 17, 18, 19, 20, 21, 22, 23, 
                           24, 25, 26, 27, 28, 29, 30, 31);
        let r = avx2::_mm256_add_epi8(a, b);
        let e = i8x32::new(0, 2, 4, 6, 8, 10, 12, 14, 16, 
                           18, 20, 22, 24, 26, 28, 30, 32, 
                           34, 36, 38, 40, 42, 44, 46, 48, 
                           50, 52, 54, 56, 58, 60,62);
        assert_eq!(r, e);
    }

}
