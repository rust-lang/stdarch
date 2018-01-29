use x86::__m128i;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.aesni.aesdec"]
    fn aesdec(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesdeclast"]
    fn aesdeclast(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesenc"]
    fn aesenc(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesenclast"]
    fn aesenclast(a: __m128i, round_key: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aesimc"]
    fn aesimc(a: __m128i) -> __m128i;
    #[link_name = "llvm.x86.aesni.aeskeygenassist"]
    fn aeskeygenassist(a: __m128i, imm8: u8) -> __m128i;
}

/// Perform one round of an AES decryption flow on data (state) in `a`.
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesdec))]
pub unsafe fn _mm_aesdec_si128(a: __m128i, round_key: __m128i) -> __m128i {
    aesdec(a, round_key)
}

/// Perform the last round of an AES decryption flow on data (state) in `a`.
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesdeclast))]
pub unsafe fn _mm_aesdeclast_si128(a: __m128i, round_key: __m128i) -> __m128i {
    aesdeclast(a, round_key)
}

/// Perform one round of an AES encryption flow on data (state) in `a`.
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesenc))]
pub unsafe fn _mm_aesenc_si128(a: __m128i, round_key: __m128i) -> __m128i {
    aesenc(a, round_key)
}

/// Perform the last round of an AES encryption flow on data (state) in `a`.
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesenclast))]
pub unsafe fn _mm_aesenclast_si128(a: __m128i, round_key: __m128i) -> __m128i {
    aesenclast(a, round_key)
}

/// Perform the “InvMixColumns” transformation on `a`.
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aesimc))]
pub unsafe fn _mm_aesimc_si128(a: __m128i) -> __m128i {
    aesimc(a)
}

/// Assist in expanding the AES cipher key.
///
/// Assist in expanding the AES cipher key by computing steps towards generating
/// a round key for encryption cipher using data from `a` and an 8-bit round constant.
#[inline]
#[target_feature(enable = "aes")]
#[cfg_attr(test, assert_instr(aeskeygenassist))]
pub unsafe fn _mm_aeskeygenassist_si128(a: __m128i, imm8: u8) -> __m128i {
    aeskeygenassist(a, imm8)
}
