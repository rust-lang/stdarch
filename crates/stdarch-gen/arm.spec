/// Vector bitwise and
name = vand
fn = simd_and
arm = vand
aarch64 = and
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00
b = 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
e = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00
types = int*_t, uint*_t

/// Vector bitwise or (immediate, inclusive)
name = vorr
fn = simd_or
arm = vorr
aarch64 = orr
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
e = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
types = int*_t, uint*_t

/// Vector bitwise exclusive or (vector)
name = veor
fn = simd_xor
arm = veor
aarch64 = eor
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
e = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
types = int*_t, uint*_t

////////////////////
// equality
////////////////////

/// Compare bitwise Equal (vector)
name = vceq
fn = simd_eq
arm = cmeq
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint*_t, int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Floating-point compare equal
name = vceq
fn = simd_eq
arm = fcmeq
a = 1.2, 3.4, 5.6, 7.8
b = 1.2, 3.4, 5.6, 7.8
e = -1, -1, -1, -1, -1, -1
types = float16x4_t:uint16x4_t, float16x8_t:uint16x8_t, float32x2_t:uint32x2_t, float32x4_t:uint32x4_t, float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

// AARCH64 only variatns

/// Compare bitwise Equal (vector)
name = vceq
fn = simd_eq
aarch64 = cmeq
a = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
b = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint64x1_t, uint64x2_t, int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

////////////////////
// greater then
////////////////////

/// Compare signed greater than
name = vcgt
fn = simd_gt
arm = cmgt
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned highe
name = vcgt
fn = simd_gt
arm = cmhi
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint*_t

/// Floating-point compare greater than
name = vcgt
fn = simd_gt
arm = fcmgt
a = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
b = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
e = -1, -1, -1, -1, -1, -1
types = float16x4_t:uint16x4_t, float16x8_t:uint16x8_t, float32x2_t:uint32x2_t, float32x4_t:uint32x4_t, float64x1_t:uint64x1_t, float64x2_t:uint64x2_t


// AARCH64 only variatns

/// Compare signed greater than
name = vcgt
fn = simd_gt
aarch64 = cmgt
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

/// Compare unsigned highe
name = vcgt
fn = simd_gt
aarch64 = cmgt
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint64x1_t, uint64x2_t

////////////////////
// lesser then
////////////////////

/// Compare signed less than
name = vclt
fn = simd_lt
arm = cmgt
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned less than
name = vclt
fn = simd_lt
arm = cmhi
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint*_t

/// Floating-point compare less than
name = vclt
fn = simd_lt
arm = fcmgt
b = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
a = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
e = -1, -1, -1, -1, -1, -1
types = float16x4_t:uint16x4_t, float16x8_t:uint16x8_t, float32x2_t:uint32x2_t, float32x4_t:uint32x4_t, float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

// AARCH64 only variatns

/// Compare signed less than
name = vclt
fn = simd_gt
aarch64 = cmgt
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

/// Compare unsigned less than
name = vclt
fn = simd_gt
aarch64 = cmhi
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint64x1_t, uint64x2_t


////////////////////
// lesser then equals
////////////////////

/// Compare signed less than or equal
name = vcle
fn = simd_lt
arm = cmge
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned less than or equal
name = vcle
fn = simd_lt
arm = cmhs
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint*_t

/// Floating-point compare less than or equal
name = vcle
fn = simd_lt
arm = fcmge
a = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
b = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
e = -1, -1, -1, -1, -1, -1
types = float16x4_t:uint16x4_t, float16x8_t:uint16x8_t, float32x2_t:uint32x2_t, float32x4_t:uint32x4_t, float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

// AARCH64 only variatns

/// Compare signed less than or equal
name = vcle
fn = simd_gt
aarch64 = cmge
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

/// Compare unsigned less than or equal
name = vcle
fn = simd_gt
aarch64 = cmhs
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint64x1_t, uint64x2_t

////////////////////
// greater then equals
////////////////////

/// Compare signed greater than or equal
name = vcge
fn = simd_lt
arm = cmge
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int8x8_t:uint8x8_t, int8x16_t:uint8x16_t, int16x4_t:uint16x4_t, int16x8_t:uint16x8_t, int32x2_t:uint32x2_t, int32x4_t:uint32x4_t

/// Compare unsigned greater than or equal
name = vcge
fn = simd_lt
arm = cmhs
a = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
b = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint*_t

/// Floating-point compare greater than or equal
name = vcge
fn = simd_lt
arm = fcmge
a = 0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8
b = 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9 
e = -1, -1, -1, -1, -1, -1
types = float16x4_t:uint16x4_t, float16x8_t:uint16x8_t, float32x2_t:uint32x2_t, float32x4_t:uint32x4_t, float64x1_t:uint64x1_t, float64x2_t:uint64x2_t

// AARCH64 only variatns

/// Compare signed greater than or equal
name = vcge
fn = simd_gt
aarch64 = cmge
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = int64x1_t:uint64x1_t, int64x2_t:uint64x2_t

/// Compare unsigned greater than or equal
name = vcge
fn = simd_gt
aarch64 = cmhs
a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
b = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
e = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
types = uint64x1_t, uint64x2_t