//! Bit manipulation utilities.

pub fn test(x: usize, bit: u32) -> bool {
    debug_assert!(bit < 32, "bit index out-of-bounds");
    x & (1 << bit) != 0
}
