//! Caches run-time feature detection so that it only needs to be computed
//! once.

use core::sync::atomic::{AtomicU64, Ordering};

/// Sets the `bit` of `x`.
pub const fn set_bit(x: u64, bit: u32) -> u64 {
    x | 1 << bit
}

/// Tests the `bit` of `x`.
pub const fn test_bit(x: u64, bit: u32) -> bool {
    x & (1 << bit) != 0
}

/// Maximum number of features that can be cached.
const CACHE_CAPACITY: u32 = 63;

/// This type is used to initialize the cache
pub struct Initializer(u64);

impl Initializer {
    /// Creates a cleared cache.
    pub fn new() -> Self {
        Initializer(0)
    }
    /// Tests the `bit` of the cache.
    pub fn test(&self, bit: u32) -> bool {
        // FIXME: this way of making sure that the cache is large enough is
        // brittle.
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        test_bit(self.0, bit)
    }
    /// Sets the `bit` of the cache.
    pub fn set(&mut self, bit: u32) {
        // FIXME: this way of making sure that the cache is large enough is
        // brittle.
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        let v = self.0;
        self.0 = set_bit(v, bit);
    }
}

/// Feature cache with capacity for `CACHE_CAPACITY` features.
///
/// Note: the last feature bit is used to represent an
/// uninitialized cache.
struct Cache(AtomicU64);

/// This global variable is a cache of the features supported by the CPU.
static CACHE: Cache = Cache::uninitialized();

impl Cache {
    /// Creates an uninitialized cache.
    const fn uninitialized() -> Self {
        Cache(AtomicU64::new(u64::max_value()))
    }
    /// Is the cache uninitialized?
    pub fn is_uninitialized(&self) -> bool {
        self.0.load(Ordering::Relaxed) == u64::max_value()
    }

    /// Is the `bit` in the cache set?
    pub fn test(&self, bit: u32) -> bool {
        test_bit(CACHE.0.load(Ordering::Relaxed), bit)
    }

    pub fn set(&self, value: Initializer) {
        self.0.store(value.0, Ordering::Relaxed);
    }
}

/// Test the `bit` of the storage. If the storage has not been initialized,
/// initializes it with the result of `f()`.
///
/// On its first invocation, it detects the CPU features and caches them in the
/// `FEATURES` global variable as an `AtomicU64`.
///
/// It uses the `__Feature` variant to index into this variable as a bitset. If
/// the bit is set, the feature is enabled, and otherwise it is disabled.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
pub fn test<F>(bit: u32, f: F) -> bool
where
    F: FnOnce() -> Initializer,
{
    if CACHE.is_uninitialized() {
        CACHE.set(f());
    }
    CACHE.test(bit)
}
