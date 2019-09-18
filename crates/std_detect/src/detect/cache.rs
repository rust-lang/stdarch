//! Caches run-time feature detection so that it only needs to be computed
//! once.

#![allow(dead_code)] // not used on all platforms

use crate::sync::atomic::Ordering;

#[cfg(target_pointer_width = "64")]
use crate::sync::atomic::AtomicU64;

#[cfg(target_pointer_width = "32")]
use crate::sync::atomic::AtomicU32;

/// Sets the `bit` of `x`.
#[inline]
const fn set_bit(x: u64, bit: u32) -> u64 {
    x | 1 << bit
}

/// Tests the `bit` of `x`.
#[inline]
const fn test_bit(x: u64, bit: u32) -> bool {
    x & (1 << bit) != 0
}

/// Unset the `bit of `x`.
#[inline]
const fn unset_bit(x: u64, bit: u32) -> u64 {
    x & !(1 << bit)
}

/// This type is used to initialize the cache
#[derive(Copy, Clone)]
pub(crate) struct Initializer(u64);

#[allow(clippy::use_self)]
impl Default for Initializer {
    fn default() -> Self {
        Initializer(0)
    }
}

// NOTE: the `debug_assert!` would catch that we do not add more Features than
// the one fitting our cache.
impl Initializer {
    /// Tests the `bit` of the cache.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn test(self, bit: u32) -> bool {
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        test_bit(self.0, bit)
    }

    /// Sets the `bit` of the cache.
    #[inline]
    pub(crate) fn set(&mut self, bit: u32) {
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        let v = self.0;
        self.0 = set_bit(v, bit);
    }

    /// Unsets the `bit` of the cache.
    #[inline]
    pub(crate) fn unset(&mut self, bit: u32) {
        debug_assert!(
            bit < CACHE_CAPACITY,
            "too many features, time to increase the cache size!"
        );
        let v = self.0;
        self.0 = unset_bit(v, bit);
    }
}

/// This global variable is a cache of the features supported by the CPU.
static CACHE: Cache = Cache::uninitialized();

/// Feature cache with capacity for `CACHE_CAPACITY` features.
///
/// Note: the last feature bit is used to represent an
/// uninitialized cache.
#[cfg(target_pointer_width = "64")]
struct Cache(AtomicU64);

#[cfg(target_pointer_width = "64")]
const UNINITIALIZED_VALUE: u64 = u64::max_value();

#[cfg(target_pointer_width = "64")]
const UNINITIALIZED_BIT: u32 = 63;

/// Maximum number of features that can be cached.
/// We reserve a single bit to check if the cache is initialized
#[cfg(target_pointer_width = "64")]
const CACHE_CAPACITY: u32 = 63;

#[cfg(target_pointer_width = "64")]
#[allow(clippy::use_self)]
impl Cache {
    /// Creates an uninitialized cache.
    #[allow(clippy::declare_interior_mutable_const)]
    const fn uninitialized() -> Self {
        Cache(AtomicU64::new(UNINITIALIZED_VALUE))
    }

    /// Is the `bit` in the cache set?
    #[inline]
    fn test(&self, bit: u32) -> bool {
        let cache_value = CACHE.0.load(Ordering::Relaxed);
        if !test_bit(cache_value, UNINITIALIZED_BIT) {
            test_bit(cache_value, bit)
        } else {
            initialize(bit)
        }
    }

    /// Initializes the cache.
    #[inline]
    fn initialize(&self, value: Initializer) {
        let value = unset_bit(value.0, UNINITIALIZED_BIT);
        self.0.store(value, Ordering::Relaxed);
    }
}

/// Feature cache with capacity for `CACHE_CAPACITY` features.
///
/// Note: the last feature bit is used to represent an
/// uninitialized cache.
#[cfg(target_pointer_width = "32")]
struct Cache(AtomicU32, AtomicU32);

#[cfg(target_pointer_width = "32")]
const UNINITIALIZED_VALUE: u32 = u32::max_value();

#[cfg(target_pointer_width = "32")]
const UNINITIALIZED_BIT: u32 = 31;

/// Maximum number of features that can be cached.
/// We reserve a single bit in each of the two atomic values
/// to check if the cache is initialized
#[cfg(target_pointer_width = "32")]
const CACHE_CAPACITY: u32 = 62;

#[cfg(target_pointer_width = "32")]
impl Cache {
    /// Creates an uninitialized cache.
    const fn uninitialized() -> Self {
        Cache(
            AtomicU32::new(UNINITIALIZED_VALUE),
            AtomicU32::new(UNINITIALIZED_VALUE),
        )
    }

    /// Is the `bit` in the cache set?
    #[inline]
    fn test(&self, bit: u32) -> bool {
        if bit < 31 {
            let cache_value = CACHE.0.load(Ordering::Relaxed);
            if !test_bit(cache_value, UNINITIALIZED_BIT) {
                test_bit(cache_value, bit)
            } else {
                initializer(bit)
            }
        } else {
            let cache_value = CACHE.1.load(Ordering::Relaxed);
            if !test_bit(cache_value, UNINITIALIZED_BIT) {
                test_bit(cache_value, bit - 31)
            } else {
                initializer(bit)
            }
        }
    }

    /// Initializes the cache.
    #[inline]
    fn initialize(&self, value: Initializer) {
        let lo: u32 = unset_bit(value.0, UNINITIALIZED_BIT) as u32;
        let hi: u32 = unset_bit(value.0 >> 31, UNINITIALIZED_BIT) as u32;
        self.0.store(lo, Ordering::Relaxed);
        self.1.store(hi, Ordering::Relaxed);
    }
}
cfg_if! {
    if #[cfg(feature = "std_detect_env_override")] {
        #[inline(never)]
        fn initialize(bit: u32) -> bool {
            let mut value = crate::detect::os::detect_features();
            if let Ok(disable) = crate::env::var("RUST_STD_DETECT_UNSTABLE") {
                for v in disable.split(" ") {
                    let _ = super::Feature::from_str(v).map(|v| value.unset(v as u32));
                }
            }
            CACHE.initialize(value);
            test_bit(value.0, bit)
        }
    } else {
        #[inline]
        fn initialize(bit: u32) -> bool {
            let value = crate::detect::os::detect_features();
            CACHE.initialize(value);
            test_bit(value.0, bit)
        }
    }
}

/// Tests the `bit` of the storage. If the storage has not been initialized,
/// initializes it with the result of `os::detect_features()`.
///
/// On its first invocation, it detects the CPU features and caches them in the
/// `CACHE` global variable as an `AtomicU64`.
///
/// It uses the `Feature` variant to index into this variable as a bitset. If
/// the bit is set, the feature is enabled, and otherwise it is disabled.
///
/// If the feature `std_detect_env_override` is enabled looks for the env
/// variable `RUST_STD_DETECT_UNSTABLE` and uses its its content to disable
/// Features that would had been otherwise detected.
#[inline]
pub(crate) fn test(bit: u32) -> bool {
    CACHE.test(bit)
}
