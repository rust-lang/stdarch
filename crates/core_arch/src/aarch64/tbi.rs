extern crate alloc;
use crate::arch::asm;
use alloc::boxed::Box;
use core::convert::{AsMut, AsRef};
use core::ops::{Deref, DerefMut, Drop};

/// A TBI-enabled [Box].
///
/// This allows setting the top byte to arbitrary values via top-byte ignore (TBI).
#[derive(Debug)]
#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
pub struct TBIBox<T>(
    Option<Box<T>>,
    /* original pointer for deallocation: */ *mut T,
    // Without keeping a copy of the top byte around, it is difficult to cleanly extract the top byte from the
    // underlying pointer on-demand since `from_raw` requires consuming `self.0`.
    /* top byte for easy retrieval: */
    u8,
);

impl<T> TBIBox<T> {
    fn construct_new_tbibox(b: Box<T>, top_byte: u8) -> TBIBox<T> {
        // Pointer modificaton: this `Box` hasn't been exposed to the user yet, so the user cannot have a copy of the pointer with
        // an unset top byte; we use this chance to set the top byte before they can see it.
        let original_ptr: *mut T = Box::into_raw(b);
        let ptr =
            original_ptr.map_addr(|addr| (addr & 0x00ffffffffffffff) | ((top_byte as usize) << 56));
        unsafe {
            asm!(
                "/* pretend that we're memcpy'ing from {original_ptr} to {ptr}... */",
                original_ptr = in(reg) original_ptr,
                ptr = in(reg) ptr
            )
        };
        // Reconstruct the `Box` using the address with the new top byte and return that, wrapped as a TBIBox
        Self(Some(unsafe { Box::from_raw(ptr) }), original_ptr, top_byte)
    }

    fn interior_box(&self) -> &Box<T> {
        self.0.as_ref().expect("Interior Box can only be missing if this TBIBox has already returned it, in which case this TBIBox should not exist anymore!")
    }

    fn interior_box_mut(&mut self) -> &mut Box<T> {
        self.0.as_mut().expect("Interior Box can only be missing if this TBIBox has already returned it, in which case this TBIBox should not exist anymore!")
    }

    /// Construct a new [TBIBox] with the given value allocated on the heap, and the top byte of the allocation's address set to a given
    /// 8-bit value.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub fn new(value: T, top_byte: u8) -> Self {
        Self::construct_new_tbibox(Box::new(value), top_byte)
    }

    /// Returns a new [TBIBox] with the same contents as the provided [Box], with the top byte of the address it points to set to a given value.
    ///
    /// Note that the original [Box] and the returned [TBIBox] do not alias.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub fn new_from_box(b: &Box<T>, top_byte: u8) -> TBIBox<T>
    where
        T: Clone,
    {
        // Use a new Box with the same data; we can't reuse the provided Box as its address may have been exposed, so it may
        // result in aliasing pointers.
        Self::construct_new_tbibox(b.clone(), top_byte)
    }

    /// Construct a new [TBIBox] with cloned data, but with a new top byte set on the address.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub fn realloc_with_top_byte(&self, top_byte: u8) -> TBIBox<T>
    where
        T: Clone,
    {
        // Switch to a new `Box` with the same data - the address of the old allocation should be invalidated.
        Self::construct_new_tbibox(self.interior_box().clone(), top_byte)
    }

    /// Upgrades the provided [Box] to a [TBIBox].
    ///
    /// This reuses the existing allocation, and will inspect the address of the allocation to determine the top byte.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub fn from_box(b: Box<T>) -> TBIBox<T> {
        // Extract the pointer for inspection...
        let ptr: *mut T = Box::into_raw(b);
        // Reconstruct the `Box` using the address.
        // SAFETY: This is fine as we're reconstructing the Box with the exact same pointer.
        TBIBox(
            Some(unsafe { Box::from_raw(ptr) }),
            ptr,
            (ptr.addr() >> 56) as u8,
        )
    }

    /// Deconstructs the [TBIBox] to a raw pointer to its allocation,
    /// along with a `u8` value representing the top byte of the address
    /// as it was initially allocated with (see safety information).
    ///
    /// # Safety
    ///
    /// Care must be taken when handling the allocated memory if the top
    /// byte has been set using the [TBIBox]:
    /// - You may dereference the pointer following the usual rules.
    /// - You may **not** free the allocation directly, including via FFI.
    /// - If you use the pointer to construct a [Box], that [Box] is not suitable for upgrading back to a [TBIBox], and you must ensure that the [Box] is not dropped whilst Rust is managing its memory.
    ///
    /// To reconstruct the [TBIBox] for safely deallocating the memory, you **must** use `from_raw_ptr` with both values returned from this function.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub unsafe fn to_raw_ptr(mut self) -> (*mut T, u8) {
        (
            Box::into_raw(self.0.take().expect("Interior Box can only be missing if this TBIBox has already returned it, in which case this TBIBox should not exist anymore!")),
            (self.1.addr() >> 56) as u8
        )
    }

    /// Construct a [TBIBox] with a given pointer and provided original top byte,
    /// such as reconstructing one which was previous deconstructed with `to_raw_ptr`.
    ///
    /// # Safety
    ///
    /// This operation is only safe if the pointer has not been modified, and the original
    /// top byte is correct. Failure of either constraint will lead to Undefined Behavior
    /// when deallocating the memory, or when changing the top byte.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub unsafe fn from_raw_ptr(ptr: *mut T, original_top_byte: u8) -> Self {
        TBIBox(
            Some(unsafe { Box::from_raw(ptr) }),
            ptr.map_addr(|addr| (addr & 0x00ffffffffffffff) | ((original_top_byte as usize) << 56)),
            (ptr.addr() >> 56) as u8,
        )
    }

    /// Returns the top byte that is currently set on the address of the `TBIBox`.
    /// No introspection is performed on the actual value of the internal pointer.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    pub fn top_byte(&self) -> u8 {
        self.2
    }
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> Drop for TBIBox<T> {
    fn drop(&mut self) {
        if let Some(old_b) = self.0.as_mut() {
            // It is undefined behaviour since C11 to call `free` on a pointer which was not returned by a call to `malloc` et al.
            // Thus, reset the `Box` back to the original pointer, so that its destruction is well-defined.
            // First, construct the corrected `Box`...
            let mut b = unsafe { Box::from_raw(self.1) };
            // We **do not want Rust to free the current Box in self.0**. Thus, we first swap it out for the new Box, but keep it around...
            core::mem::swap(&mut b, old_b);
            // ... and then consume the old Box with `into_raw` so that Rust does not manage it anymore. This should avoid an invalid `free` call.
            let ptr = Box::into_raw(b);
            unsafe {
                asm!(
                    "/* pretend that we're memcpy'ing from {ptr} to {original_ptr}... */",
                    ptr = in(reg) ptr,
                    original_ptr = in(reg) self.1
                )
            };
        }
        // It is now safe to destroy self.
    }
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> Deref for TBIBox<T> {
    type Target = Box<T>;
    fn deref(&self) -> &Self::Target {
        &self.interior_box()
    }
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> DerefMut for TBIBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.interior_box_mut()
    }
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> AsMut<T> for TBIBox<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut *self.interior_box_mut()
    }
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> AsRef<T> for TBIBox<T> {
    fn as_ref(&self) -> &T {
        &*self.interior_box()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_tbi(b: u8) {
        let mut value = TBIBox::new(10u32, b);
        let top_byte = value.top_byte();
        assert_eq!(top_byte, b);
        assert_eq!(**value, 10);
        **value = 255;
        assert_eq!(**value, 255);
    }

    fn test_tbi_array(b: u8) {
        let mut value: TBIBox<[u32; 4]> = TBIBox::new([10, 255, 65535, 0xffffffff], b);
        let top_byte = value.top_byte();
        assert_eq!(top_byte, b);
        assert_eq!(**value, [10, 255, 65535, 0xffffffff]);
        value[0] = 25;
        assert_eq!(**value, [25, 255, 65535, 0xffffffff]);
    }

    #[test]
    fn tbi() {
        for i in 0x00..=0xff {
            println!("Top byte: {i:#x}");
            test_tbi(i);
        }
    }

    #[test]
    fn tbi_array() {
        for i in 0x00..=0xff {
            println!("Top byte: {i:#x}");
            test_tbi_array(i);
        }
    }
}
