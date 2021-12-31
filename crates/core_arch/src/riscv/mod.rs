//! RISC-V intrinsics

use crate::arch::asm;

/// Generates the `PAUSE` instruction
///
/// The PAUSE instruction is a HINT that indicates the current hart's rate of instruction retirement
/// should be temporarily reduced or paused. The duration of its effect must be bounded and may be zero.
#[inline]
pub fn pause() {
    unsafe { asm!(".insn i 0x0F, 0, x0, x0, 0x010", options(nomem, nostack)) }
}

/// Generates the `NOP` instruction
///
/// The NOP instruction does not change any architecturally visible state, except for
/// advancing the `pc` and incrementing any applicable performance counters.
#[inline]
pub fn nop() {
    unsafe { asm!("nop") }
}

/// Generates the `WFI` instruction
///
/// The WFI instruction provides a hint to the implementation that the current hart can be stalled
/// until an interrupt might need servicing. This instruction is a hint,
/// and a legal implementation is to simply implement WFI as a NOP.
#[inline]
pub unsafe fn wfi() {
    asm!("wfi")
}

/// Generates the `FENCE.I` instruction
///
/// A FENCE.I instruction ensures that a subsequent instruction fetch on a RISC-V hart will see
/// any previous data stores already visible to the same RISC-V hart.
///
/// FENCE.I does not ensure that other RISC-V harts' instruction fetches will observe the
/// local hart's stores in a multiprocessor system.
#[inline]
pub unsafe fn fence_i() {
    asm!("fence.i")
}

/// Generates the `SFENCE.VMA` instruction for given virtual address and address space
///
/// The fence orders only reads and writes made to leaf page table entries corresponding to
/// the virtual address in parameter `vaddr`, for the address space identified by integer parameter
/// `asid`. Accesses to global mappings are not ordered. The fence also invalidates all
/// address-translation cache entries that contain leaf page table entries corresponding to the
/// virtual address in parameter `vaddr` and that match the address space identified by integer
/// parameter `asid`, except for entries containing global mappings.
#[inline]
pub unsafe fn sfence_vma(vaddr: usize, asid: usize) {
    asm!("sfence.vma {}, {}", in(reg) vaddr, in(reg) asid)
}

/// Generates the `SFENCE.VMA` instruction for given virtual address
///
/// The fence orders only reads and writes made to leaf page table entries corresponding to
/// the virtual address in parameter `vaddr`, for all address spaces.
/// The fence also invalidates all address-translation cache entries that contain leaf page
/// table entries corresponding to the virtual address in parameter `vaddr`, for all address spaces.
#[inline]
pub unsafe fn sfence_vma_vaddr(vaddr: usize) {
    asm!("sfence.vma {}, x0", in(reg) vaddr)
}

/// Generates the `SFENCE.VMA` instruction for given address space
///
/// The fence orders all reads and writes made to any level of the page tables,
/// but only for the address space identified by integer parameter `asid`.
///
/// Accesses to global mappings are not ordered. The fence also invalidates all
/// address-translation cache entries matching the address space identified by integer
/// parameter `asid`, except for entries containing global mappings.
#[inline]
pub unsafe fn sfence_vma_asid(asid: usize) {
    asm!("sfence.vma x0, {}", in(reg) asid)
}

/// Generates the `SFENCE.VMA` instruction for all address spaces and virtual addresses
///
/// The fence orders all reads and writes made to any level of the page
/// tables, for all address spaces. The fence also invalidates all address-translation cache entries,
/// for all address spaces.
#[inline]
pub unsafe fn sfence_vma_all() {
    asm!("sfence.vma")
}

/// Generates the `SINVAL.VMA` instruction for given virtual address and address space
///
/// This instruction invalidates any address-translation cache entries that an
/// `SFENCE.VMA` instruction with the same values of `vaddr` and `asid` would invalidate.
#[inline]
pub unsafe fn sinval_vma(vaddr: usize, asid: usize) {
    // asm!("sinval.vma {}, {}", in(reg) vaddr, in(reg) asid)
    asm!(".insn r 0x73, 0, 0x0B, x0, {}, {}", in(reg) vaddr, in(reg) asid)
}

/// Generates the `SINVAL.VMA` instruction for given virtual address
///
/// This instruction invalidates any address-translation cache entries that an
/// `SFENCE.VMA` instruction with the same values of `vaddr` and `asid` would invalidate.
#[inline]
pub unsafe fn sinval_vma_vaddr(vaddr: usize) {
    asm!(".insn r 0x73, 0, 0x0B, x0, {}, x0", in(reg) vaddr)
}

/// Generates the `SINVAL.VMA` instruction for given address space
///
/// This instruction invalidates any address-translation cache entries that an
/// `SFENCE.VMA` instruction with the same values of `vaddr` and `asid` would invalidate.
#[inline]
pub unsafe fn sinval_vma_asid(asid: usize) {
    asm!(".insn r 0x73, 0, 0x0B, x0, x0, {}", in(reg) asid)
}

/// Generates the `SINVAL.VMA` instruction for all address spaces and virtual addresses
///
/// This instruction invalidates any address-translation cache entries that an
/// `SFENCE.VMA` instruction with the same values of `vaddr` and `asid` would invalidate.
#[inline]
pub unsafe fn sinval_vma_all() {
    asm!(".insn r 0x73, 0, 0x0B, x0, x0, x0")
}

/// Generates the `SFENCE.W.INVAL` instruction
///
/// This instruction guarantees that any previous stores already visible to the current RISC-V hart
/// are ordered before subsequent `SINVAL.VMA` instructions executed by the same hart.
#[inline]
pub unsafe fn sfence_w_inval() {
    asm!(".insn i 0x73, 0, x0, x0, 0x180")
}

/// Generates the `SFENCE.INVAL.IR` instruction
///
/// This instruction guarantees that any previous SINVAL.VMA instructions executed by the current hart
/// are ordered before subsequent implicit references by that hart to the memory-management data structures.
#[inline]
pub unsafe fn sfence_inval_ir() {
    asm!(".insn i 0x73, 0, x0, x0, 0x181")
}
