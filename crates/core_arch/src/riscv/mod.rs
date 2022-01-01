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

/// Loads memory from hypervisor by signed byte integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLV.B` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlv_b(src: *const i8) -> i8 {
    let value: i8;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x600", out(reg) value, in(reg) src);
    value
}

/// Loads memory from hypervisor by unsigned byte integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLV.BU` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlv_bu(src: *const u8) -> u8 {
    let value: u8;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x601", out(reg) value, in(reg) src);
    value
}

/// Loads memory from hypervisor by signed half integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLV.H` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlv_h(src: *const i16) -> i16 {
    let value: i16;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x640", out(reg) value, in(reg) src);
    value
}

/// Loads memory from hypervisor by unsigned half integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLV.HU` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlv_hu(src: *const u16) -> u16 {
    let value: u16;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x641", out(reg) value, in(reg) src);
    value
}

/// Accesses instruction from hypervisor by unsigned half integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// the memory being read must be executable in both stages of address translation,
/// but read permission is not required.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLVX.HU` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlvx_hu(src: *const u16) -> u16 {
    let insn: u16;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x643", out(reg) insn, in(reg) src);
    insn
}

/// Loads memory from hypervisor by signed word integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLV.W` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlv_w(src: *const i32) -> i32 {
    let value: i32;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x680", out(reg) value, in(reg) src);
    value
}

/// Accesses instruction from hypervisor by unsigned word integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// the memory being read must be executable in both stages of address translation,
/// but read permission is not required.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HLVX.WU` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hlvx_wu(src: *const u32) -> u32 {
    let insn: u32;
    asm!(".insn i 0x73, 0x4, {}, {}, 0x683", out(reg) insn, in(reg) src);
    insn
}

/// Stores memory from hypervisor by byte integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HSV.B` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hsv_b(dst: *mut i8, src: i8) {
    asm!(".insn r 0x73, 0x4, 0x31, x0, {}, {}", in(reg) dst, in(reg) src);
}

/// Stores memory from hypervisor by half integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HSV.H` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hsv_h(dst: *mut i16, src: i16) {
    asm!(".insn r 0x73, 0x4, 0x33, x0, {}, {}", in(reg) dst, in(reg) src);
}

/// Stores memory from hypervisor by word integer
///
/// This instruction performs an explicit memory access as though `V=1`;
/// i.e., with the address translation and protection, and the endianness, that apply to memory
/// accesses in either VS-mode or VU-mode.
///
/// # Unsafety
///
/// This function accesses the virtual supervisor or user via a `HSV.W` instruction which is effectively
/// an unreference to any memory address, thus is wrapped into an unsafe function.
#[inline]
pub unsafe fn hsv_w(dst: *mut i32, src: i32) {
    asm!(".insn r 0x73, 0x4, 0x35, x0, {}, {}", in(reg) dst, in(reg) src);
}
