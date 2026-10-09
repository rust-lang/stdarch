//! RISC-V RV32 specific intrinsics

mod zb;
mod zk;

#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
pub use zb::*;

#[stable(feature = "riscv_zk_intrinsics", since = "CURRENT_RUSTC_VERSION")]
pub use zk::*;
