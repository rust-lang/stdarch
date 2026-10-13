//! RISC-V RV32 specific intrinsics

mod zb;
mod zk;

#[stable(feature = "stdarch_riscv_zb", since = "CURRENT_RUSTC_VERSION")]
pub use zb::*;

#[stable(feature = "stdarch_riscv_zk", since = "CURRENT_RUSTC_VERSION")]
pub use zk::*;
