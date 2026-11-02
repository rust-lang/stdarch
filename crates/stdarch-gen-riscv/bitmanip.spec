// This code is automatically generated. DO NOT MODIFY.
// ```
// cargo run -p stdarch-gen-riscv
// ```

name = clz_32
arch = both
return_type = unsigned
arguments = uint32_t x
extensions = zbb

name = ctz_32
arch = both
return_type = unsigned
arguments = uint32_t x
extensions = zbb

name = cpop_32
arch = both
return_type = unsigned
arguments = uint32_t x
extensions = zbb

name = orc_b_32
arch = both
return_type = uint32_t
arguments = uint32_t x
extensions = zbb

name = clz_64
arch = rv64
return_type = unsigned
arguments = uint64_t x
extensions = zbb

name = ctz_64
arch = rv64
return_type = unsigned
arguments = uint64_t x
extensions = zbb

name = cpop_64
arch = rv64
return_type = unsigned
arguments = uint64_t x
extensions = zbb

name = orc_b_64
arch = rv64
return_type = uint64_t
arguments = uint64_t x
extensions = zbb

name = ror_32
arch = both
return_type = uint32_t
arguments = uint32_t x, uint32_t shamt
extensions = zbb, zbkb

name = rol_32
arch = both
return_type = uint32_t
arguments = uint32_t x, uint32_t shamt
extensions = zbb, zbkb

name = rev8_32
arch = both
return_type = uint32_t
arguments = uint32_t x
extensions = zbb, zbkb

name = ror_64
arch = rv64
return_type = uint64_t
arguments = uint64_t x, uint32_t shamt
extensions = zbb, zbkb

name = rol_64
arch = rv64
return_type = uint64_t
arguments = uint64_t x, uint32_t shamt
extensions = zbb, zbkb

name = rev8_64
arch = rv64
return_type = uint64_t
arguments = uint64_t x
extensions = zbb, zbkb

name = brev8_32
arch = both
return_type = uint32_t
arguments = uint32_t x
extensions = zbkb

name = zip_32
arch = rv32
return_type = uint32_t
arguments = uint32_t x
extensions = zbkb

name = unzip_32
arch = rv32
return_type = uint32_t
arguments = uint32_t x
extensions = zbkb

name = brev8_64
arch = rv64
return_type = uint64_t
arguments = uint64_t x
extensions = zbkb

name = clmul_32
arch = both
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zbc, zbkc

name = clmulh_32
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zbc, zbkc

name = clmul_64
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zbc, zbkc

name = clmulh_64
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zbc, zbkc

name = clmulr_32
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zbc

name = clmulr_64
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zbc

name = xperm4_32
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zbkx

name = xperm8_32
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zbkx

name = xperm4_64
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zbkx

name = xperm8_64
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zbkx
