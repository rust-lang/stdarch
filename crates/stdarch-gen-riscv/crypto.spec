// This code is automatically generated. DO NOT MODIFY.
// ```
// cargo run -p stdarch-gen-riscv
// ```

name = aes32dsi
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2, const int bs
extensions = zknd

name = aes32dsmi
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2, const int bs
extensions = zknd

name = aes64ds
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zknd

name = aes64dsm
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zknd

name = aes64im
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1
extensions = zknd

name = aes64ks1i
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, const int rnum
extensions = zknd, zkne

name = aes64ks2
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zknd, zkne

name = aes32esi
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2, const int bs
extensions = zkne

name = aes32esmi
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2, const int bs
extensions = zkne

name = aes64es
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zkne

name = aes64esm
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1, uint64_t rs2
extensions = zkne

name = sha256sig0
arch = both
return_type = uint32_t
arguments = uint32_t rs1
extensions = zknh

name = sha256sig1
arch = both
return_type = uint32_t
arguments = uint32_t rs1
extensions = zknh

name = sha256sum0
arch = both
return_type = uint32_t
arguments = uint32_t rs1
extensions = zknh

name = sha256sum1
arch = both
return_type = uint32_t
arguments = uint32_t rs1
extensions = zknh

name = sha512sig0h
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zknh

name = sha512sig0l
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zknh

name = sha512sig1h
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zknh

name = sha512sig1l
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zknh

name = sha512sum0r
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zknh

name = sha512sum1r
arch = rv32
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2
extensions = zknh

name = sha512sig0
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1
extensions = zknh

name = sha512sig1
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1
extensions = zknh

name = sha512sum0
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1
extensions = zknh

name = sha512sum1
arch = rv64
return_type = uint64_t
arguments = uint64_t rs1
extensions = zknh

name = sm3p0
arch = both
return_type = uint32_t
arguments = uint32_t rs1
extensions = zksh

name = sm3p1
arch = both
return_type = uint32_t
arguments = uint32_t rs1
extensions = zksh

name = sm4ed
arch = both
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2, const int bs
extensions = zksed

name = sm4ks
arch = both
return_type = uint32_t
arguments = uint32_t rs1, uint32_t rs2, const int bs
extensions = zksed
