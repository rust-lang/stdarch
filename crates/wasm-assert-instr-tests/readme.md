# assert_instr on WASM32

This crate uses `assert_instr` to verify the assembly of wasm functions.

# Set up

This crate needs a couple of tools installed:

1. Install latest version of `wasm-bindgen` CLI tools

```
git clone git@github.com:rustwasm/wasm-bindgen
cd wasm-bindgen
cargo install --path crates/cli

# This makes wasm-bindgen-test-runner the test runner for wasm32-unknown-unknown:
``` 

2. Install WABT

```
# MacOSX
brew install wabt

# From source: 
git clone --recursive https://github.com/WebAssembly/wabt
make -C wabt -j

# Add it to the path
PATH=$PATH:/wabt/bin
```

The `stdsimd-test` proc macro needs to be able to find these in the path. We
could add an environment variable to configure these.

3. Install Node

Using `nvm`, homebrew, or manually. The test runner needs to find a recent
enough node in the `PATH`:

```
# MacOSX
brew install node

# Other
curl https://nodejs.org/dist/v10.8.0/node-v10.8.0-linux-x64.tar.xz | tar xJf -
PATH=$PATH:/node-v10.8.0-linux-x64/bin
```

4. Compile and install linker shim

```
# In stdsimd/
cd ci
rustc lld-shim -o lld-shim
```

# Running the tests

This is how you can run the tests:

```
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER=PATH/TO/lld-shim \
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
cargo test --target=wasm32-unknown-unknown --release
```

you can also set the `CARGO_TARGET_WASM32_...` linker and test runner globally
with `export ...`.

To see the build fail, pass it `RUSTFLAGS="-C target-feature=+simd128"`.
