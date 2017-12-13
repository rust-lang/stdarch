extern crate cc;

use std::env;

fn main() {
    if env::var("TARGET")
        .expect("TARGET was not set")
        .contains("linux")
    {
        cc::Build::new()
            .file("src/runtime/linux/getauxval-wrapper.c")
            .compile("libgetauxval-wrapper");
    }
}
