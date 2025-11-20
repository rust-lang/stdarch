use std::collections::HashSet;
use std::sync::LazyLock;

#[macro_export]
macro_rules! feature_present {
    ($feat:tt) => {
        ::std::cfg_select! {
            any(target_arch = "x86", target_arch = "x86_64") => ::std::arch::is_x86_feature_detected!($feat),
            target_arch = "arm" => ::std::arch::is_arm_feature_detected!($feat),
            any(target_arch = "aarch64", target_arch = "arm64ec") => ::std::arch::is_aarch64_feature_detected!($feat),
            target_arch = "powerpc" => ::std::arch::is_powerpc_feature_detected!($feat),
            target_arch = "powerpc64" => ::std::arch::is_powerpc64_feature_detected!($feat),
            any(target_arch = "loongarch32", target_arch = "loongarch64") => ::std::arch::is_loongarch_feature_detected!($feat),
            target_arch = "s390x" => ::std::arch::is_s390x_feature_detected!($feat),
        }
    }
}

pub static SKIPPED_FUNCTIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    option_env!("STDARCH_TEST_SKIP_FUNCTION")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .collect()
});
pub static SKIPPED_FEATURES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    option_env!("STDARCH_TEST_SKIP_FEATURE")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .collect()
});

#[macro_export]
macro_rules! simd_test {
    attr($($feat:tt),*) ($( #[$meta:meta] )* $(unsafe)? fn $name:ident () $body:block) => {
        #[allow(non_snake_case)]
        #[test]
        $( #[$meta] )* 
        fn $name () {
            let mut missing_features = ::std::vec::Vec::new();
            $(
                if !$crate::feature_present!($feat) {
                    missing_features.push($feat);
                }
            )*

            if $crate::SKIPPED_FUNCTIONS.contains(stringify!($name))
                || [$($feat),*]
                    .iter()
                    .any(|&feat| $crate::SKIPPED_FEATURES.contains(feat))
            {
                println!("Skipped {}", stringify!($name));
                return;
            }

            if missing_features.is_empty() {
                unsafe { $name() };
            } else {
                $crate::assert_skip_test_ok(stringify!($name), &missing_features);
            }

            #[target_feature($(enable = $feat),*)]
            unsafe fn $name () $body
        }
    };
}
