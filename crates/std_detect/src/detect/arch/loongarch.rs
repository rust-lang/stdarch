//! Run-time feature detection on LoongArch.

features! {
    @TARGET: loongarch;
    @CFG: target_arch = "loongarch64";
    @MACRO_NAME: is_loongarch_feature_detected;
    @MACRO_ATTRS:
    /// Checks if `loongarch` feature is enabled.
    /// Supported arguments are:
    ///
    /// * `"f"`
    /// * `"d"`
    /// * `"frecipe"`
    /// * `"lsx"`
    /// * `"lasx"`
    /// * `"lbt"`
    /// * `"lvz"`
    /// * `"ual"`
    #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")]
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] f: "f";
    /// F
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] d: "d";
    /// D
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] frecipe: "frecipe";
    /// Frecipe
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] lsx: "lsx";
    /// LSX
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] lasx: "lasx";
    /// LASX
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] lbt: "lbt";
    /// LBT
    @FEATURE: #[stable(feature = "stdarch_loongarch_feature", since = "1.88.0")] lvz: "lvz";
    /// LVZ
    @FEATURE: #[unstable(feature = "stdarch_loongarch_feature_detection", issue = "117425")] ual: "ual";
    /// UAL
}
