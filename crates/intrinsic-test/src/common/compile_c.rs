use std::path::PathBuf;

#[derive(Clone)]
pub struct CompilationCommandBuilder {
    compiler: String,
    target: Option<String>,
    cxx_toolchain_dir: Option<String>,
    arch_flags: Vec<String>,
    optimization: String,
    include_paths: Vec<String>,
    project_root: Option<String>,
    output: String,
    linker: Option<String>,
    extra_flags: Vec<String>,
}

impl CompilationCommandBuilder {
    pub fn new() -> Self {
        Self {
            compiler: String::new(),
            target: None,
            cxx_toolchain_dir: None,
            arch_flags: Vec::new(),
            optimization: "2".to_string(),
            include_paths: Vec::new(),
            project_root: None,
            output: String::new(),
            linker: None,
            extra_flags: Vec::new(),
        }
    }

    pub fn set_compiler(mut self, compiler: &str) -> Self {
        self.compiler = compiler.to_string();
        self
    }

    pub fn set_target(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    pub fn set_cxx_toolchain_dir(mut self, path: Option<&str>) -> Self {
        self.cxx_toolchain_dir = path.map(|p| p.to_string());
        self
    }

    pub fn add_arch_flags(mut self, flags: Vec<&str>) -> Self {
        let mut new_arch_flags = flags.into_iter().map(|v| v.to_string()).collect();
        self.arch_flags.append(&mut new_arch_flags);

        self
    }

    pub fn set_opt_level(mut self, optimization: &str) -> Self {
        self.optimization = optimization.to_string();
        self
    }

    /// Sets a list of include paths for compilation.
    /// The paths that are passed must be relative to the
    /// "cxx_toolchain_dir" directory path.
    pub fn set_include_paths(mut self, paths: Vec<&str>) -> Self {
        self.include_paths = paths.into_iter().map(|path| path.to_string()).collect();
        self
    }

    /// Sets the root path of all the generated test files.
    pub fn set_project_root(mut self, path: &str) -> Self {
        self.project_root = Some(path.to_string());
        self
    }

    pub fn set_linker(mut self, linker: String) -> Self {
        self.linker = Some(linker);
        self
    }

    pub fn add_extra_flags(mut self, flags: Vec<&str>) -> Self {
        let mut flags: Vec<String> = flags.into_iter().map(|f| f.to_string()).collect();
        self.extra_flags.append(&mut flags);
        self
    }

    pub fn add_extra_flag(self, flag: &str) -> Self {
        self.add_extra_flags(vec![flag])
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum CompilationCommand {
    Simple(std::process::Command),
    CustomLinker {
        cmd: std::process::Command,
        linker: std::process::Command,
        cleanup: PathBuf,
    },
}

impl CompilationCommand {
    pub fn command_mut(&mut self) -> &mut std::process::Command {
        match self {
            CompilationCommand::Simple(command) => command,
            CompilationCommand::CustomLinker { cmd, .. } => cmd,
        }
    }

    pub fn output(self) -> std::io::Result<std::process::Output> {
        match self {
            CompilationCommand::Simple(mut cmd) => cmd.output(),
            CompilationCommand::CustomLinker {
                mut cmd,
                mut linker,
                cleanup,
            } => {
                let output = cmd.output()?;

                linker.current_dir("c_programs");

                if log::log_enabled!(log::Level::Trace) {
                    linker.stdout(std::process::Stdio::inherit());
                    linker.stderr(std::process::Stdio::inherit());
                }

                if let Err(e) = linker.output() {
                    panic!(
                        "Failed running custom linker {:?}:\n{e:?}",
                        linker.get_program(),
                    );
                }
                if cleanup.exists() {
                    std::fs::remove_file(cleanup)?;
                }

                Ok(output)
            }
        }
    }
}

impl CompilationCommandBuilder {
    pub fn into_command(self) -> CompilationCommand {
        let project_root = self.project_root.unwrap_or_default();
        let project_root_str = project_root.as_str();

        let mut cmd = std::process::Command::new(self.compiler);

        let flags = std::env::var("CPPFLAGS").unwrap_or("".into());
        cmd.args(flags.split_whitespace());

        cmd.arg(format!("-march={}", self.arch_flags.join("+")));

        cmd.arg(format!("-O{}", self.optimization));

        cmd.args(self.extra_flags);

        if let Some(target) = &self.target {
            cmd.arg(format!("--target={target}"));
        }

        if let (Some(linker), Some(cxx_toolchain_dir)) = (&self.linker, &self.cxx_toolchain_dir) {
            cmd.arg("-c");
            cmd.args(
                self.include_paths
                    .iter()
                    .map(|path| "--include-directory=".to_string() + cxx_toolchain_dir + path),
            );

            let output = "dummy_value";
            let mut linker_cmd = std::process::Command::new(linker);
            linker_cmd.arg(format!("{project_root_str}/{output}"));

            linker_cmd.arg("-o");
            linker_cmd.arg(format!("{project_root_str}/{}", self.output));

            let remove_path = PathBuf::from(format!("{project_root_str}/{output}"));

            CompilationCommand::CustomLinker {
                cmd,
                linker: linker_cmd,
                cleanup: remove_path,
            }
        } else {
            CompilationCommand::Simple(cmd)
        }
    }
}
