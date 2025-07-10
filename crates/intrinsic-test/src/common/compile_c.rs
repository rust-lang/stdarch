#[derive(Clone)]
pub struct CompilationCommandBuilder {
    compiler: String,
    target: Option<String>,
    cxx_toolchain_dir: Option<String>,
    arch_flags: Vec<String>,
    optimization: String,
    include_paths: Vec<String>,
    project_root: Option<String>,
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

impl CompilationCommandBuilder {
    pub fn into_cpp_compilation(self) -> CppCompilation {
        let mut cpp_compiler = std::process::Command::new(self.compiler);

        if let Some(project_root) = self.project_root {
            cpp_compiler.current_dir(project_root);
        }

        let flags = std::env::var("CPPFLAGS").unwrap_or("".into());
        cpp_compiler.args(flags.split_whitespace());

        cpp_compiler.arg(format!("-march={}", self.arch_flags.join("+")));

        cpp_compiler.arg(format!("-O{}", self.optimization));

        cpp_compiler.args(self.extra_flags);

        if let Some(target) = &self.target {
            cpp_compiler.arg(format!("--target={target}"));
        }

        if let (Some(linker), Some(cxx_toolchain_dir)) = (&self.linker, &self.cxx_toolchain_dir) {
            cpp_compiler.args(
                self.include_paths
                    .iter()
                    .map(|path| "--include-directory=".to_string() + cxx_toolchain_dir + path),
            );

            CppCompilation::CustomLinker {
                cpp_compiler,
                linker: linker.to_owned(),
            }
        } else {
            CppCompilation::Simple(cpp_compiler)
        }
    }
}

pub enum CppCompilation {
    Simple(std::process::Command),
    CustomLinker {
        cpp_compiler: std::process::Command,
        linker: String,
    },
}

fn clone_command(command: &std::process::Command) -> std::process::Command {
    let mut cmd = std::process::Command::new(command.get_program());
    if let Some(current_dir) = command.get_current_dir() {
        cmd.current_dir(current_dir);
    }
    cmd.args(command.get_args());

    for (key, val) in command.get_envs() {
        cmd.env(key, val.unwrap_or_default());
    }

    cmd
}

impl CppCompilation {
    pub fn run(&self, inputs: &[String], output: &str) -> std::io::Result<std::process::Output> {
        match self {
            CppCompilation::Simple(command) => {
                let mut cmd = clone_command(command);
                cmd.args(inputs);
                cmd.args(["-o", output]);

                cmd.output()
            }
            CppCompilation::CustomLinker {
                cpp_compiler,
                linker,
            } => {
                let object_file = &format!("{output}.o");

                // Build an object file using the cpp compiler.
                let mut cmd = clone_command(cpp_compiler);
                cmd.args(inputs);
                cmd.args(["-c", "-o", object_file]);

                let cpp_output = cmd.output()?;
                if !cpp_output.status.success() {
                    error!("c++ compilaton failed");
                    return Ok(cpp_output);
                }

                trace!("using custom linker");

                // Use the custom linker to turn the object file into an executable.
                let mut cmd = std::process::Command::new(linker);
                cmd.args([object_file, "-o", output]);

                if let Some(current_dir) = cpp_compiler.get_current_dir() {
                    cmd.current_dir(current_dir);
                }

                for (key, val) in cpp_compiler.get_envs() {
                    cmd.env(key, val.unwrap_or_default());
                }

                let linker_output = cmd.output()?;
                if !linker_output.status.success() {
                    error!("custom linker failed");
                    return Ok(linker_output);
                }

                trace!("removing {object_file}");
                let object_file_path = match cpp_compiler.get_current_dir() {
                    Some(current_dir) => &format!("{}/{object_file}", current_dir.display()),
                    None => object_file,
                };

                std::fs::remove_file(object_file_path)?;

                Ok(cpp_output)
            }
        }
    }
}
