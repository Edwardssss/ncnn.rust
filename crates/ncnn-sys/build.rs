use std::{env, path::PathBuf};

fn main() {
    for key in [
        "NCNN_INSTALL_DIR",
        "NCNN_LINK_KIND",
        "NCNN_EXTRA_LIBS",
        "NCNN_EXTRA_LIB_DIRS",
        "NCNN_ALLOW_UNTESTED_VERSION",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed=native");
    let target = env::var("TARGET").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let abi = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    assert!(
        matches!(arch.as_str(), "x86_64" | "aarch64")
            && matches!(
                (os.as_str(), abi.as_str()),
                ("linux", "gnu") | ("windows", "msvc") | ("macos", "")
            ),
        "Provide an x86-64 or ARM64 SDK for Linux GNU, Windows MSVC, or macOS"
    );
    let prefix = env::var_os("NCNN_INSTALL_DIR")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .expect("Set NCNN_INSTALL_DIR to an installed ncnn 20261020 SDK (include/ncnn and lib)");
    let prefix = if prefix.is_absolute() {
        prefix
    } else {
        env::current_dir().unwrap().join(prefix)
    };
    assert!(
        prefix.is_dir(),
        "NCNN_INSTALL_DIR must name an existing installation directory"
    );
    let requested_kind = env::var("NCNN_LINK_KIND").ok();
    assert!(
        matches!(requested_kind.as_deref(), None | Some("static" | "dylib")),
        "NCNN_LINK_KIND must be static or dylib"
    );
    let include = prefix.join("include/ncnn");
    let lib_dirs = [prefix.join("lib"), prefix.join("lib64")];
    // MSVC names a static library and a DLL import library `ncnn.lib`, so the file
    // name cannot say which one the SDK ships and the DLL decides the layout.
    let shared_name = if os == "macos" {
        "libncnn.dylib"
    } else if target.contains("windows") {
        "ncnn.lib"
    } else {
        "libncnn.so"
    };
    let static_name = if target.contains("windows") {
        "ncnn.lib"
    } else {
        "libncnn.a"
    };
    let find = |name: &str| lib_dirs.iter().find(|dir| dir.join(name).is_file());
    let dll = [prefix.join("bin/ncnn.dll"), prefix.join("lib/ncnn.dll")]
        .into_iter()
        .find(|path| path.is_file());
    let has_static_file = find(static_name).is_some();
    let has_shared_file = if target.contains("windows") {
        dll.is_some()
    } else {
        find(shared_name).is_some()
    };
    assert!(
        has_static_file || has_shared_file,
        "NCNN_INSTALL_DIR must contain lib/{static_name} or lib/{shared_name} in lib or lib64"
    );
    // A shared MSVC install ships ncnn.lib as well, so the DLL separates the two
    // layouts; the other targets name the files apart.
    let layout = if target.contains("windows") {
        if has_shared_file {
            "dylib"
        } else {
            "static"
        }
    } else if has_shared_file && !has_static_file {
        "dylib"
    } else {
        "static"
    };
    let link_kind = match requested_kind.as_deref() {
        Some(requested) => {
            // A mismatch stays an error instead of a silent override: the kind also
            // selects NCNN_STATIC_DEFINE and the NCNN_EXTRA_LIBS closure, so the
            // requested and the linked kind must agree.
            let both_shipped = !target.contains("windows") && has_static_file && has_shared_file;
            assert!(
                requested == layout || both_shipped,
                "NCNN_LINK_KIND={requested} contradicts the SDK layout, which ships only {}; set NCNN_LINK_KIND={layout} or install a matching SDK",
                if has_shared_file && !has_static_file {
                    shared_name
                } else {
                    static_name
                }
            );
            requested
        }
        None => layout,
    };
    let library_name = if link_kind == "static" || target.contains("windows") {
        static_name
    } else {
        shared_name
    };
    let lib = find(library_name).unwrap_or_else(|| {
        panic!("NCNN_INSTALL_DIR must contain lib/{library_name} or lib64/{library_name}")
    });
    for path in [include.join("platform.h"), lib.join(library_name)] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if target.contains("windows") && link_kind == "dylib" {
        if let Some(dll) = &dll {
            println!(
                "cargo:warning=ncnn links ncnn.dll; put {} on PATH before running",
                dll.parent().expect("ncnn.dll has a parent").display()
            );
        }
    }
    let platform =
        std::fs::read_to_string(include.join("platform.h")).expect("Installed platform.h");
    for flag in [
        "NCNN_C_API",
        "NCNN_STDIO",
        "NCNN_STRING",
        "NCNN_PIXEL",
        "NCNN_PIXEL_ROTATE",
        "NCNN_PIXEL_AFFINE",
        "NCNN_PIXEL_DRAWING",
        "NCNN_BATCH",
        "NCNN_INT8",
        "NCNN_BF16",
        "NCNN_VALIDATION",
    ] {
        assert!(
            platform.contains(&format!("#define {flag} 1")),
            "Required native feature: {flag}"
        );
    }
    let vulkan = env::var_os("CARGO_FEATURE_VULKAN").is_some();
    assert!(
        platform.contains(&format!("#define NCNN_VULKAN {}", u8::from(vulkan))),
        "Cargo vulkan feature must match installed NCNN_VULKAN"
    );
    assert!(
        !vulkan
            || link_kind != "static"
            || env::var("NCNN_EXTRA_LIBS")
                .is_ok_and(|value| value.split(',').any(|library| !library.trim().is_empty())),
        "Static Vulkan requires explicit NCNN_EXTRA_LIBS from the SDK's exported CMake dependency closure"
    );
    // Compatibility runs point at a newer ncnn release on purpose; the interface and
    // numerical checks decide whether it still works.
    let version = platform
        .lines()
        .find_map(|line| line.strip_prefix("#define NCNN_VERSION_NUMBER "))
        .unwrap_or("unknown");
    if version != "20261020" {
        assert!(
            env::var_os("NCNN_ALLOW_UNTESTED_VERSION").is_some(),
            "Required ncnn version: 20261020 (found {version}); set NCNN_ALLOW_UNTESTED_VERSION to test another release"
        );
        println!("cargo:warning=ncnn {version} is not the pinned 20261020 release");
    }
    let mut adapter = cc::Build::new();
    if target.contains("windows") {
        adapter.define("NOMINMAX", None);
    }
    if link_kind == "static" {
        adapter.define("NCNN_STATIC_DEFINE", None);
    }
    adapter
        .cpp(true)
        .flag_if_supported("/EHsc")
        .include(&include)
        .file("native/adapter.cpp")
        .file("native/cpu_adapter.cpp")
        .file("native/topology.cpp")
        .file("native/image_adapter.cpp")
        .file("native/gpu_adapter.cpp")
        .flag_if_supported("-std=c++11")
        .compile("ncnn_rust_adapter");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib={link_kind}=ncnn");
    if let Some(paths) = env::var_os("NCNN_EXTRA_LIB_DIRS") {
        for path in env::split_paths(&paths) {
            assert!(
                path.is_dir(),
                "NCNN_EXTRA_LIB_DIRS must contain existing directories"
            );
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    for library in env::var("NCNN_EXTRA_LIBS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        let name = if let Some((kind, name)) = library.split_once('=') {
            assert!(
                matches!(kind, "static" | "dylib"),
                "NCNN_EXTRA_LIBS accepts static= or dylib= kinds"
            );
            name
        } else {
            library
        };
        assert!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_-+.".contains(&byte)),
            "NCNN_EXTRA_LIBS is a comma-separated list of [static=|dylib=]library names"
        );
        println!("cargo:rustc-link-lib={library}");
    }
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
    }
    let bindings = bindgen::Builder::default()
        .header("native/adapter.h")
        .clang_arg(format!("--target={target}"))
        .clang_arg(format!("-I{}", include.display()))
        .clang_args((link_kind == "static").then_some("-DNCNN_STATIC_DEFINE"))
        .allowlist_function("ncnn_.*")
        .allowlist_type("ncnn_.*|__ncnn_.*")
        .allowlist_var("NCNN_.*")
        .opaque_type("FILE|_IO_FILE")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Generate ncnn C bindings");
    bindings
        .write_to_file(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("bindings.rs"))
        .unwrap();
    // A warning surfaces in default-output builds, so keep it to the summary line.
    println!(
        "cargo:warning=ncnn {version}; vulkan={vulkan}; target={target}; prefix={}",
        prefix.display()
    );
}
