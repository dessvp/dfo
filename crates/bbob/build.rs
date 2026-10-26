//! Compiles the vendored COCO amalgamation (coco/coco.c, BSD-3, see
//! coco/LICENSE) with the system C compiler into a static library.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("coco/coco.c");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("coco.o");
    let lib = out.join("libcoco.a");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut cmd = Command::new(&cc);
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Match rustc's default deployment target.
        cmd.arg("-mmacosx-version-min=11.0");
    }
    let status = cmd.args(["-O2", "-c", "-w", "-o"]).arg(&obj).arg(&src).status().expect("failed to run the C compiler");
    assert!(status.success(), "compiling coco.c failed");
    let status = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=coco");
    println!("cargo:rerun-if-changed=coco/coco.c");
    println!("cargo:rerun-if-changed=coco/coco.h");
}
