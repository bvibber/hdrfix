use std::env;
use std::path::PathBuf;

fn main() {
    let src = [
        // SRC_SYS
        "jxrlib/image/sys/adapthuff.c",
        "jxrlib/image/sys/image.c",
        "jxrlib/image/sys/strcodec.c",
        "jxrlib/image/sys/strPredQuant.c",
        "jxrlib/image/sys/strTransform.c",
        "jxrlib/image/sys/perfTimerANSI.c",
        // SRC_DEC
        "jxrlib/image/decode/decode.c",
        "jxrlib/image/decode/postprocess.c",
        "jxrlib/image/decode/segdec.c",
        "jxrlib/image/decode/strdec.c",
        "jxrlib/image/decode/strInvTransform.c",
        "jxrlib/image/decode/strPredQuantDec.c",
        "jxrlib/image/decode/JXRTranscode.c",
        // SRC_ENC
        "jxrlib/image/encode/encode.c",
        "jxrlib/image/encode/segenc.c",
        "jxrlib/image/encode/strenc.c",
        "jxrlib/image/encode/strFwdTransform.c",
        "jxrlib/image/encode/strPredQuantEnc.c",
        // glue lib
        "jxrlib/jxrgluelib/JXRGlue.c",
        "jxrlib/jxrgluelib/JXRGlueJxr.c",
        "jxrlib/jxrgluelib/JXRGluePFC.c",
        "jxrlib/jxrgluelib/JXRMeta.c",
    ];
    let mut builder = cc::Build::new();
    let build = builder
        .files(src.iter())
        .include("jxrlib")
        .include("jxrlib/common/include")
        .include("jxrlib/image/sys")
        .include("jxrlib/jxrgluelib")
        .define("__ANSI__", None)
        .define("DISABLE_PERF_MEASUREMENT", None)
        .opt_level(2);

    build.compile("jpegxr");

    let bindings = bindgen::Builder::default()
        .header("jxrlib/jxrgluelib/JXRGlue.h")
        .allowlist_function("^(WMP|PK|PixelFormatLookup|GetPixelFormatFromHash|GetImageEncodeIID|GetImageDecodeIID|FreeDescMetadata).*")
        .allowlist_var("^(WMP|PK|LOOKUP|GUID_PK|IID).*")
        .allowlist_type("^(WMP|PK|ERR|BITDEPTH|BD_|BITDEPTH_BITS|COLORFORMAT).*")
        .clang_args(&[
            "-Ijxrlib/jxrgluelib",
            "-Ijxrlib/common/include",
            "-Ijxrlib/image/sys"
        ])
        .derive_eq(true)
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Error building libjpegxr bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let tmp_path = PathBuf::from("/tmp");
    bindings
        .write_to_file(tmp_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
