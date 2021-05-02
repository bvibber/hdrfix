# hdrfix - a tool for mapping HDR screenshots to SDR

This is a tool I wrote for my personal usage dealing with HDR (high dynamic range) screenshots of Microsoft Flight Simulator, as taken with Nvidia's GeForce Experience game overlay capture utility which saves a JPEG XR in 32-bit float precision scRGB, and an 8-bit-per-channel PNG with lower resolution information, but encoded with BT.2100 color space and transfer function.

Outputs files as regular SDR (standard dynamic range) PNGs in bog-standard sRGB colorspace. There are a few parameters for adjusting the conversion.

JPEG XR conversion is done with the `jpegxr` crate, which wraps Microsoft's BSD-licensed JPEG XR codec.

# Author, repo, etc

* Brion Vibber `<brion @ pobox.com>`
* https://github.com/brion/hdrfix
* license: MIT (wrapper and conversion code), BSD (jpegxr library)

# Dependencies

* clap for CLI setup
* time for speed checks in the CLI utility
* thiserror for error conglomeration
* rayon for multithreading
* glam for vector/matrix math
* png for reading input PNG
* mtpng for writing output PNG
* jpegxr for the JPEG XR C libray (and through it, bindgen and cc)

# Installation

From source checkout:

```
cargo install --path=.
```

From crates.io:

```
cargo install hdrfix
```

# Usage

Basic conversion:

```
hdrfix screenshot.png output.png
```

Interactive help!

```
hdrfix --help
```

Adjustable parmeters:
* `--pre-scale=N` multiplies the input signal. The default is `1.0`, passing through the original signal.
* `--pre-gamma-N` applies an exponential gamma curve to the input after scaling. The default is `1.0`, passing through the original signal.
* `--tone-map=A` sets the HDR to SDR tone-mapping algorithm; choices are `linear` which will clip/correct anything brighter than 1.0, or one of `reinhard-luma` or `reinhard-rgb` which applies the Reinhard tone-mapping algorithm on either the luminance or separate RGB color channels. Luminance mode preserves colors better; RGB mode will apply desaturation on brighter colors nicely but also can shift colors and alter luminance a bit. Default is `reinhard-luma`.
* `--desaturation-coeff` sets the desaturation coefficient for chroma mapping in the Reinhard luma version. Default is `1.0` (no desaturation), but `0.75` looks nice on my test images.
* `--hdr-max=N` sets the maximum luminance level for the Reinhard tone-mapping algorithm. Higher values will preserve more detail in very bright areas, at the cost of slightly poorer contrast in highlights. The default is `10000` nits which is the maximum for HDR10 input. A lower value will cause very bright details to blow out, but slightly lighten dark areas.
* `--post-gamma-N` applies an exponential gamma curve to the output after tone mapping. The default is `1.0`, passing through the original signal.
* `--post-scale=N` multiplies the output signal. The default is `1.0`, passing through the original signal.
* `--color-map=A` sets the color-mapping algorithm for out of gamut colors after tone-mapping. Choices are `clip` which can alter color and brightness, `darken` which can cause major shifts in relative contrast but preserves color precisely, or `desaturate` which preserves luminance but desaturates color as necessary to fit in gamut. Deafult is `desaturate`.


# Todo / roadmap

Definitely/short-term:
* auto-output-filename feature to make it easier to use on live folders
* allow auto-set of the pre-shift/pre-scale with histogram percentiles
* add JPEG output
* add compression params for JPEG output

Probably:
* 'folder watch' feature to convert all new .jxr files appearing in a folder while we run with default parameters

Maybe/later/no rush:
* see if can get a performance boost from Vec3A instead of Vec3 (so far it's been slightly slower when tested)
* a basic GUI with HDR and SDR side-by-side view
* GUI sliders for the adjustable parameters
* drag/drop and open/save dialog support

# Building

```
cargo build --release
```

Requires Rust and Cargo, and a C compiler. On Windows, install Visual Studio Community Edition with C++ development tools or else the command-line build tools. On Linux or Mac there may be some compilation problems at the moment as the jpegxr C library code is still being adapted.

You must install LLVM + Clang to complete a build due to the C code; on Windows you can get a release from https://github.com/llvm/llvm-project/releases/tag/llvmorg-12.0.0 or whatever the current release is. On Linux or Mac, use the system or user-preferred package manager.
