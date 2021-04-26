# hdrfix - a tool for mapping HDR screenshots to SDR

This is a tool I wrote for my personal usage dealing with HDR (high dynamic range) screenshots of Microsoft Flight Simulator, as taken with Nvidia's GeForce Experience game overlay capture utility which saves a JPEG XR which should contain full 10-bit range, and an 8-bit-per-channel PNG with lower resolution information, but still encoded in BT.2100 color space.

Output files as regular SDR (standard dynamic range) PNGs in bog-standard sRGB colorspace. There are a few parameters for adjusting the conversion.

# Author, repo, etc

* Brion Vibber `<brion @ pobox.com>`
* https://github.com/brion/hdrfix
* license: MIT

# Dependencies

* clap for CLI setup
* glam for vector/matrix math
* thiserror for error conglomeration
* png for reading input PNG
* mtpng for writing output PNG

# Installation

(untested, not yet published)

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

Adjustable parmeters are `--gamma`, `--sdr-white` and `--hdr-max`, all of which take numeric arguments:
* `--sdr-white=N` linearly scales the input signal such that a signal representing standard dark-room SDR white point of 80 nits is scaled up to the given value instead. The default is `80`, passing through the standard signal. A higher value will darken the image linearly.
* `--gamma=N` applies a power curve against the tone-mapped luminance signal before color correction and saving. The default is `1.0`, which is linear. A modest gamma of `1.2` or `1.4` looks nice on many images, boosting contrast.
* `--hdr-max=N` sets the maximum luminance level for the Reinhold tone-mapping algorithm. Higher values will preserve more detail in very bright areas, at the cost of poorer contrast in highlights. The default is `10000` nits; anything brighter than that will be capped.



# Todo / roadmap

Definitely/short-term:
* check transform performance
* use Vec3A or Vec4 for speed if it helps
* auto-output-filename feature to make it easier to use on live folders
* [IN PROGRESS] add JPEG XR input (should reduce banding in sky vs using the PNGs)
* add JPEG output
* add compression params for JPEG output
* make a InputStream<R: Read+Seek> and OutputStream<W: Write+Seek>; remember the methods are for the library to use, and for us to provide. the struct must maintain lifetime of the underlying Read/Write, and can return access to its internal Stream struct, which is not copyable.

Probably:
* 'folder watch' feature to convert all new .jxr files appearing in a folder while we run with default parameters

Maybe/later/no rush:
* a basic GUI with HDR and SDR side-by-side view
* GUI sliders for the adjustable parameters
* drag/drop and open/save dialog support

# Building

```
cargo build --release
```

Requires Rust to be installed ()

# Updating jpegxr library bindings

JPEG XR input is read using Microsoft's BSD-licensed libjpegxr, which I've bundled for now but will later split out to its own crate for ease of reuse.

bindgen is used to create a low-level Rust interface to the C library; the bindings are created automatically as part of the build process

You must install LLVM + Clang to complete a build; on Windows you can get a release from https://github.com/llvm/llvm-project/releases/tag/llvmorg-12.0.0 or whatever the current release is. On Linux or Mac, use the system or user-preferred package manager.
