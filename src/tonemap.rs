use glam::Vec3;
use oklab::Oklab;

use crate::{transforms::*, Context};

/// Tonemap implementations
#[derive(Copy, Clone, Debug, PartialEq, clap::ValueEnum)]
pub enum Tonemap {
    Linear,
    Reinhard,
    ReinhardRgb,
    Aces,
    Uncharted2,
    Hable,
}

impl Tonemap {
    /// Perform mapping for the [Tonemap] variant
    pub fn map(&self, c_in: Vec3, options: &Context) -> Vec3 {
        match self {
            Tonemap::Linear => tonemap_linear(c_in, options),
            Tonemap::Reinhard => tonemap_reinhard_oklab(c_in, options),
            Tonemap::ReinhardRgb => tonemap_reinhard_rgb(c_in, options),
            Tonemap::Aces => tonemap_aces(c_in, options),
            Tonemap::Uncharted2 => tonemap_uncharted2(c_in, options),
            Tonemap::Hable => tonemap_hable(c_in, options),
        }
    }
}

pub fn tonemap_linear(c_in: Vec3, _options: &Context) -> Vec3 {
    c_in
}

/// Map luminance from HDR to SDR domain, and scale the input color
/// in oklab perceptual color space.
///
/// oklab color space: https://bottosson.github.io/posts/oklab/
pub fn tonemap_reinhard_oklab(c_in: Vec3, options: &Context) -> Vec3 {
    //
    let white = options.hdr_max;
    let white2 = white * white;

    // use Oklab's L coordinate as luminance
    let oklab_in = scrgb_to_oklab(c_in);
    let luma_in = luma_oklab(oklab_in);

    // Reinhard tone-mapping algo.
    //
    // Original:
    // http://www.cmap.polytechnique.fr/%7Epeyre/cours/x2005signal/hdr_photographic.pdf
    //
    // Extended:
    // https://64.github.io/tonemapping/#reinhard
    // TMO_reinhardext​(C) = C(1 + C/C_white^2​) / (1 + C)
    //
    let luma_out = luma_in * (1.0 + luma_in / white2) / (1.0 + luma_in);
    let oklab_out = scale_oklab_desat(oklab_in, luma_out, options.saturation);
    oklab_to_scrgb(oklab_out)
}

pub fn scale_oklab_desat(oklab_in: Oklab, luma_out: f32, saturation: f32) -> Oklab {
    let l_in = oklab_in.l;
    if l_in == 0.0 {
        oklab_in
    } else {
        let l_out = oklab_l_for_luma(luma_out);
        // oklab coords scale cubically
        // 1.0 -> desaturate linearly according to luma compression ratio
        // 0.5 -> desaturate more aggressively
        // 2.0 -> saturate more aggressively
        let ratio = (l_out / l_in).powf(3.0 / saturation);
        Oklab {
            l: l_out,
            a: oklab_in.a * ratio,
            b: oklab_in.b * ratio,
        }
    }
}

/// Variant that maps R, G, and B channels separately.
/// This should desaturate very bright colors gradually, but will
/// possible cause some color shift.
pub fn tonemap_reinhard_rgb(c_in: Vec3, options: &Context) -> Vec3 {
    let white = options.hdr_max;
    let white2 = white * white;
    c_in * (Vec3::ONE + c_in / white2) / (Vec3::ONE + c_in)
}

// https://64.github.io/tonemapping/#uncharted-2
// Uncharted 2 / Hable Filmic
pub fn uncharted2_tonemap_partial(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + (C * B)) + (D * E)) / (x * (A * x + (B)) + (D * F))) - (E / F)
}

pub fn tonemap_uncharted2(v: Vec3, _options: &Context) -> Vec3 {
    let exposure_bias: f32 = 2.0;
    let luma = luma_rgb(v);
    let curr = uncharted2_tonemap_partial(luma * exposure_bias);

    let w = 11.2f32;
    let white_scale = 1.0f32 / uncharted2_tonemap_partial(w);
    let luma_out = curr * white_scale;

    scale_rgb(v, luma_out)
}

pub fn tonemap_hable(val: Vec3, _options: &Context) -> Vec3 {
    // stolen from ffmpeg's vf_tonemap

    // desat
    let luma = luma_rgb(val);
    let desaturation: f32 = 2.0;
    let epsilon: f32 = 1e-6;
    let overbright = f32::max(luma - desaturation, epsilon) / f32::max(luma, epsilon);
    let rgb_out = val * (1.0 - overbright) + luma * overbright;
    let sig_orig = f32::max(rgb_out.max_element(), epsilon);

    // hable/uncharted2
    let exposure_bias: f32 = 2.0;
    let luma = sig_orig;
    let curr = uncharted2_tonemap_partial(luma * exposure_bias);
    let w = 11.2f32;
    let white_scale = 1.0f32 / uncharted2_tonemap_partial(w);
    let sig = curr * white_scale;

    rgb_out * (sig / sig_orig)
}
