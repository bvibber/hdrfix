//! Image format reader and writers

use std::cmp::Ordering;

// Math bits
use glam::{Mat3, Vec3};
// 16-bit floats
use half::prelude::*;
use oklab::{linear_srgb_to_oklab, oklab_to_linear_srgb, Oklab};

use crate::Context;

pub fn read_srgb_rgb24(_data: &[u8]) -> Vec3 {
    panic!("not yet implemented");
}

pub fn write_srgb_rgb24(data: &mut [u8], val: Vec3) {
    let gamma_out = linear_to_srgb(val);
    let clipped = clip(gamma_out);
    let scaled = clipped * 255.0;
    data[0] = scaled.x as u8;
    data[1] = scaled.y as u8;
    data[2] = scaled.z as u8;
}

pub fn read_rec2100_rgb24(data: &[u8]) -> Vec3 {
    let scale = Vec3::splat(1.0 / 255.0);
    let rgb_rec2100 = Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32) * scale;
    let rgb_linear = pq_to_linear(rgb_rec2100);
    rec2100_to_scrgb(rgb_linear)
}

pub fn write_rec2100_rgb24(_data: &mut [u8], _rgb: Vec3) {
    panic!("not yet implemented");
}

pub fn read_scrgb_rgb64half(data: &[u8]) -> Vec3 {
    let data_ref_f16: &f16 = unsafe { std::mem::transmute(&data[0]) };
    let data_f16 = unsafe { std::slice::from_raw_parts(data_ref_f16, data.len()) };
    Vec3::new(
        data_f16[0].to_f32(),
        data_f16[1].to_f32(),
        data_f16[2].to_f32(),
    )
}

pub fn write_scrgb_rgb64half(data: &mut [u8], rgb: Vec3) {
    let data_ref_f16: &mut f16 = unsafe { std::mem::transmute(&mut data[0]) };
    let data_f16 = &mut unsafe { std::slice::from_raw_parts_mut(data_ref_f16, data.len()) };
    data_f16[0] = f16::from_f32(rgb.x);
    data_f16[1] = f16::from_f32(rgb.y);
    data_f16[2] = f16::from_f32(rgb.z);
}

pub fn read_scrgb_rgb128float(data: &[u8]) -> Vec3 {
    let data_ref_f32: &f32 = unsafe { std::mem::transmute(&data[0]) };
    let data_f32 = unsafe { std::slice::from_raw_parts(data_ref_f32, data.len()) };
    Vec3::new(data_f32[0], data_f32[1], data_f32[2])
}

pub fn write_scrgb_rgb128float(data: &mut [u8], rgb: Vec3) {
    let data_ref_f32: &mut f32 = unsafe { std::mem::transmute(&mut data[0]) };
    let data_f32 = &mut unsafe { std::slice::from_raw_parts_mut(data_ref_f32, data.len()) };
    data_f32[0] = rgb.x;
    data_f32[1] = rgb.y;
    data_f32[2] = rgb.z;
}

pub fn pq_to_linear(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let inv_m1: f32 = 1.0 / 0.15930176;
    let inv_m2: f32 = 1.0 / 78.84375;
    let c1 = Vec3::splat(0.8359375);
    let c2 = Vec3::splat(18.851563);
    let c3 = Vec3::splat(18.6875);
    let val_powered = val.powf(inv_m2);
    (Vec3::max(val_powered - c1, Vec3::ZERO) / (c2 - c3 * val_powered)).powf(inv_m1)
}

pub fn rec2100_to_scrgb(val: Vec3) -> Vec3 {
    let matrix = Mat3::from_cols_array(&[
        1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187,
    ]);
    let scale = REC2100_MAX / SDR_WHITE;
    matrix.mul_vec3(val * scale)
}

pub fn luma_scrgb(val: Vec3) -> f32 {
    luma_oklab(scrgb_to_oklab(val))
}

pub fn luma_oklab(val: Oklab) -> f32 {
    // oklab's l is not linear
    // so translate it back to linear srgb desaturated
    // and take one of its rgb values
    let oklab_gray = Oklab {
        l: val.l,
        a: 0.0,
        b: 0.0,
    };
    let rgb_gray = oklab_to_scrgb(oklab_gray);
    rgb_gray.x
}

pub const EPSILON: f32 = 0.001; // good enough for us for now

pub fn close_enough(a: f32, b: f32) -> Ordering {
    let delta = a - b;
    if delta.abs() < EPSILON {
        Ordering::Equal
    } else if delta < 0.0 {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

pub fn binary_search<I, O, F, G>(input: I, min: f32, max: f32, func: F, comparator: G) -> O
where
    I: Copy + Clone,
    O: Copy + Clone,
    F: Fn(I, f32) -> O,
    G: Fn(O) -> Ordering,
{
    let mid = (min + max) / 2.0;
    let result = func(input, mid);
    match close_enough(min, max) {
        Ordering::Equal => result,
        _ => match comparator(result) {
            Ordering::Less => binary_search(input, mid, max, func, comparator),
            Ordering::Greater => binary_search(input, min, mid, func, comparator),
            Ordering::Equal => result,
        },
    }
}

pub fn clip(input: Vec3) -> Vec3 {
    input.max(Vec3::ZERO).min(Vec3::ONE)
}

pub fn darken_oklab(c_in: Oklab, brightness: f32) -> Vec3 {
    let c_out = Oklab {
        l: c_in.l * brightness,
        a: c_in.a * brightness,
        b: c_in.b * brightness,
    };
    oklab_to_scrgb(c_out)
}

pub fn desat_oklab(c_in: Oklab, saturation: f32) -> Vec3 {
    let c_out = Oklab {
        l: c_in.l,
        a: c_in.a * saturation,
        b: c_in.b * saturation,
    };
    oklab_to_scrgb(c_out)
}

pub fn luma_rgb(val: Vec3) -> f32 {
    val.x * 0.2126 + val.y * 0.7152 + val.z * 0.0722
}

pub fn scale_rgb(val: Vec3, luma_out: f32) -> Vec3 {
    let luma_in = luma_rgb(val);
    let scale = luma_out / luma_in;
    val * scale
}

// can't use glam's Mat3 as a constant literal?
pub type Matrix3x3 = [[f32; 3]; 3];

// https://64.github.io/tonemapping/#aces
// ACES (Academy Color Encoding System)
pub const ACES_INPUT_MATRIX: Matrix3x3 = [
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
];

pub const ACES_OUTPUT_MATRIX: Matrix3x3 = [
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
];

#[allow(clippy::many_single_char_names)]
pub fn aces_mul(m: &Matrix3x3, v: Vec3) -> Vec3 {
    let x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
    let y = m[1][0] * v[1] + m[1][1] * v[1] + m[1][2] * v[2];
    let z = m[2][0] * v[1] + m[2][1] * v[1] + m[2][2] * v[2];
    Vec3::new(x, y, z)
}

pub fn aces_rtt_and_odt_fit(v: Vec3) -> Vec3 {
    let a = v * (v + Vec3::splat(0.0245786)) - Vec3::splat(0.000090537);
    let b = v * (Vec3::splat(0.983729) * v + Vec3::splat(0.432951)) + Vec3::splat(0.238081);
    a / b
}

pub fn tonemap_aces(c_in: Vec3, _options: &Context) -> Vec3 {
    let v = c_in;
    let v = aces_mul(&ACES_INPUT_MATRIX, v);
    let v = aces_rtt_and_odt_fit(v);
    aces_mul(&ACES_OUTPUT_MATRIX, v)
}

/*
pub fn srgb_to_linear(val: Vec3) -> Vec3 {
    Vec3::select(
        val.cmple(Vec3::splat(0.04045)),
        val / Vec3::splat(12.92),
        ((val + Vec3::splat(0.055)) / Vec3::splat(1.055)).powf(2.4)
    )
}
*/

pub fn linear_to_srgb(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let min = Vec3::splat(0.0031308);
    let linear = val * Vec3::splat(12.92);
    let gamma = (val * Vec3::splat(1.055)).powf(1.0 / 2.4) - Vec3::splat(0.055);
    Vec3::select(val.cmple(min), linear, gamma)
}

pub const REC2100_MAX: f32 = 10000.0; // the 1.0 value for BT.2100 linear
pub const SDR_WHITE: f32 = 80.0;

pub fn exposure_scale(stops: f32) -> f32 {
    2.0_f32.powf(stops)
}

pub fn hdr_to_sdr_pixel(rgb_scrgb: Vec3, options: &Context) -> Vec3 {
    let val = rgb_scrgb * options.scale;
    let val = options.tone_map.map(val, options);
    options.color_map.map(val)
}

pub fn scrgb_to_linear_srgb(c: Vec3) -> oklab::RGB<f32> {
    oklab::RGB::new(c.x, c.y, c.z)
}

pub fn linear_srgb_to_scrgb(c: oklab::RGB<f32>) -> Vec3 {
    Vec3::new(c.r, c.g, c.b)
}

pub fn oklab_l_for_luma(luma: f32) -> f32 {
    let gray_rgb = oklab::RGB::new(luma, luma, luma);
    let gray_oklab = linear_srgb_to_oklab(gray_rgb);
    gray_oklab.l
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

pub fn scale_oklab(oklab_in: Oklab, luma_out: f32) -> Oklab {
    if oklab_in.l == 0.0 {
        oklab_in
    } else {
        let gray_l = oklab_l_for_luma(luma_out);
        let ratio = gray_l / oklab_in.l;
        Oklab {
            l: gray_l,
            a: oklab_in.a * ratio,
            b: oklab_in.b * ratio,
        }
    }
}

pub fn scrgb_to_oklab(c: Vec3) -> Oklab {
    linear_srgb_to_oklab(scrgb_to_linear_srgb(c))
}

pub fn oklab_to_scrgb(c: Oklab) -> Vec3 {
    linear_srgb_to_scrgb(oklab_to_linear_srgb(c))
}

pub fn apply_levels(c_in: Vec3, level_min: f32, level_max: f32, gamma: f32) -> Vec3 {
    let offset = level_min;
    let scale = level_max - level_min;
    let oklab_in = scrgb_to_oklab(c_in);
    let luma_in = luma_oklab(oklab_in);
    let luma_out = ((luma_in - offset) / scale).powf(gamma);
    let oklab_out = scale_oklab(oklab_in, luma_out);
    oklab_to_scrgb(oklab_out)
}
