use glam::Vec3;

use crate::transforms::*;

/// Color mappings
#[derive(Copy, Clone, Debug, PartialEq, clap::ValueEnum)]
pub enum ColorMap {
    Clip,
    Darken,
    Desaturate,
}

impl ColorMap {
    pub fn map(&self, input: Vec3) -> Vec3 {
        match self {
            ColorMap::Clip => color_clip(input),
            ColorMap::Darken => color_darken_oklab(input),
            ColorMap::Desaturate => color_desat_oklab(input),
        }
    }
}

pub fn color_clip(input: Vec3) -> Vec3 {
    clip(input)
}

pub fn color_desat_oklab(c_in: Vec3) -> Vec3 {
    let max = c_in.max_element();
    if max > 1.0 {
        let c_in_oklab = scrgb_to_oklab(c_in);
        let c_out = binary_search(c_in_oklab, 0.0, 1.0, desat_oklab, |rgb| {
            close_enough(rgb.max_element(), 1.0)
        });
        clip(c_out)
    } else {
        c_in
    }
}

pub fn color_darken_oklab(c_in: Vec3) -> Vec3 {
    let max = c_in.max_element();
    if max > 1.0 {
        let c_in_oklab = scrgb_to_oklab(c_in);
        let c_out = binary_search(c_in_oklab, 0.0, 1.0, darken_oklab, |rgb| {
            close_enough(rgb.max_element(), 1.0)
        });
        clip(c_out)
    } else {
        c_in
    }
}
