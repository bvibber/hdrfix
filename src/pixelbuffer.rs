use glam::Vec3;
use rayon::prelude::*;

use crate::{transforms::*, PixelFormat};

// Note: currently assumes stride == width
pub struct PixelBuffer {
    pub width: usize,
    pub height: usize,
    pub bytes_per_pixel: usize,
    pub data: Vec<u8>,

    // If we wanted these could be traits
    // but we don't need that level of complexity
    pub read_rgb_func: fn(&[u8]) -> Vec3,
    pub write_rgb_func: fn(&mut [u8], Vec3),
}

impl PixelBuffer {
    pub fn new(width: usize, height: usize, format: PixelFormat) -> Self {
        let bytes_per_pixel = match format {
            PixelFormat::SDR8bit | PixelFormat::HDR8bit => 3,
            PixelFormat::HDRFloat16 => 8,
            PixelFormat::HDRFloat32 => 16,
        };
        let read_rgb_func = match format {
            PixelFormat::SDR8bit => read_srgb_rgb24,
            PixelFormat::HDR8bit => read_rec2100_rgb24,
            PixelFormat::HDRFloat16 => read_scrgb_rgb64half,
            PixelFormat::HDRFloat32 => read_scrgb_rgb128float,
        };
        let write_rgb_func = match format {
            PixelFormat::SDR8bit => write_srgb_rgb24,
            PixelFormat::HDR8bit => write_rec2100_rgb24,
            PixelFormat::HDRFloat16 => write_scrgb_rgb64half,
            PixelFormat::HDRFloat32 => write_scrgb_rgb128float,
        };
        let stride = width * bytes_per_pixel;
        let size = stride * height;
        let data = vec![0u8; size];

        PixelBuffer {
            width,
            height,
            bytes_per_pixel,
            data,
            read_rgb_func,
            write_rgb_func,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &[u8]> {
        self.data.par_chunks(self.bytes_per_pixel)
    }

    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [u8]> {
        self.data.par_chunks_mut(self.bytes_per_pixel)
    }

    pub fn pixels(&self) -> impl '_ + IndexedParallelIterator<Item = Vec3> {
        self.par_iter().map(self.read_rgb_func)
    }

    pub fn fill<T>(&mut self, source: T)
    where
        T: IndexedParallelIterator<Item = Vec3>,
    {
        let write_rgb_func = self.write_rgb_func;
        self.par_iter_mut()
            .zip(source)
            .for_each(|(dest, rgb)| write_rgb_func(dest, rgb))
    }
}
