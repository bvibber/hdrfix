use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{transforms::luma_scrgb, PixelBuffer};

pub struct Histogram {
    luma_vals: Vec<f32>,
}

impl Histogram {
    pub fn new(source: &PixelBuffer) -> Self {
        // @todo maybe do a proper histogram with buckets
        // instead of sorting every pixel value
        let mut luma_vals = Vec::<f32>::new();
        source
            .pixels()
            .map(luma_scrgb)
            .collect_into_vec(&mut luma_vals);
        luma_vals.par_sort_unstable_by(|a, b| match a.partial_cmp(b) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        });
        Self { luma_vals }
    }

    pub fn percentile(&self, target: f32) -> f32 {
        let max_index = self.luma_vals.len() - 1;
        let target_index = (max_index as f32 * target / 100.0) as usize;
        self.luma_vals[target_index]
    }

    pub fn average_below_percentile(&self, percent: f32) -> f32 {
        let max = self.percentile(percent);
        let (sum, count) = self
            .luma_vals
            .iter()
            .fold((0.0f32, 0usize), |(sum, count), luma| {
                if *luma > max {
                    (sum, count)
                } else {
                    (sum + luma, count + 1)
                }
            });
        sum / count as f32
    }
}
