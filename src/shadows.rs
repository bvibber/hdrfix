use rayon::prelude::*;

fn encode_srgb(luma: f32) -> f32 {
    if luma <= 0.0031308 {
        luma * 12.92
    } else {
        1.055 * luma.powf(1.0 / 2.4) - 0.055
    }
}

fn decode_srgb(signal: f32) -> f32 {
    if signal <= 0.04045 {
        signal / 12.92
    } else {
        ((signal + 0.055) / 1.055).powf(2.4)
    }
}

const SURROUND_RADIUS_DIVISOR: usize = 40;
const EDGE_EPSILON: f32 = 0.01;
const LIFT_STRENGTH: f32 = 3.3;
const DARKEST_SURROUND: f32 = 0.01;

fn average_boxes(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let stride = width + 1;
    let mut integral = vec![0f64; stride * (height + 1)];
    for y in 0..height {
        let mut row_sum = 0f64;
        for x in 0..width {
            row_sum += values[y * width + x] as f64;
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row_sum;
        }
    }

    let mut averages = vec![0f32; values.len()];
    averages.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let top = y.saturating_sub(radius);
        let bottom = (y + radius + 1).min(height);
        for (x, average) in row.iter_mut().enumerate() {
            let left = x.saturating_sub(radius);
            let right = (x + radius + 1).min(width);
            let sum = integral[bottom * stride + right] + integral[top * stride + left]
                - integral[top * stride + right] - integral[bottom * stride + left];
            *average = (sum / ((right - left) * (bottom - top)) as f64) as f32;
        }
    });
    averages
}

fn smooth_keeping_edges(signals: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let squares: Vec<f32> = signals.par_iter().map(|signal| signal * signal).collect();
    let means = average_boxes(signals, width, height, radius);
    let square_means = average_boxes(&squares, width, height, radius);

    let edge_weights: Vec<f32> = means.par_iter().zip(square_means.par_iter()).map(|(mean, square_mean)| {
        let variance = (square_mean - mean * mean).max(0.0);
        variance / (variance + EDGE_EPSILON)
    }).collect();
    let offsets: Vec<f32> = means.par_iter().zip(edge_weights.par_iter()).map(|(mean, edge_weight)| mean * (1.0 - edge_weight)).collect();

    let edge_weight_means = average_boxes(&edge_weights, width, height, radius);
    let offset_means = average_boxes(&offsets, width, height, radius);

    signals.par_iter().zip(edge_weight_means.par_iter()).zip(offset_means.par_iter())
        .map(|((signal, edge_weight), offset)| (edge_weight * signal + offset).clamp(0.0, 1.0))
        .collect()
}

fn find_gain(surround: f32, amount: f32) -> f32 {
    let surround = surround.max(DARKEST_SURROUND);
    let fully_lifted = surround.powf(1.0 / (1.0 + LIFT_STRENGTH * amount));
    let shadow_weight = (1.0 - surround).powi(2);
    (surround + shadow_weight * (fully_lifted - surround)) / surround
}

pub fn lift_shadows(lumas: &[f32], width: usize, height: usize, amount: f32) -> Vec<f32> {
    let signals: Vec<f32> = lumas.par_iter().map(|luma| encode_srgb(luma.max(0.0).min(1.0))).collect();
    let radius = width.max(height) / SURROUND_RADIUS_DIVISOR;
    let surrounds = smooth_keeping_edges(&signals, width, height, radius);

    signals.par_iter().zip(surrounds.par_iter())
        .map(|(signal, surround)| decode_srgb((signal * find_gain(*surround, amount)).min(1.0)))
        .collect()
}
