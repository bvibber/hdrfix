use rayon::prelude::*;

const BIN_COUNT: usize = 5;
const SCALE_COUNT: usize = 5;
const TEXTURE_EPSILON: f64 = 0.04;
const LOW_PERCENTILE: f64 = 0.001;
const HIGH_PERCENTILE: f64 = 0.999;
const LUMA_FLOOR: f32 = 1e-9;
const UNLIT_LEVEL: f32 = -1.0;
const DISPLAY_LOG_RANGE: f32 = 6.9078;
const BELOW_BLACK_LOG_RANGE: f32 = 4.1589;
const SOFT_DARK_LOG_RANGE: f32 = 0.2;
const SOFT_BRIGHT_LOG_RANGE: f32 = 0.1;
const SMALLEST_LOG_RANGE: f32 = 1e-3;
const CONTRAST_GAIN_LIMIT: f32 = 1.5;

type BinShares = [f64; BIN_COUNT];

fn squeeze(excess: f32, room: f32) -> f32 {
    room * (1.0 - (-excess / room).exp())
}

struct ToneRange {
    core_darkest: f32,
    core_brightest: f32,
    scene_darkest: f32,
    scene_range: f32,
    display_top: f32,
    display_span: f32,
}

impl ToneRange {
    fn measure(logs: &[f32]) -> Self {
        let mut sorted = logs.to_vec();
        sorted.par_sort_unstable_by(f32::total_cmp);
        let last_index = (sorted.len() - 1) as f64;
        let scene_brightest = sorted[(last_index * HIGH_PERCENTILE) as usize];
        let percentile_darkest = sorted[(last_index * LOW_PERCENTILE) as usize];

        let display_top = scene_brightest.clamp(SMALLEST_LOG_RANGE - DISPLAY_LOG_RANGE, 0.0);
        let scene_floor = display_top - DISPLAY_LOG_RANGE - BELOW_BLACK_LOG_RANGE;
        let core_darkest = percentile_darkest.max(scene_floor);
        let scene_darkest = core_darkest - SOFT_DARK_LOG_RANGE;
        let scene_range = (scene_brightest + SOFT_BRIGHT_LOG_RANGE - scene_darkest).max(SMALLEST_LOG_RANGE);
        let display_span = scene_range.min(DISPLAY_LOG_RANGE + display_top);

        ToneRange { core_darkest, core_brightest: scene_brightest, scene_darkest, scene_range, display_top, display_span }
    }

    fn squeeze_ends(&self, log: f32) -> f32 {
        if log < self.core_darkest {
            self.core_darkest - squeeze(self.core_darkest - log, SOFT_DARK_LOG_RANGE)
        } else if log > self.core_brightest {
            self.core_brightest + squeeze(log - self.core_brightest, SOFT_BRIGHT_LOG_RANGE)
        } else {
            log
        }
    }

    fn normalize(&self, log: f32) -> f32 {
        ((self.squeeze_ends(log) - self.scene_darkest) / self.scene_range).clamp(0.0, 1.0)
    }

    fn find_bin_ceiling(&self) -> f64 {
        (CONTRAST_GAIN_LIMIT * self.scene_range / (BIN_COUNT as f32 * self.display_span)) as f64
    }

    fn decode(&self, display_position: f32) -> f32 {
        (self.display_top - self.display_span + display_position * self.display_span).exp()
    }
}

struct Window {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

impl Window {
    fn around(x: usize, y: usize, half_size: usize, width: usize, height: usize) -> Self {
        Window {
            left: x.saturating_sub(half_size),
            top: y.saturating_sub(half_size),
            right: (x + half_size + 1).min(width),
            bottom: (y + half_size + 1).min(height),
        }
    }
}

struct IntegralTables {
    stride: usize,
    bin_counts: Vec<u32>,
    sums: Vec<f64>,
    square_sums: Vec<f64>,
}

impl IntegralTables {
    fn build(levels: &[f32], width: usize, height: usize) -> Self {
        let stride = width + 1;
        let cell_count = stride * (height + 1);
        let mut bin_counts = vec![0u32; cell_count * BIN_COUNT];
        let mut sums = vec![0f64; cell_count];
        let mut square_sums = vec![0f64; cell_count];

        for y in 0..height {
            let mut row_bins = [0u32; BIN_COUNT];
            let mut row_sum = 0f64;
            let mut row_square_sum = 0f64;

            for x in 0..width {
                let level = levels[y * width + x];
                if level >= 0.0 {
                    row_bins[find_bin(level)] += 1;
                    row_sum += level as f64;
                    row_square_sum += level as f64 * level as f64;
                }

                let above = y * stride + x + 1;
                let here = above + stride;
                sums[here] = sums[above] + row_sum;
                square_sums[here] = square_sums[above] + row_square_sum;
                for bin in 0..BIN_COUNT {
                    bin_counts[here * BIN_COUNT + bin] = bin_counts[above * BIN_COUNT + bin] + row_bins[bin];
                }
            }
        }

        IntegralTables { stride, bin_counts, sums, square_sums }
    }

    fn find_corners(&self, window: &Window) -> [usize; 4] {
        [
            window.bottom * self.stride + window.right,
            window.top * self.stride + window.left,
            window.top * self.stride + window.right,
            window.bottom * self.stride + window.left,
        ]
    }

    fn sum_window(table: &[f64], corners: &[usize; 4]) -> f64 {
        table[corners[0]] + table[corners[1]] - table[corners[2]] - table[corners[3]]
    }

    fn count_bins(&self, corners: &[usize; 4]) -> BinShares {
        let mut counts = [0f64; BIN_COUNT];
        for (bin, count) in counts.iter_mut().enumerate() {
            let added = self.bin_counts[corners[0] * BIN_COUNT + bin] + self.bin_counts[corners[1] * BIN_COUNT + bin];
            let removed = self.bin_counts[corners[2] * BIN_COUNT + bin] + self.bin_counts[corners[3] * BIN_COUNT + bin];
            *count = (added - removed) as f64;
        }
        counts
    }

    fn measure_texture(&self, corners: &[usize; 4], pixel_count: f64) -> f64 {
        let mean = Self::sum_window(&self.sums, corners) / pixel_count;
        let variance = (Self::sum_window(&self.square_sums, corners) / pixel_count - mean * mean).max(0.0);
        variance / (variance + TEXTURE_EPSILON)
    }
}

fn find_bin(level: f32) -> usize {
    ((level * BIN_COUNT as f32) as usize).min(BIN_COUNT - 1)
}

fn limit_contrast(mut shares: BinShares, ceiling: f64) -> BinShares {
    for _ in 0..BIN_COUNT {
        let excess: f64 = shares.iter().map(|share| (share - ceiling).max(0.0)).sum();
        let open_bin_count = shares.iter().filter(|share| **share < ceiling).count();
        if open_bin_count == 0 {
            break;
        }

        let refill = excess / open_bin_count as f64;
        for share in shares.iter_mut() {
            *share = if *share < ceiling { *share + refill } else { ceiling };
        }
    }
    shares
}

fn equalize(shares: &BinShares, level: f32) -> f64 {
    let bin = find_bin(level);
    let position_in_bin = (level * BIN_COUNT as f32 - bin as f32) as f64;
    let darker_share: f64 = shares[..bin].iter().sum();
    let total_share: f64 = shares.iter().sum();
    (darker_share + position_in_bin * shares[bin]) / total_share
}

fn fuse_scales(tables: &IntegralTables, bin_ceiling: f64, level: f32, x: usize, y: usize, width: usize, height: usize) -> f32 {
    let largest_half_size = width.max(height);
    let mut weighted_position = 0f64;
    let mut weight_total = 0f64;

    for scale in 0..SCALE_COUNT {
        let window = Window::around(x, y, largest_half_size >> scale, width, height);
        let corners = tables.find_corners(&window);
        let counts = tables.count_bins(&corners);
        let pixel_count: f64 = counts.iter().sum();
        if pixel_count == 0.0 {
            continue;
        }

        let shares = limit_contrast(counts.map(|count| count / pixel_count), bin_ceiling);
        let weight = tables.measure_texture(&corners, pixel_count).powi(scale as i32);

        weighted_position += weight * equalize(&shares, level);
        weight_total += weight;
    }

    (weighted_position / weight_total) as f32
}

pub fn map_luminance_multi_scale(lumas: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut mapped = vec![0f32; lumas.len()];
    let lit_logs: Vec<f32> = lumas.par_iter().filter(|luma| **luma > LUMA_FLOOR).map(|luma| luma.ln()).collect();
    if lit_logs.is_empty() {
        return mapped;
    }

    let tone_range = ToneRange::measure(&lit_logs);
    let levels: Vec<f32> = lumas.par_iter()
        .map(|luma| if *luma > LUMA_FLOOR { tone_range.normalize(luma.ln()) } else { UNLIT_LEVEL })
        .collect();
    let tables = IntegralTables::build(&levels, width, height);
    let bin_ceiling = tone_range.find_bin_ceiling();

    mapped.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, luma_out) in row.iter_mut().enumerate() {
            let level = levels[y * width + x];
            if level >= 0.0 {
                let position = fuse_scales(&tables, bin_ceiling, level, x, y, width, height);
                *luma_out = tone_range.decode(position);
            }
        }
    });

    mapped
}
