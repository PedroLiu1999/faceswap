use std::f32::consts::PI;

pub const STANDARD_TEMPLATE_128: [[f32; 2]; 5] = [
    [38.2946, 51.6963],  // Left Eye
    [73.5318, 51.5014],  // Right Eye
    [56.0252, 71.7366],  // Nose Tip
    [41.5493, 92.3655],  // Left Mouth Corner
    [70.7299, 104.0304], // Right Mouth Corner
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMatrix2x3 {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub c: f32,
    pub d: f32,
    pub ty: f32,
}

impl AffineMatrix2x3 {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            tx: 0.0,
            c: 0.0,
            d: 1.0,
            ty: 0.0,
        }
    }

    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-9 {
            return None;
        }

        let inv_det = 1.0 / det;
        let inv_a = self.d * inv_det;
        let inv_b = -self.b * inv_det;
        let inv_c = -self.c * inv_det;
        let inv_d = self.a * inv_det;

        let inv_tx = -(inv_a * self.tx + inv_b * self.ty);
        let inv_ty = -(inv_c * self.tx + inv_d * self.ty);

        Some(Self {
            a: inv_a,
            b: inv_b,
            tx: inv_tx,
            c: inv_c,
            d: inv_d,
            ty: inv_ty,
        })
    }
}

/// 2D Umeyama least-squares similarity transform without reflection.
/// Maps src_points to dst_points minimizing mean squared error: dst ≈ M * src.
pub fn estimate_similarity_transform(
    src_points: &[[f32; 2]; 5],
    dst_points: &[[f32; 2]; 5],
) -> Option<AffineMatrix2x3> {
    let n = 5.0f32;

    // Centroids
    let (mut src_mx, mut src_my) = (0.0f32, 0.0f32);
    let (mut dst_mx, mut dst_my) = (0.0f32, 0.0f32);
    for i in 0..5 {
        src_mx += src_points[i][0];
        src_my += src_points[i][1];
        dst_mx += dst_points[i][0];
        dst_my += dst_points[i][1];
    }
    src_mx /= n;
    src_my /= n;
    dst_mx /= n;
    dst_my /= n;

    // Centered sums
    let mut var_src = 0.0f32;
    let mut num_a = 0.0f32;
    let mut num_b = 0.0f32;

    for i in 0..5 {
        let sx = src_points[i][0] - src_mx;
        let sy = src_points[i][1] - src_my;
        let dx = dst_points[i][0] - dst_mx;
        let dy = dst_points[i][1] - dst_my;

        var_src += sx * sx + sy * sy;
        num_a += sx * dx + sy * dy;
        num_b += sx * dy - sy * dx;
    }

    if var_src < 1e-7 {
        return None;
    }

    let a = num_a / var_src;
    let b = -num_b / var_src;
    let c = num_b / var_src;
    let d = num_a / var_src;

    let tx = dst_mx - (a * src_mx + b * src_my);
    let ty = dst_my - (c * src_mx + d * src_my);

    Some(AffineMatrix2x3 { a, b, tx, c, d, ty })
}

/// Warps a region of src_image into dst_w x dst_h using matrix M (dst = M * src).
pub fn warp_affine_rgb(
    src_image: &[u8],
    src_w: u32,
    src_h: u32,
    matrix: &AffineMatrix2x3,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (dst_w * dst_h * 3) as usize];
    let inv = match matrix.inverse() {
        Some(m) => m,
        None => return out,
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let (sx, sy) = inv.transform_point(dx as f32, dy as f32);

            let dst_idx = ((dy * dst_w + dx) * 3) as usize;

            if sx >= 0.0 && sx < (src_w - 1) as f32 && sy >= 0.0 && sy < (src_h - 1) as f32 {
                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;

                let i00 = ((y0 * src_w + x0) * 3) as usize;
                let i10 = ((y0 * src_w + x1) * 3) as usize;
                let i01 = ((y1 * src_w + x0) * 3) as usize;
                let i11 = ((y1 * src_w + x1) * 3) as usize;

                for c in 0..3 {
                    let val = w00 * src_image[i00 + c] as f32
                        + w10 * src_image[i10 + c] as f32
                        + w01 * src_image[i01 + c] as f32
                        + w11 * src_image[i11 + c] as f32;
                    out[dst_idx + c] = val.round().clamp(0.0, 255.0) as u8;
                }
            } else if sx >= 0.0 && sx < src_w as f32 && sy >= 0.0 && sy < src_h as f32 {
                let x = sx.floor() as u32;
                let y = sy.floor() as u32;
                let idx = ((y * src_w + x) * 3) as usize;
                out[dst_idx..dst_idx + 3].copy_from_slice(&src_image[idx..idx + 3]);
            }
        }
    }

    out
}

/// Composites a 128x128 swapped face onto canvas_w x canvas_h using inv_matrix
/// (which maps 128x128 face coordinates to canvas coordinates).
pub fn composite_face_with_feather(
    canvas: &mut [u8],
    canvas_w: u32,
    canvas_h: u32,
    swapped_face_128: &[u8],
    inv_matrix: &AffineMatrix2x3,
    _feather_sigma: f32,
) {
    let forward_matrix = match inv_matrix.inverse() {
        Some(m) => m,
        None => return,
    };

    // Calculate bounding box on canvas
    let corners = [(0.0, 0.0), (128.0, 0.0), (128.0, 128.0), (0.0, 128.0)];
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for (cx, cy) in corners {
        let (tx, ty) = inv_matrix.transform_point(cx, cy);
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
    }

    let start_x = (min_x.floor() as i32).clamp(0, canvas_w as i32 - 1) as u32;
    let end_x = (max_x.ceil() as i32).clamp(0, canvas_w as i32 - 1) as u32;
    let start_y = (min_y.floor() as i32).clamp(0, canvas_h as i32 - 1) as u32;
    let end_y = (max_y.ceil() as i32).clamp(0, canvas_h as i32 - 1) as u32;

    for cy in start_y..=end_y {
        for cx in start_x..=end_x {
            let (fx, fy) = forward_matrix.transform_point(cx as f32, cy as f32);

            if fx >= 0.0 && fx < 127.0 && fy >= 0.0 && fy < 127.0 {
                // Elliptical mask distance from center (64, 64)
                let ndx = (fx - 64.0) / 60.0;
                let ndy = (fy - 64.0) / 60.0;
                let dist = (ndx * ndx + ndy * ndy).sqrt();

                let alpha = if dist <= 0.65 {
                    1.0f32
                } else if dist < 1.0 {
                    let t = (dist - 0.65) / 0.35;
                    1.0 - (3.0 * t * t - 2.0 * t * t * t)
                } else {
                    0.0f32
                };

                if alpha > 0.001 {
                    // Bilinear sample swapped face
                    let x0 = fx.floor() as u32;
                    let y0 = fy.floor() as u32;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    let sfx = fx - x0 as f32;
                    let sfy = fy - y0 as f32;

                    let i00 = ((y0 * 128 + x0) * 3) as usize;
                    let i10 = ((y0 * 128 + x1) * 3) as usize;
                    let i01 = ((y1 * 128 + x0) * 3) as usize;
                    let i11 = ((y1 * 128 + x1) * 3) as usize;

                    let c_idx = ((cy * canvas_w + cx) * 3) as usize;

                    for c in 0..3 {
                        let face_val = (1.0 - sfx) * (1.0 - sfy) * swapped_face_128[i00 + c] as f32
                            + sfx * (1.0 - sfy) * swapped_face_128[i10 + c] as f32
                            + (1.0 - sfx) * sfy * swapped_face_128[i01 + c] as f32
                            + sfx * sfy * swapped_face_128[i11 + c] as f32;

                        let orig_val = canvas[c_idx + c] as f32;
                        let blended = (1.0 - alpha) * orig_val + alpha * face_val;
                        canvas[c_idx + c] = blended.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
}

/// 1€ filter for a single float signal
struct LowPassFilter {
    y: Option<f32>,
    s: Option<f32>,
}

impl LowPassFilter {
    fn new() -> Self {
        Self { y: None, s: None }
    }

    fn filter(&mut self, value: f32, alpha: f32) -> f32 {
        let res = match self.s {
            Some(prev) => alpha * value + (1.0 - alpha) * prev,
            None => value,
        };
        self.y = Some(value);
        self.s = Some(res);
        res
    }
}

/// OneEuroFilter implementation for smoothing 1D scalar
pub struct OneEuroFilter {
    freq: f32,
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_filter: LowPassFilter,
    dx_filter: LowPassFilter,
    last_time: Option<f64>,
}

impl OneEuroFilter {
    pub fn new(freq: f32, min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            freq,
            min_cutoff,
            beta,
            d_cutoff,
            x_filter: LowPassFilter::new(),
            dx_filter: LowPassFilter::new(),
            last_time: None,
        }
    }

    fn alpha(rate: f32, cutoff: f32) -> f32 {
        let tau = 1.0 / (2.0 * PI * cutoff);
        let te = 1.0 / rate;
        1.0 / (1.0 + tau / te)
    }

    pub fn filter(&mut self, value: f32, timestamp: f64) -> f32 {
        let rate = if let Some(lt) = self.last_time {
            let dt = (timestamp - lt) as f32;
            if dt > 0.0 { 1.0 / dt } else { self.freq }
        } else {
            self.freq
        };
        self.last_time = Some(timestamp);

        let dx = match self.x_filter.y {
            Some(prev_x) => (value - prev_x) * rate,
            None => 0.0,
        };

        let edx = self.dx_filter.filter(dx, Self::alpha(rate, self.d_cutoff));
        let cutoff = self.min_cutoff + self.beta * edx.abs();
        self.x_filter.filter(value, Self::alpha(rate, cutoff))
    }
}

/// OneEuroFilter applied to all 6 parameters of AffineMatrix2x3
pub struct OneEuroFilter2x3 {
    filters: [OneEuroFilter; 6],
}

impl OneEuroFilter2x3 {
    pub fn new(freq: f32, min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            filters: [
                OneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
                OneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
                OneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
                OneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
                OneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
                OneEuroFilter::new(freq, min_cutoff, beta, d_cutoff),
            ],
        }
    }

    pub fn filter(&mut self, matrix: &AffineMatrix2x3, timestamp: f64) -> AffineMatrix2x3 {
        let a = self.filters[0].filter(matrix.a, timestamp);
        let b = self.filters[1].filter(matrix.b, timestamp);
        let tx = self.filters[2].filter(matrix.tx, timestamp);
        let c = self.filters[3].filter(matrix.c, timestamp);
        let d = self.filters[4].filter(matrix.d, timestamp);
        let ty = self.filters[5].filter(matrix.ty, timestamp);

        AffineMatrix2x3 { a, b, tx, c, d, ty }
    }
}
