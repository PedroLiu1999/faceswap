use faceswap::pipeline::alignment::{
    composite_face_with_feather, estimate_similarity_transform, warp_affine_rgb, AffineMatrix2x3,
    OneEuroFilter2x3, STANDARD_TEMPLATE_128,
};

#[test]
fn test_umeyama_identity() {
    let template = STANDARD_TEMPLATE_128;
    let matrix = estimate_similarity_transform(&template, &template).expect("estimation failed");

    // With identical points, matrix should be identity
    assert!((matrix.a - 1.0).abs() < 1e-4, "matrix.a should be 1.0, got {}", matrix.a);
    assert!(matrix.b.abs() < 1e-4, "matrix.b should be 0.0, got {}", matrix.b);
    assert!(matrix.tx.abs() < 1e-4, "matrix.tx should be 0.0, got {}", matrix.tx);
    assert!(matrix.c.abs() < 1e-4, "matrix.c should be 0.0, got {}", matrix.c);
    assert!((matrix.d - 1.0).abs() < 1e-4, "matrix.d should be 1.0, got {}", matrix.d);
    assert!(matrix.ty.abs() < 1e-4, "matrix.ty should be 0.0, got {}", matrix.ty);
}

#[test]
fn test_matrix_inverse_and_point_roundtrip() {
    let matrix = AffineMatrix2x3 {
        a: 0.8,
        b: -0.6,
        tx: 45.0,
        c: 0.6,
        d: 0.8,
        ty: -30.0,
    };

    let inv = matrix.inverse().expect("matrix must be invertible");

    let p = (100.0, 75.0);
    let (tx, ty) = matrix.transform_point(p.0, p.1);
    let (rx, ry) = inv.transform_point(tx, ty);

    assert!((rx - p.0).abs() < 1e-3, "recovered X mismatch: {} vs {}", rx, p.0);
    assert!((ry - p.1).abs() < 1e-3, "recovered Y mismatch: {} vs {}", ry, p.1);
}

#[test]
fn test_warp_affine_rgb() {
    // 200x200 image, fill with uniform color (120, 150, 200)
    let w = 200;
    let h = 200;
    let src = vec![120u8, 150, 200].repeat((w * h) as usize);

    let matrix = AffineMatrix2x3::identity();
    let warped = warp_affine_rgb(&src, w, h, &matrix, 128, 128);

    assert_eq!(warped.len(), 128 * 128 * 3);
    // Interior pixels should match source color
    assert_eq!(warped[0], 120);
    assert_eq!(warped[1], 150);
    assert_eq!(warped[2], 200);
}

#[test]
fn test_composite_face_with_feather() {
    let canvas_w = 200;
    let canvas_h = 200;
    let mut canvas = vec![0u8; (canvas_w * canvas_h * 3) as usize]; // pure black

    // 128x128 pure white face
    let face_128 = vec![255u8; 128 * 128 * 3];

    // Identity inverse positioned at (36, 36)
    let inv_matrix = AffineMatrix2x3 {
        a: 1.0,
        b: 0.0,
        tx: 36.0,
        c: 0.0,
        d: 1.0,
        ty: 36.0,
    };

    composite_face_with_feather(&mut canvas, canvas_w, canvas_h, &face_128, &inv_matrix, 10.0);

    // Center of face should be bright white (255)
    let center_idx = ((36 + 64) * canvas_w + (36 + 64)) as usize * 3;
    assert!(canvas[center_idx] > 240, "center should be solid white");

    // Outer corner (0, 0) should remain pure black (0)
    assert_eq!(canvas[0], 0);
    assert_eq!(canvas[1], 0);
    assert_eq!(canvas[2], 0);
}

#[test]
fn test_one_euro_filter_smoothing() {
    let mut filter = OneEuroFilter2x3::new(30.0, 1.0, 0.007, 1.0);

    let base = AffineMatrix2x3::identity();
    let mut noisy_samples = Vec::new();

    // 10 frames with slight jitter (+/- 0.5 px)
    for i in 0..10 {
        let noise = if i % 2 == 0 { 0.5 } else { -0.5 };
        let m = AffineMatrix2x3 {
            tx: base.tx + noise,
            ty: base.ty - noise,
            ..base
        };
        noisy_samples.push(filter.filter(&m, i as f64 / 30.0));
    }

    // Filtered jitter should be substantially damped compared to raw +/- 0.5 noise
    let last = &noisy_samples[9];
    assert!(last.tx.abs() < 0.35, "jitter should be smoothed: {}", last.tx);
}
