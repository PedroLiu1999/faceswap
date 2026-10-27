use std::fs;
use tempfile::tempdir;

use faceswap::pipeline::model_manager::{ModelError, ModelKind, ModelManager};

#[test]
fn test_model_paths_and_availability() {
    let tmp = tempdir().expect("failed to create temp dir");
    let manager = ModelManager::with_cache_dir(tmp.path().to_path_buf());

    let inswapper_path = manager.model_path(ModelKind::Inswapper128);
    assert_eq!(inswapper_path, tmp.path().join("inswapper_128.onnx"));

    // File doesn't exist yet
    assert!(!manager.is_available(ModelKind::Inswapper128));

    // Create empty file
    fs::write(&inswapper_path, b"test").expect("failed to write dummy file");
    assert!(manager.is_available(ModelKind::Inswapper128));
}

#[test]
fn test_verify_checksum_success_and_failure() {
    let tmp = tempdir().expect("failed to create temp dir");
    let manager = ModelManager::with_cache_dir(tmp.path().to_path_buf());

    // File not found error
    let err = manager.verify_checksum(ModelKind::Inswapper128);
    assert!(matches!(err, Err(ModelError::NotFound(_))));

    // Write file with invalid hash
    let path = manager.model_path(ModelKind::Inswapper128);
    fs::write(&path, b"corrupted data").expect("failed to write corrupted file");
    let is_valid = manager.verify_checksum(ModelKind::Inswapper128).expect("check should succeed");
    assert!(!is_valid, "corrupted file should fail checksum verification");
}

#[test]
fn test_custom_data_checksum_calculation() {
    let tmp = tempdir().expect("failed to create temp dir");
    let manager = ModelManager::with_cache_dir(tmp.path().to_path_buf());

    let test_file = tmp.path().join("sample.bin");
    fs::write(&test_file, b"hello faceswap").expect("failed to write sample");

    // Standard SHA-256 for b"hello faceswap"
    let hash = manager.calculate_sha256(&test_file).expect("hash calculation succeeded");
    assert_eq!(
        hash,
        "eaebbd32f21dba3a11cdf334cc8b2b548566f348f0715cd92141136fd1ea3522"
    );
}

#[test]
fn test_default_cache_dir_resolution() {
    let default_dir = ModelManager::default_cache_dir();
    assert!(default_dir.ends_with("faceswap/models") || default_dir.ends_with("faceswap\\models"));
}
