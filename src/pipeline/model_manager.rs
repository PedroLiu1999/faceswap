use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model file not found at path: {0}")]
    NotFound(PathBuf),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Checksum mismatch for {model:?}: expected {expected}, actual {actual}")]
    ChecksumMismatch {
        model: ModelKind,
        expected: String,
        actual: String,
    },

    #[error("Download error: {0}")]
    DownloadFailed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    Inswapper128,
    Scrfd2_5g,
    ArcFaceW600k,
}

impl ModelKind {
    pub fn filename(&self) -> &'static str {
        match self {
            ModelKind::Inswapper128 => "inswapper_128.onnx",
            ModelKind::Scrfd2_5g => "scrfd_2.5g_bnkps.onnx",
            ModelKind::ArcFaceW600k => "w600k_r50.onnx",
        }
    }

    pub fn expected_sha256(&self) -> &'static str {
        match self {
            // Official sha256 hashes for InsightFace weights
            ModelKind::Inswapper128 => "e4a3f08c5365edd32de62ff7071f654b706c9a9d20c56c2d47e4526541f5a536",
            ModelKind::Scrfd2_5g => "6236b2f7041a31d0ec2e99dcf5c2cf5e6be76e48b598d197af534f3c7e3ff75c",
            ModelKind::ArcFaceW600k => "2be162eb351918a28723a1ef5d45d4c82b04f727376c94441584c0e3a6c11a0d",
        }
    }

    pub fn download_url(&self) -> &'static str {
        match self {
            ModelKind::Inswapper128 => "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
            ModelKind::Scrfd2_5g => "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_2.5g_bnkps.onnx",
            ModelKind::ArcFaceW600k => "https://huggingface.co/MonsterMMORPG/tools/resolve/main/w600k_r50.onnx",
        }
    }
}

pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::with_cache_dir(Self::default_cache_dir())
    }

    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    pub fn default_cache_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("faceswap")
            .join("models")
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn model_path(&self, kind: ModelKind) -> PathBuf {
        self.cache_dir.join(kind.filename())
    }

    pub fn is_available(&self, kind: ModelKind) -> bool {
        self.model_path(kind).exists()
    }

    pub fn calculate_sha256(&self, path: &Path) -> Result<String, ModelError> {
        if !path.exists() {
            return Err(ModelError::NotFound(path.to_path_buf()));
        }

        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    pub fn verify_checksum(&self, kind: ModelKind) -> Result<bool, ModelError> {
        let path = self.model_path(kind);
        let actual = self.calculate_sha256(&path)?;
        Ok(actual.eq_ignore_ascii_case(kind.expected_sha256()))
    }

    pub fn download_model<F>(&self, kind: ModelKind, on_progress: Option<F>) -> Result<PathBuf, ModelError>
    where
        F: Fn(DownloadProgress),
    {
        fs::create_dir_all(&self.cache_dir)?;
        let target_path = self.model_path(kind);

        // If file exists and checksum matches, return immediately
        if target_path.exists() && self.verify_checksum(kind).unwrap_or(false) {
            return Ok(target_path);
        }

        let tmp_path = self.cache_dir.join(format!("{}.tmp", kind.filename()));
        let url = kind.download_url();

        let response = ureq::get(url)
            .call()
            .map_err(|e| ModelError::DownloadFailed(format!("Failed to connect to {url}: {e}")))?;

        let total_bytes = response
            .header("Content-Length")
            .and_then(|h| h.parse::<u64>().ok());

        let mut reader = response.into_reader();
        let mut writer = File::create(&tmp_path)?;
        let mut downloaded_bytes = 0u64;
        let mut buffer = [0u8; 64 * 1024];

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            writer.write_all(&buffer[..n])?;
            downloaded_bytes += n as u64;

            if let Some(ref cb) = on_progress {
                cb(DownloadProgress {
                    downloaded_bytes,
                    total_bytes,
                });
            }
        }
        writer.flush()?;
        drop(writer);

        // Verify checksum of downloaded temp file
        let actual_hash = self.calculate_sha256(&tmp_path)?;
        if !actual_hash.eq_ignore_ascii_case(kind.expected_sha256()) {
            let _ = fs::remove_file(&tmp_path);
            return Err(ModelError::ChecksumMismatch {
                model: kind,
                expected: kind.expected_sha256().to_string(),
                actual: actual_hash,
            });
        }

        // Atomically rename .tmp to final path
        fs::rename(&tmp_path, &target_path)?;
        Ok(target_path)
    }

    pub fn ensure_all_models<F>(&self, on_progress: Option<F>) -> Result<Vec<PathBuf>, ModelError>
    where
        F: Fn(ModelKind, DownloadProgress) + Copy,
    {
        let models = [
            ModelKind::Inswapper128,
            ModelKind::Scrfd2_5g,
            ModelKind::ArcFaceW600k,
        ];

        let mut paths = Vec::with_capacity(models.len());
        for kind in models {
            let progress_cb = on_progress.map(|cb| move |p| cb(kind, p));
            let path = self.download_model(kind, progress_cb)?;
            paths.push(path);
        }

        Ok(paths)
    }
}
