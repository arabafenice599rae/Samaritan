//! Snapshot storage for model versions.

use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug)]
pub struct SnapshotStore {
    _data_dir: PathBuf,
}

impl SnapshotStore {
    pub async fn open(data_dir: PathBuf) -> Result<Self> {
        Ok(Self { _data_dir: data_dir })
    }

    pub async fn create_snapshot<B>(&mut self, _engine: &crate::neural_engine::NeuralEngine<B>) -> Result<()> {
        Ok(())
    }
}
