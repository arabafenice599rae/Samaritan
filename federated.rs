//! Federated learning state.

use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug)]
pub struct FederatedState {
    _data_dir: PathBuf,
}

impl FederatedState {
    pub async fn new(data_dir: PathBuf) -> Result<Self> {
        Ok(Self { _data_dir: data_dir })
    }

    pub async fn is_training_enabled(&self) -> Result<bool> {
        Ok(false)
    }

    pub async fn run_local_epoch(&mut self, _intensity: f64) -> Result<()> {
        Ok(())
    }

    pub fn should_submit_delta(&self) -> bool {
        false
    }

    pub async fn compute_and_package_delta(&mut self) -> Result<crate::net::DeltaMessage> {
        Ok(crate::net::DeltaMessage {})
    }

    pub fn disable_training(&mut self) {}
}
