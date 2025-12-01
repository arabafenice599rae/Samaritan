//! Update agent for binary updates.

use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug)]
pub struct UpdateAgent {
    _data_dir: PathBuf,
}

impl UpdateAgent {
    pub fn new(data_dir: PathBuf) -> Self {
        Self { _data_dir: data_dir }
    }

    pub async fn check_for_updates(&self) -> Result<()> {
        Ok(())
    }
}
