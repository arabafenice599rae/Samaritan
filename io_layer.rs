//! I/O layer for user interactions.

use anyhow::Result;
use std::path::PathBuf;

/// Layer di I/O verso l'utente.
#[derive(Debug)]
pub struct IOLayer {
    _data_dir: PathBuf,
}

impl IOLayer {
    pub async fn new(data_dir: PathBuf) -> Result<Self> {
        Ok(Self { _data_dir: data_dir })
    }

    pub fn try_recv_user_input(&mut self) -> Option<String> {
        None
    }

    pub fn prepare_model_inputs(&self, _input: String) -> Result<ModelInput> {
        Ok(ModelInput {})
    }

    pub async fn deliver_to_user(&mut self, _decision: PolicyDecision) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct ModelInput {}

#[derive(Debug)]
pub struct PolicyDecision {}
