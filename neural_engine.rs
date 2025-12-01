//! Neural engine with ONNX backend.

use anyhow::Result;
use std::path::Path;

pub struct NeuralEngine<B> {
    _backend: B,
}

impl<B> NeuralEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { _backend: backend }
    }

    pub async fn infer(&self, _input: &crate::io_layer::ModelInput) -> Result<ModelOutput> {
        Ok(ModelOutput {})
    }
}

pub struct OnnxBackend {}

impl OnnxBackend {
    pub fn load(_path: &Path) -> Result<Self> {
        Ok(Self {})
    }
}

#[derive(Debug)]
pub struct ModelOutput {}
