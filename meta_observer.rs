//! Meta observer for metrics.

#[derive(Debug)]
pub struct MetaObserver {}

impl MetaObserver {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn sample<B>(&mut self, _engine: &crate::neural_engine::NeuralEngine<B>) {}
}

impl Default for MetaObserver {
    fn default() -> Self {
        Self::new()
    }
}
