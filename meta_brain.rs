//! Meta brain for ADR and distillation.

#[derive(Debug)]
pub struct MetaBrain {}

impl MetaBrain {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for MetaBrain {
    fn default() -> Self {
        Self::new()
    }
}
