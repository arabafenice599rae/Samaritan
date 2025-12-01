//! Network client for delta submission.

use anyhow::Result;

#[derive(Debug)]
pub struct NetClient {
    _id: crate::NodeId,
    _endpoint: Option<String>,
}

impl NetClient {
    pub fn new(id: crate::NodeId) -> Self {
        Self { _id: id, _endpoint: None }
    }

    pub async fn submit_delta(&self, _delta: DeltaMessage) -> Result<()> {
        Ok(())
    }

    pub fn set_endpoint(&mut self, endpoint: String) {
        self._endpoint = Some(endpoint);
    }
}

#[derive(Debug)]
pub struct DeltaMessage {}
