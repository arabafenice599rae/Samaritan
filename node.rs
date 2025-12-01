//! Node configuration and runner.

use anyhow::Result;
use std::path::{Path, PathBuf};
use serde::Deserialize;

use crate::node_profile::NodeProfile;
use crate::NeuroNode;

#[derive(Debug, Clone, Deserialize)]
pub struct NodeConfig {
    pub data_dir: PathBuf,
    pub model_path: PathBuf,
    #[serde(default)]
    pub profile_override: Option<NodeProfile>,
    #[serde(default)]
    pub federated: FederatedConfig,
    #[serde(default)]
    pub policy: PolicyConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FederatedConfig {
    #[serde(default = "FederatedConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub endpoint: Option<String>,
}

impl FederatedConfig {
    fn default_enabled() -> bool {
        true
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PolicyConfig {
    #[serde(default)]
    pub strict_mode: bool,
}

impl NodeConfig {
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let cfg: NodeConfig = serde_yaml::from_str(&raw)?;
        Ok(cfg)
    }

    pub fn load_default() -> Result<Self> {
        Self::from_yaml(Path::new("samaritan.yaml"))
    }
}

pub async fn build_node(_config: &NodeConfig) -> Result<NeuroNode> {
    unimplemented!("build_node")
}

pub async fn run_node(_node: NeuroNode) -> ! {
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}
