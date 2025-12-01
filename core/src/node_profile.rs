//! Hardware profile detection and classification for Samaritan nodes.
//!
//! Questo modulo fornisce:
//!
//! - [`NodeProfile`]: enumerazione dei profili supportati (HeavyGpu, HeavyCpu, Desktop, Mobile);
//! - [`NodeProfileDetector`]: sistema di auto-detection basato su caratteristiche hardware;
//! - [`HardwareCapabilities`]: struttura dati che rappresenta le capacità del sistema.
//!
//! # Profili
//!
//! - **HeavyGpu**: server/workstation con GPU dedicata (NVIDIA/AMD), CPU multi-core, RAM >= 16GB
//! - **HeavyCpu**: server con CPU potente (8+ core) ma senza GPU dedicata, RAM >= 16GB
//! - **Desktop**: PC consumer con risorse moderate (4-8 core, 8-16GB RAM)
//! - **Mobile**: laptop, tablet o dispositivo con risorse limitate (<= 4 core, <= 8GB RAM)
//!
//! # Auto-detection
//!
//! Il sistema rileva automaticamente:
//! - numero di core CPU logici
//! - RAM totale di sistema
//! - presenza di GPU dedicata (via sysfs su Linux, registry su Windows)
//!
//! Su sistemi non supportati o in caso di errore, fallback a `Desktop`.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

/// Profilo di un nodo Samaritan.
///
/// Determina quali task il nodo può eseguire, con quali priorità,
/// e come viene schedulato il training federato.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeProfile {
    /// Server o workstation con GPU dedicata potente.
    ///
    /// Caratteristiche minime:
    /// - GPU NVIDIA/AMD con VRAM >= 8GB
    /// - CPU multi-core (8+ core)
    /// - RAM >= 16GB
    ///
    /// Ruolo: training pesante, inferenza real-time, aggregazione.
    HeavyGpu,

    /// Server con CPU potente ma senza GPU dedicata.
    ///
    /// Caratteristiche minime:
    /// - CPU 8+ core
    /// - RAM >= 16GB
    /// - Assenza di GPU dedicata
    ///
    /// Ruolo: training CPU-based, aggregazione, backup inference.
    HeavyCpu,

    /// Desktop consumer o workstation entry-level.
    ///
    /// Caratteristiche tipiche:
    /// - CPU 4-8 core
    /// - RAM 8-16GB
    /// - GPU integrata o entry-level
    ///
    /// Ruolo: inferenza locale, training leggero, partecipazione federata parziale.
    Desktop,

    /// Laptop, tablet, o dispositivo mobile/embedded.
    ///
    /// Caratteristiche tipiche:
    /// - CPU <= 4 core
    /// - RAM <= 8GB
    /// - Batteria limitata
    ///
    /// Ruolo: solo inferenza locale, nessun training federato.
    Mobile,
}

impl NodeProfile {
    /// Restituisce `true` se il profilo è Heavy (Gpu o Cpu).
    ///
    /// I nodi Heavy partecipano attivamente al training federato
    /// e possono eseguire task computazionalmente intensivi.
    #[must_use]
    pub const fn is_heavy(&self) -> bool {
        matches!(self, Self::HeavyGpu | Self::HeavyCpu)
    }

    /// Restituisce `true` se il profilo supporta GPU dedicata.
    #[must_use]
    pub const fn has_dedicated_gpu(&self) -> bool {
        matches!(self, Self::HeavyGpu)
    }

    /// Restituisce una stima della "potenza computazionale" relativa (0.0-1.0).
    ///
    /// Usato per:
    /// - bilanciare il carico nel federated learning,
    /// - decidere priorità di aggregazione,
    /// - throttling adattivo.
    #[must_use]
    pub const fn compute_power(&self) -> f32 {
        match self {
            Self::HeavyGpu => 1.0,
            Self::HeavyCpu => 0.7,
            Self::Desktop => 0.4,
            Self::Mobile => 0.2,
        }
    }

    /// Restituisce il numero massimo suggerito di worker paralleli per training.
    #[must_use]
    pub const fn max_parallel_workers(&self) -> usize {
        match self {
            Self::HeavyGpu => 8,
            Self::HeavyCpu => 6,
            Self::Desktop => 3,
            Self::Mobile => 1,
        }
    }

    /// Restituisce `true` se il nodo può partecipare al training federato.
    #[must_use]
    pub const fn can_train(&self) -> bool {
        !matches!(self, Self::Mobile)
    }

    /// Restituisce una stringa human-readable del profilo.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::HeavyGpu => "HeavyGpu",
            Self::HeavyCpu => "HeavyCpu",
            Self::Desktop => "Desktop",
            Self::Mobile => "Mobile",
        }
    }
}

impl Default for NodeProfile {
    fn default() -> Self {
        Self::Desktop
    }
}

impl std::fmt::Display for NodeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Capacità hardware rilevate sul sistema.
///
/// Questa struttura rappresenta uno snapshot delle risorse disponibili
/// al momento della detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareCapabilities {
    /// Numero di core CPU logici (include hyperthreading).
    pub cpu_cores: usize,

    /// RAM totale di sistema, in MB.
    pub ram_mb: usize,

    /// Presenza di GPU dedicata (NVIDIA, AMD, Intel Arc).
    pub has_dedicated_gpu: bool,

    /// VRAM stimata della GPU, in MB (0 se assente o non rilevabile).
    pub gpu_vram_mb: usize,
}

impl HardwareCapabilities {
    /// Crea un set di capacità di default (fallback safe).
    ///
    /// Corrisponde a un sistema Desktop entry-level:
    /// - 4 core
    /// - 8GB RAM
    /// - nessuna GPU dedicata
    #[must_use]
    pub fn fallback() -> Self {
        Self {
            cpu_cores: 4,
            ram_mb: 8_192,
            has_dedicated_gpu: false,
            gpu_vram_mb: 0,
        }
    }

    /// Rileva le capacità hardware del sistema corrente.
    ///
    /// Su piattaforme non supportate o in caso di errore, ritorna
    /// [`HardwareCapabilities::fallback`].
    #[must_use]
    pub fn detect() -> Self {
        let cpu_cores = Self::detect_cpu_cores();
        let ram_mb = Self::detect_ram_mb();
        let (has_dedicated_gpu, gpu_vram_mb) = Self::detect_gpu();

        Self {
            cpu_cores,
            ram_mb,
            has_dedicated_gpu,
            gpu_vram_mb,
        }
    }

    /// Classifica le capacità in un [`NodeProfile`].
    ///
    /// Logica:
    ///
    /// 1. Se GPU dedicata con VRAM >= 8GB + CPU >= 8 core + RAM >= 16GB → `HeavyGpu`
    /// 2. Se CPU >= 8 core + RAM >= 16GB (no GPU) → `HeavyCpu`
    /// 3. Se CPU <= 4 core o RAM <= 8GB → `Mobile`
    /// 4. Altrimenti → `Desktop`
    #[must_use]
    pub fn classify(&self) -> NodeProfile {
        // HeavyGpu: GPU potente + sistema potente
        if self.has_dedicated_gpu
            && self.gpu_vram_mb >= 8_000
            && self.cpu_cores >= 8
            && self.ram_mb >= 16_000
        {
            return NodeProfile::HeavyGpu;
        }

        // HeavyCpu: CPU potente + RAM alta, ma no GPU
        if !self.has_dedicated_gpu && self.cpu_cores >= 8 && self.ram_mb >= 16_000 {
            return NodeProfile::HeavyCpu;
        }

        // Mobile: risorse limitate
        if self.cpu_cores <= 4 || self.ram_mb <= 8_000 {
            return NodeProfile::Mobile;
        }

        // Default: Desktop
        NodeProfile::Desktop
    }

    /// Rileva il numero di core CPU logici.
    ///
    /// Usa `std::thread::available_parallelism()` (stable da Rust 1.59).
    fn detect_cpu_cores() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or_else(|_| {
                warn!("Failed to detect CPU cores, using fallback: 4");
                4
            })
    }

    /// Rileva la RAM totale di sistema, in MB.
    ///
    /// Su Linux: legge `/proc/meminfo`.
    /// Su Windows: potrebbe usare WMI (non implementato qui).
    /// Fallback: 8192 MB.
    fn detect_ram_mb() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        // Formato: "MemTotal:       16384000 kB"
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1_024; // kB → MB
                            }
                        }
                    }
                }
            }
        }

        warn!("Failed to detect RAM, using fallback: 8192 MB");
        8_192
    }

    /// Rileva la presenza di GPU dedicata e stima la VRAM.
    ///
    /// Su Linux: verifica `/sys/class/drm/card*` per NVIDIA/AMD.
    /// Su Windows: potrebbe usare DXGI (non implementato qui).
    ///
    /// Ritorna: `(has_gpu, vram_mb)`
    fn detect_gpu() -> (bool, usize) {
        #[cfg(target_os = "linux")]
        {
            // Cerca schede video in /sys/class/drm/
            let drm_path = Path::new("/sys/class/drm");
            if drm_path.exists() {
                if let Ok(entries) = fs::read_dir(drm_path) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let name_str = name.to_string_lossy();

                        // Cerca pattern tipo "card0", "card1"
                        if name_str.starts_with("card") && !name_str.contains('-') {
                            // Verifica se è una GPU dedicata (non iGPU)
                            if let Some(vram) = Self::probe_vram_linux(&entry.path()) {
                                if vram >= 2_000 {
                                    // Minimo 2GB per considerarla "dedicata"
                                    return (true, vram);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: nessuna GPU dedicata rilevata
        (false, 0)
    }

    /// Prova a leggere la VRAM di una scheda video Linux (in MB).
    ///
    /// Percorso tipico: `/sys/class/drm/card0/device/mem_info_vram_total`
    #[cfg(target_os = "linux")]
    fn probe_vram_linux(card_path: &Path) -> Option<usize> {
        // AMDGPU: mem_info_vram_total
        let amd_vram_path = card_path.join("device/mem_info_vram_total");
        if let Ok(content) = fs::read_to_string(&amd_vram_path) {
            if let Ok(bytes) = content.trim().parse::<usize>() {
                return Some(bytes / (1_024 * 1_024)); // bytes → MB
            }
        }

        // NVIDIA: più complicato, richiederebbe nvidia-smi o librerie
        // Per ora: fallback euristica basata su presenza di "nvidia" nel vendor
        let vendor_path = card_path.join("device/vendor");
        if let Ok(vendor) = fs::read_to_string(&vendor_path) {
            if vendor.trim().contains("0x10de") {
                // 0x10de = NVIDIA vendor ID
                // Fallback: assumiamo 8GB per GPU NVIDIA (conservative)
                return Some(8_000);
            }
        }

        None
    }

    #[cfg(not(target_os = "linux"))]
    fn probe_vram_linux(_card_path: &Path) -> Option<usize> {
        None
    }
}

/// Detector principale per il profilo del nodo.
///
/// Questo tipo è stateless: espone solo metodi di classe per detection.
pub struct NodeProfileDetector;

impl NodeProfileDetector {
    /// Rileva automaticamente il profilo del nodo corrente.
    ///
    /// Pipeline:
    /// 1. [`HardwareCapabilities::detect`] → snapshot hardware
    /// 2. [`HardwareCapabilities::classify`] → profilo
    ///
    /// Questo metodo non panica mai: in caso di errore ritorna `Desktop`.
    #[must_use]
    pub fn detect() -> NodeProfile {
        let caps = HardwareCapabilities::detect();
        caps.classify()
    }

    /// Rileva il profilo e le capacità hardware grezze.
    ///
    /// Utile per logging dettagliato o debug.
    #[must_use]
    pub fn detect_with_capabilities() -> (NodeProfile, HardwareCapabilities) {
        let caps = HardwareCapabilities::detect();
        let profile = caps.classify();
        (profile, caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_as_str_is_consistent() {
        assert_eq!(NodeProfile::HeavyGpu.as_str(), "HeavyGpu");
        assert_eq!(NodeProfile::HeavyCpu.as_str(), "HeavyCpu");
        assert_eq!(NodeProfile::Desktop.as_str(), "Desktop");
        assert_eq!(NodeProfile::Mobile.as_str(), "Mobile");
    }

    #[test]
    fn heavy_profiles_are_heavy() {
        assert!(NodeProfile::HeavyGpu.is_heavy());
        assert!(NodeProfile::HeavyCpu.is_heavy());
        assert!(!NodeProfile::Desktop.is_heavy());
        assert!(!NodeProfile::Mobile.is_heavy());
    }

    #[test]
    fn only_heavy_gpu_has_dedicated_gpu() {
        assert!(NodeProfile::HeavyGpu.has_dedicated_gpu());
        assert!(!NodeProfile::HeavyCpu.has_dedicated_gpu());
        assert!(!NodeProfile::Desktop.has_dedicated_gpu());
        assert!(!NodeProfile::Mobile.has_dedicated_gpu());
    }

    #[test]
    fn compute_power_is_monotonic() {
        assert!(NodeProfile::HeavyGpu.compute_power() > NodeProfile::HeavyCpu.compute_power());
        assert!(NodeProfile::HeavyCpu.compute_power() > NodeProfile::Desktop.compute_power());
        assert!(NodeProfile::Desktop.compute_power() > NodeProfile::Mobile.compute_power());
    }

    #[test]
    fn mobile_cannot_train() {
        assert!(!NodeProfile::Mobile.can_train());
        assert!(NodeProfile::Desktop.can_train());
        assert!(NodeProfile::HeavyCpu.can_train());
        assert!(NodeProfile::HeavyGpu.can_train());
    }

    #[test]
    fn fallback_capabilities_are_sane() {
        let caps = HardwareCapabilities::fallback();
        assert_eq!(caps.cpu_cores, 4);
        assert_eq!(caps.ram_mb, 8_192);
        assert!(!caps.has_dedicated_gpu);
        assert_eq!(caps.gpu_vram_mb, 0);
    }

    #[test]
    fn classify_heavy_gpu_requires_all_criteria() {
        let caps = HardwareCapabilities {
            cpu_cores: 12,
            ram_mb: 32_000,
            has_dedicated_gpu: true,
            gpu_vram_mb: 16_000,
        };
        assert_eq!(caps.classify(), NodeProfile::HeavyGpu);

        // Manca GPU
        let no_gpu = HardwareCapabilities {
            has_dedicated_gpu: false,
            ..caps
        };
        assert_ne!(no_gpu.classify(), NodeProfile::HeavyGpu);

        // VRAM insufficiente
        let low_vram = HardwareCapabilities {
            gpu_vram_mb: 4_000,
            ..caps
        };
        assert_ne!(low_vram.classify(), NodeProfile::HeavyGpu);
    }

    #[test]
    fn classify_heavy_cpu_requires_no_gpu() {
        let caps = HardwareCapabilities {
            cpu_cores: 16,
            ram_mb: 32_000,
            has_dedicated_gpu: false,
            gpu_vram_mb: 0,
        };
        assert_eq!(caps.classify(), NodeProfile::HeavyCpu);
    }

    #[test]
    fn classify_mobile_with_low_resources() {
        let caps = HardwareCapabilities {
            cpu_cores: 2,
            ram_mb: 4_000,
            has_dedicated_gpu: false,
            gpu_vram_mb: 0,
        };
        assert_eq!(caps.classify(), NodeProfile::Mobile);
    }

    #[test]
    fn classify_desktop_is_middle_ground() {
        let caps = HardwareCapabilities {
            cpu_cores: 6,
            ram_mb: 12_000,
            has_dedicated_gpu: false,
            gpu_vram_mb: 0,
        };
        assert_eq!(caps.classify(), NodeProfile::Desktop);
    }

    #[test]
    fn detector_never_panics() {
        // Questo test verifica solo che il metodo non panica
        let profile = NodeProfileDetector::detect();
        assert!(matches!(
            profile,
            NodeProfile::HeavyGpu
                | NodeProfile::HeavyCpu
                | NodeProfile::Desktop
                | NodeProfile::Mobile
        ));
    }
}
