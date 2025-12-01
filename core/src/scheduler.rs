//! Priority scheduler for Samaritan NeuroNode tasks.
//!
//! Questo modulo implementa uno scheduler a tre corsie di priorità:
//!
//! - **Critical**: inferenza utente, policy, I/O — latenza minima
//! - **Normal**: training locale, delta computation — priorità media
//! - **Background**: snapshot, meta-observer, aggiornamenti — best-effort
//!
//! Lo scheduler NON è un vero task scheduler preemptive: è una struttura
//! decisionale che aiuta il [`NeuroNode::tick`] a scegliere cosa eseguire
//! in base a priorità, throttle e carico.
//!
//! # Funzionamento
//!
//! Ad ogni tick, il nodo chiama [`PriorityScheduler::schedule_tick`] che:
//!
//! 1. determina quali lane possono essere eseguite (based on throttle),
//! 2. assegna slot temporali alle diverse lane,
//! 3. restituisce una struttura [`ScheduledWork`] che indica cosa fare.
//!
//! Il nodo poi interpreta questo piano ed esegue i task appropriati.

use std::collections::VecDeque;

/// Corsia di priorità per i task del nodo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// Massima priorità: inferenza utente, policy, I/O.
    ///
    /// Questi task devono essere eseguiti immediatamente per garantire
    /// responsività al user.
    Critical,

    /// Priorità normale: training locale, delta computation.
    ///
    /// Questi task contribuiscono al federated learning ma possono
    /// essere ritardati se la latenza è alta.
    Normal,

    /// Priorità bassa: snapshot, meta-observer, aggiornamenti.
    ///
    /// Questi task sono best-effort e vengono skippati sotto throttle.
    Background,
}

impl Lane {
    /// Restituisce il peso relativo della lane (per scheduling).
    ///
    /// Critical = 10, Normal = 5, Background = 1
    #[must_use]
    pub const fn weight(&self) -> u32 {
        match self {
            Self::Critical => 10,
            Self::Normal => 5,
            Self::Background => 1,
        }
    }

    /// Restituisce `true` se la lane è considerata background.
    ///
    /// Usato dal throttle per decidere se skippare questa lane.
    #[must_use]
    pub const fn is_background(&self) -> bool {
        matches!(self, Self::Background)
    }

    /// Restituisce una stringa human-readable della lane.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Critical => "Critical",
            Self::Normal => "Normal",
            Self::Background => "Background",
        }
    }
}

impl std::fmt::Display for Lane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Tipo di task che può essere schedulato.
///
/// Questa è una rappresentazione logica, non un vero task eseguibile.
/// Il [`NeuroNode`] interpreta questo enum e chiama i metodi appropriati.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// Inferenza per richiesta utente.
    UserInference,

    /// Valutazione policy su output del modello.
    PolicyEvaluation,

    /// Consegna risposta all'utente (I/O).
    UserDelivery,

    /// Epoch di training locale (DP-SGD).
    LocalTraining,

    /// Calcolo e compressione del delta federato.
    DeltaComputation,

    /// Invio del delta al server federato.
    DeltaSubmission,

    /// Campionamento metriche dal meta-observer.
    MetricsSampling,

    /// Creazione snapshot del modello.
    SnapshotCreation,

    /// Controllo aggiornamenti binari.
    UpdateCheck,

    /// Applicazione ADR dal meta-brain.
    AdrApplication,
}

impl TaskKind {
    /// Restituisce la lane di appartenenza del task.
    #[must_use]
    pub const fn lane(&self) -> Lane {
        match self {
            Self::UserInference | Self::PolicyEvaluation | Self::UserDelivery => Lane::Critical,
            Self::LocalTraining | Self::DeltaComputation | Self::DeltaSubmission => Lane::Normal,
            Self::MetricsSampling
            | Self::SnapshotCreation
            | Self::UpdateCheck
            | Self::AdrApplication => Lane::Background,
        }
    }

    /// Stima approssimativa del costo computazionale del task (0.0-1.0).
    ///
    /// Usato per bilanciare il carico tra lane.
    #[must_use]
    pub const fn cost(&self) -> f64 {
        match self {
            Self::UserInference => 0.3,
            Self::PolicyEvaluation => 0.05,
            Self::UserDelivery => 0.01,
            Self::LocalTraining => 0.8,
            Self::DeltaComputation => 0.2,
            Self::DeltaSubmission => 0.1,
            Self::MetricsSampling => 0.02,
            Self::SnapshotCreation => 0.4,
            Self::UpdateCheck => 0.05,
            Self::AdrApplication => 0.1,
        }
    }
}

/// Piano di lavoro schedulato per un singolo tick.
///
/// Contiene la lista di task che dovrebbero essere eseguiti,
/// ordinati per priorità (Critical → Normal → Background).
#[derive(Debug, Clone)]
pub struct ScheduledWork {
    /// Task da eseguire, ordinati per priorità.
    pub tasks: Vec<TaskKind>,

    /// Budget temporale stimato per questo tick (frazione 0.0-1.0 del tick time).
    ///
    /// 1.0 = tick "pieno", 0.5 = metà tick, ecc.
    /// Usato in combinazione con throttle intensity.
    pub budget: f64,

    /// Indica se le lane background sono attive in questo tick.
    pub background_active: bool,
}

impl ScheduledWork {
    /// Crea un piano di lavoro vuoto.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            tasks: Vec::new(),
            budget: 0.0,
            background_active: false,
        }
    }

    /// Restituisce `true` se il piano contiene almeno un task.
    #[must_use]
    pub fn has_work(&self) -> bool {
        !self.tasks.is_empty()
    }

    /// Restituisce il numero totale di task schedulati.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Restituisce il costo computazionale totale stimato (somma dei costi).
    #[must_use]
    pub fn total_cost(&self) -> f64 {
        self.tasks.iter().map(|t| t.cost()).sum()
    }
}

/// Configurazione dello scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Peso della lane Critical (default: 10).
    pub critical_weight: u32,

    /// Peso della lane Normal (default: 5).
    pub normal_weight: u32,

    /// Peso della lane Background (default: 1).
    pub background_weight: u32,

    /// Budget massimo per tick (frazione 0.0-1.0).
    ///
    /// Lo scheduler non schedulerà mai più task di questo budget.
    pub max_budget_per_tick: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            critical_weight: 10,
            normal_weight: 5,
            background_weight: 1,
            max_budget_per_tick: 0.9, // Lascia 10% di margine
        }
    }
}

/// Scheduler a priorità per il NeuroNode.
///
/// Mantiene code separate per ogni lane e decide quali task eseguire
/// in base a priorità, throttle e budget temporale.
#[derive(Debug)]
pub struct PriorityScheduler {
    config: SchedulerConfig,

    // Code di task pendenti per lane
    critical_queue: VecDeque<TaskKind>,
    normal_queue: VecDeque<TaskKind>,
    background_queue: VecDeque<TaskKind>,

    // Statistiche
    ticks_scheduled: u64,
    tasks_executed: u64,
}

impl PriorityScheduler {
    /// Crea un nuovo scheduler con configurazione di default.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Crea un nuovo scheduler con configurazione custom.
    #[must_use]
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            config,
            critical_queue: VecDeque::new(),
            normal_queue: VecDeque::new(),
            background_queue: VecDeque::new(),
            ticks_scheduled: 0,
            tasks_executed: 0,
        }
    }

    /// Schedula il lavoro per un singolo tick.
    ///
    /// # Parametri
    ///
    /// * `tick_number` - numero del tick corrente (per logging/debug)
    ///
    /// # Nota
    ///
    /// Nella versione attuale, lo scheduler è "stateless" nel senso che
    /// non mantiene una coda di task reale. I task sono implicitamente
    /// definiti dalla logica del [`NeuroNode::tick`].
    ///
    /// Questa funzione restituisce un piano "suggerito" piuttosto che
    /// una lista di task effettivi da eseguire.
    pub fn schedule_tick(&mut self, tick_number: u64) -> ScheduledWork {
        self.ticks_scheduled = self.ticks_scheduled.wrapping_add(1);

        // Per ora: versione semplificata che costruisce un piano base
        // In futuro: potrebbe guardare code effettive, storia recente, ecc.

        let mut work = ScheduledWork {
            tasks: Vec::new(),
            budget: self.config.max_budget_per_tick,
            background_active: false,
        };

        // Critical lane: sempre attiva
        // (Il NeuroNode decide se c'è effettivamente input utente da processare)
        work.tasks.push(TaskKind::UserInference);
        work.tasks.push(TaskKind::PolicyEvaluation);
        work.tasks.push(TaskKind::UserDelivery);

        // Normal lane: training e delta (modulato da throttle nel tick reale)
        if tick_number % 10 == 0 {
            // Training ogni 10 tick
            work.tasks.push(TaskKind::LocalTraining);
        }
        if tick_number % 100 == 0 {
            // Delta computation ogni 100 tick
            work.tasks.push(TaskKind::DeltaComputation);
            work.tasks.push(TaskKind::DeltaSubmission);
        }

        // Background lane: snapshot, meta, update
        if tick_number % 10_000 == 0 {
            work.tasks.push(TaskKind::SnapshotCreation);
            work.background_active = true;
        }
        if tick_number % 1_000 == 0 {
            work.tasks.push(TaskKind::MetricsSampling);
            work.background_active = true;
        }
        if tick_number % 50_000 == 0 {
            work.tasks.push(TaskKind::UpdateCheck);
            work.background_active = true;
        }

        work
    }

    /// Enqueue un task in una delle code.
    ///
    /// Questo metodo è disponibile per uso futuro, quando lo scheduler
    /// gestirà code di task effettive invece di essere stateless.
    pub fn enqueue(&mut self, task: TaskKind) {
        match task.lane() {
            Lane::Critical => self.critical_queue.push_back(task),
            Lane::Normal => self.normal_queue.push_back(task),
            Lane::Background => self.background_queue.push_back(task),
        }
    }

    /// Rimuove e restituisce il prossimo task dalla coda Critical.
    pub fn dequeue_critical(&mut self) -> Option<TaskKind> {
        let task = self.critical_queue.pop_front();
        if task.is_some() {
            self.tasks_executed = self.tasks_executed.wrapping_add(1);
        }
        task
    }

    /// Rimuove e restituisce il prossimo task dalla coda Normal.
    pub fn dequeue_normal(&mut self) -> Option<TaskKind> {
        let task = self.normal_queue.pop_front();
        if task.is_some() {
            self.tasks_executed = self.tasks_executed.wrapping_add(1);
        }
        task
    }

    /// Rimuove e restituisce il prossimo task dalla coda Background.
    pub fn dequeue_background(&mut self) -> Option<TaskKind> {
        let task = self.background_queue.pop_front();
        if task.is_some() {
            self.tasks_executed = self.tasks_executed.wrapping_add(1);
        }
        task
    }

    /// Restituisce il numero di task pendenti in tutte le code.
    #[must_use]
    pub fn pending_tasks(&self) -> usize {
        self.critical_queue.len() + self.normal_queue.len() + self.background_queue.len()
    }

    /// Restituisce il numero di tick schedulati finora.
    #[must_use]
    pub const fn ticks_scheduled(&self) -> u64 {
        self.ticks_scheduled
    }

    /// Restituisce il numero totale di task eseguiti finora.
    #[must_use]
    pub const fn tasks_executed(&self) -> u64 {
        self.tasks_executed
    }

    /// Svuota tutte le code (utile per reset o test).
    pub fn clear_all(&mut self) {
        self.critical_queue.clear();
        self.normal_queue.clear();
        self.background_queue.clear();
    }
}

impl Default for PriorityScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_weights_are_correct() {
        assert_eq!(Lane::Critical.weight(), 10);
        assert_eq!(Lane::Normal.weight(), 5);
        assert_eq!(Lane::Background.weight(), 1);
    }

    #[test]
    fn only_background_lane_is_background() {
        assert!(!Lane::Critical.is_background());
        assert!(!Lane::Normal.is_background());
        assert!(Lane::Background.is_background());
    }

    #[test]
    fn task_kind_lane_mapping() {
        assert_eq!(TaskKind::UserInference.lane(), Lane::Critical);
        assert_eq!(TaskKind::PolicyEvaluation.lane(), Lane::Critical);
        assert_eq!(TaskKind::LocalTraining.lane(), Lane::Normal);
        assert_eq!(TaskKind::SnapshotCreation.lane(), Lane::Background);
    }

    #[test]
    fn task_costs_are_reasonable() {
        // Critical tasks dovrebbero essere leggeri
        assert!(TaskKind::UserDelivery.cost() < 0.1);
        assert!(TaskKind::PolicyEvaluation.cost() < 0.1);

        // Training è pesante
        assert!(TaskKind::LocalTraining.cost() > 0.5);

        // Snapshot è moderatamente pesante
        assert!(TaskKind::SnapshotCreation.cost() > 0.3);
        assert!(TaskKind::SnapshotCreation.cost() < 0.6);
    }

    #[test]
    fn scheduled_work_empty_has_no_work() {
        let work = ScheduledWork::empty();
        assert!(!work.has_work());
        assert_eq!(work.task_count(), 0);
        assert_eq!(work.total_cost(), 0.0);
    }

    #[test]
    fn scheduled_work_total_cost() {
        let work = ScheduledWork {
            tasks: vec![
                TaskKind::UserInference,     // 0.3
                TaskKind::PolicyEvaluation,  // 0.05
                TaskKind::LocalTraining,     // 0.8
            ],
            budget: 1.0,
            background_active: false,
        };

        assert_eq!(work.task_count(), 3);
        assert!((work.total_cost() - 1.15).abs() < 0.01);
    }

    #[test]
    fn new_scheduler_has_empty_queues() {
        let sched = PriorityScheduler::new();
        assert_eq!(sched.pending_tasks(), 0);
        assert_eq!(sched.ticks_scheduled(), 0);
        assert_eq!(sched.tasks_executed(), 0);
    }

    #[test]
    fn schedule_tick_returns_valid_work() {
        let mut sched = PriorityScheduler::new();
        let work = sched.schedule_tick(0);

        assert!(work.has_work());
        assert!(work.budget > 0.0);
        assert!(work.budget <= 1.0);
    }

    #[test]
    fn schedule_tick_includes_critical_tasks() {
        let mut sched = PriorityScheduler::new();
        let work = sched.schedule_tick(0);

        // Critical lane dovrebbe essere sempre presente
        let has_critical = work
            .tasks
            .iter()
            .any(|t| t.lane() == Lane::Critical);
        assert!(has_critical);
    }

    #[test]
    fn schedule_tick_background_is_periodic() {
        let mut sched = PriorityScheduler::new();

        // Tick normale: no background
        let work1 = sched.schedule_tick(5);
        assert!(!work1.background_active);

        // Tick 10_000: snapshot (background)
        let work2 = sched.schedule_tick(10_000);
        assert!(work2.background_active);
        assert!(work2.tasks.contains(&TaskKind::SnapshotCreation));
    }

    #[test]
    fn enqueue_and_dequeue_critical() {
        let mut sched = PriorityScheduler::new();

        sched.enqueue(TaskKind::UserInference);
        sched.enqueue(TaskKind::PolicyEvaluation);

        assert_eq!(sched.pending_tasks(), 2);

        let task1 = sched.dequeue_critical();
        assert_eq!(task1, Some(TaskKind::UserInference));

        let task2 = sched.dequeue_critical();
        assert_eq!(task2, Some(TaskKind::PolicyEvaluation));

        assert_eq!(sched.pending_tasks(), 0);
        assert_eq!(sched.tasks_executed(), 2);
    }

    #[test]
    fn enqueue_and_dequeue_normal() {
        let mut sched = PriorityScheduler::new();

        sched.enqueue(TaskKind::LocalTraining);
        assert_eq!(sched.pending_tasks(), 1);

        let task = sched.dequeue_normal();
        assert_eq!(task, Some(TaskKind::LocalTraining));
        assert_eq!(sched.pending_tasks(), 0);
    }

    #[test]
    fn enqueue_and_dequeue_background() {
        let mut sched = PriorityScheduler::new();

        sched.enqueue(TaskKind::SnapshotCreation);
        assert_eq!(sched.pending_tasks(), 1);

        let task = sched.dequeue_background();
        assert_eq!(task, Some(TaskKind::SnapshotCreation));
        assert_eq!(sched.pending_tasks(), 0);
    }

    #[test]
    fn clear_all_empties_queues() {
        let mut sched = PriorityScheduler::new();

        sched.enqueue(TaskKind::UserInference);
        sched.enqueue(TaskKind::LocalTraining);
        sched.enqueue(TaskKind::SnapshotCreation);

        assert_eq!(sched.pending_tasks(), 3);

        sched.clear_all();
        assert_eq!(sched.pending_tasks(), 0);
    }

    #[test]
    fn dequeue_from_empty_queue_returns_none() {
        let mut sched = PriorityScheduler::new();

        assert_eq!(sched.dequeue_critical(), None);
        assert_eq!(sched.dequeue_normal(), None);
        assert_eq!(sched.dequeue_background(), None);
    }
}
