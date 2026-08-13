//! Distributed training task types.
//!
//! Corresponds to `Client::Task` and related types in
//! `KataGo/cpp/distributed/client.h`.

use kata_data::sgf::PositionSample;

/// Description of a training or rating task returned by the distributed server.
#[derive(Debug, Clone, Default)]
pub struct Task {
    pub task_id: String,
    pub task_group: String,
    pub run_name: String,
    pub run_info_url: String,
    pub model_black: ModelInfo,
    pub model_white: ModelInfo,
    pub config: String,
    pub start_poses: Vec<PositionSample>,
    pub overrides: Vec<String>,
    pub do_write_training_data: bool,
    pub is_rating_game: bool,
}

/// Information about a model available from the distributed server.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub name: String,
    pub info_url: String,
    pub download_url: String,
    pub bytes: usize,
    pub sha256: String,
    pub is_random: bool,
}

/// Parameters describing the current distributed run.
#[derive(Debug, Clone, Default)]
pub struct RunParameters {
    pub run_name: String,
    pub info_url: String,
    pub data_board_len: i32,
    pub inputs_version: i32,
    pub max_search_threads_allowed: i32,
}
