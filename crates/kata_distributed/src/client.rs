//! Distributed training client placeholder.
//!
//! Corresponds to `KataGo/cpp/distributed/client.h` and `client.cpp`.
//! The C++ client handles HTTP(S) communication with the KataGo distributed
//! training server: fetching tasks, downloading models, uploading games. This
//! Rust crate is a skeleton exposing the expected public API.

use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use kata_core::global::StringError;
use kata_core::logger::Logger;
use kata_data::sgf::PositionSample;
use kata_data::training::FinishedGameData;
use kata_game::board::{P_BLACK, P_WHITE, Player};
use rand::Rng;
use reqwest::{Certificate, Proxy};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::task::{ModelInfo, RunParameters, Task};

/// A parsed server or proxy URL.
#[derive(Debug, Clone, Default)]
pub struct Url {
    pub original_string: String,
    pub is_ssl: bool,
    pub host: String,
    pub port: i32,
    pub path: String,
    pub username: String,
    pub password: String,
}

impl Url {
    /// Maximum allowed URL length, matching `client.cpp`.
    const MAX_LEN: usize = 4096;

    /// Parse a URL string.
    ///
    /// Mirrors `Url::parse` in `cpp/distributed/client.cpp`. Supports
    /// `http://` and `https://` schemes, optional `user:pass@` basic auth,
    /// optional explicit ports, and preserves the path.
    pub fn parse(s: &str, check_for_user_pass: bool) -> Result<Self, StringError> {
        if s.len() > Self::MAX_LEN {
            return Err(StringError::new(format!("Invalid URL, too long: {s}")));
        }

        let original_string = s.to_string();
        let mut url = s;

        let is_ssl;
        let default_port;
        if let Some(rest) = url.strip_prefix("http://") {
            is_ssl = false;
            default_port = 80;
            url = rest;
        } else if let Some(rest) = url.strip_prefix("https://") {
            is_ssl = true;
            default_port = 443;
            url = rest;
        } else {
            return Err(StringError::new(format!(
                "Url must start with 'http://' or 'https://', got: {s}"
            )));
        }

        let (host_and_port, path_part) = match url.find('/') {
            Some(idx) => (&url[..idx], &url[idx..]),
            None => (url, "/"),
        };
        let mut host_and_port = host_and_port.to_string();
        let path = if path_part.is_empty() {
            "/".to_string()
        } else {
            path_part.to_string()
        };

        let mut username = String::new();
        let mut password = String::new();
        if check_for_user_pass {
            if let Some(at_idx) = host_and_port.rfind('@') {
                let user_pass = host_and_port[..at_idx].to_string();
                host_and_port = host_and_port[at_idx + 1..].to_string();

                if let Some(colon_idx) = user_pass.find(':') {
                    username = user_pass[..colon_idx].to_string();
                    password = user_pass[colon_idx + 1..].to_string();
                } else {
                    username = user_pass;
                }
            }
        }

        let (host, port) = match host_and_port.rfind(':') {
            Some(colon_idx) => {
                let host_part = &host_and_port[..colon_idx];
                let port_str = &host_and_port[colon_idx + 1..];
                let port: i32 = port_str.parse().map_err(|_| {
                    StringError::new(format!("Could not parse port in url as int: {port_str}"))
                })?;
                if port < 0 {
                    return Err(StringError::new(format!(
                        "Url port was negative: {port_str}"
                    )));
                }
                (host_part.to_string(), port)
            }
            None => (host_and_port, default_port),
        };

        Ok(Self {
            original_string,
            is_ssl,
            host,
            port,
            path,
            username,
            password,
        })
    }

    /// Replace the path component of the URL and update `original_string`.
    ///
    /// Mirrors `Url::replacePath` in `cpp/distributed/client.cpp`.
    pub fn replace_path(&mut self, new_path: &str) {
        let mut s = String::new();
        if self.is_ssl {
            s.push_str("https://");
        } else {
            s.push_str("http://");
        }
        if !self.username.is_empty() {
            s.push_str(&self.username);
            if !self.password.is_empty() {
                s.push(':');
                s.push_str(&self.password);
            }
            s.push('@');
        }
        s.push_str(&self.host);
        if (self.is_ssl && self.port != 443) || (!self.is_ssl && self.port != 80) {
            s.push(':');
            s.push_str(&self.port.to_string());
        }
        s.push_str(new_path);
        self.original_string = s;
        self.path = new_path.to_string();
    }
}

/// Parse a model description out of a JSON object.
///
/// Mirrors the local `parseModelInfo` helper in `cpp/distributed/client.cpp`.
fn parse_model_info(network_properties: &Value) -> Result<ModelInfo, StringError> {
    Ok(ModelInfo {
        name: get_string(network_properties, "name")?,
        info_url: get_string_or_empty(network_properties, "url"),
        download_url: get_string_or_empty(network_properties, "model_file"),
        bytes: get_usize(network_properties, "model_file_bytes")?,
        sha256: get_string(network_properties, "model_file_sha256")?,
        is_random: get_bool(network_properties, "is_random")?,
    })
}

/// Parse a task response from the distributed server.
///
/// Mirrors `Connection::parseTask` in `cpp/distributed/client.cpp`.
pub fn parse_task(task: &mut Task, response: &Value) -> Result<(), StringError> {
    let start_poses_list = parse_start_poses(response)?;
    let overrides_list = parse_overrides(response)?;

    let kind = get_string(response, "kind")?;
    if kind == "selfplay" {
        let network_properties = get_object(response, "network")?;
        let run_properties = get_object(response, "run")?;

        task.task_id.clear();
        task.task_group = get_string(network_properties, "name")?;
        task.run_name = get_string(run_properties, "name")?;
        task.run_info_url = get_string_or_empty(run_properties, "url");
        task.config = get_string(response, "config")?;
        task.model_black = parse_model_info(network_properties)?;
        task.model_white = task.model_black.clone();
        task.start_poses = start_poses_list;
        task.overrides = overrides_list;
        task.do_write_training_data = true;
        task.is_rating_game = false;
    } else if kind == "rating" {
        let black_network = get_object(response, "black_network")?;
        let white_network = get_object(response, "white_network")?;
        let run_properties = get_object(response, "run")?;

        let black_created_at = get_string(black_network, "created_at")?;
        let white_created_at = get_string(white_network, "created_at")?;
        let most_recent_name = if black_created_at < white_created_at {
            get_string(white_network, "name")?
        } else {
            get_string(black_network, "name")?
        };

        task.task_id.clear();
        task.task_group = format!("rating_{most_recent_name}");
        task.run_name = get_string(run_properties, "name")?;
        task.run_info_url = get_string_or_empty(run_properties, "url");
        task.config = get_string(response, "config")?;
        task.model_black = parse_model_info(black_network)?;
        task.model_white = parse_model_info(white_network)?;
        task.start_poses = start_poses_list;
        task.overrides = overrides_list;
        task.do_write_training_data = false;
        task.is_rating_game = true;
    } else {
        return Err(StringError::new(format!(
            "kind was neither 'selfplay' or 'rating' in json response: {}",
            serde_json::to_string(response).unwrap_or_default()
        )));
    }

    Ok(())
}

fn parse_start_poses(response: &Value) -> Result<Vec<PositionSample>, StringError> {
    let mut result = Vec::new();
    if let Some(start_poses) = response.get("start_poses") {
        let arr = start_poses.as_array().ok_or_else(|| {
            StringError::new(format!(
                "start_poses was not array in response: {}",
                serde_json::to_string(response).unwrap_or_default()
            ))
        })?;
        for elt in arr {
            let line = serde_json::to_string(elt).map_err(|e| {
                StringError::new(format!("Could not serialize start_poses element: {e}"))
            })?;
            let sample = PositionSample::of_json_line(&line).map_err(|e| {
                StringError::new(format!("Could not parse start_poses element: {}", e.0))
            })?;
            result.push(sample);
        }
    }
    Ok(result)
}

fn parse_overrides(response: &Value) -> Result<Vec<String>, StringError> {
    let mut result = Vec::new();
    if let Some(overrides) = response.get("overrides") {
        let arr = overrides.as_array().ok_or_else(|| {
            StringError::new(format!(
                "overrides was not array in response: {}",
                serde_json::to_string(response).unwrap_or_default()
            ))
        })?;
        for elt in arr {
            let s = elt.as_str().ok_or_else(|| {
                StringError::new("overrides element was not a string".to_string())
            })?;
            result.push(s.to_string());
        }
    }
    Ok(result)
}

fn get_object<'a>(value: &'a Value, key: &str) -> Result<&'a Value, StringError> {
    value
        .get(key)
        .ok_or_else(|| StringError::new(format!("Missing key in json: {key}")))
}

fn get_string(value: &Value, key: &str) -> Result<String, StringError> {
    get_object(value, key)?
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| StringError::new(format!("Expected string for key: {key}")))
}

fn get_string_or_empty(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn get_bool(value: &Value, key: &str) -> Result<bool, StringError> {
    get_object(value, key)?
        .as_bool()
        .ok_or_else(|| StringError::new(format!("Expected bool for key: {key}")))
}

fn get_usize(value: &Value, key: &str) -> Result<usize, StringError> {
    get_object(value, key)?
        .as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| StringError::new(format!("Expected non-negative integer for key: {key}")))
}

fn get_i32_in_range(value: &Value, key: &str, min: i32, max: i32) -> Result<i32, StringError> {
    let n = get_object(value, key)?
        .as_i64()
        .ok_or_else(|| StringError::new(format!("Expected integer for key: {key}")))?;
    if n < i64::from(min) || n > i64::from(max) {
        return Err(StringError::new(format!(
            "Value for {key} out of range [{min}, {max}]: {n}"
        )));
    }
    Ok(n as i32)
}

/// Parse the run parameters returned by the server.
///
/// Mirrors `Connection::getRunParameters` in `cpp/distributed/client.cpp`.
pub fn parse_run_parameters(run: &Value) -> Result<RunParameters, StringError> {
    Ok(RunParameters {
        run_name: get_string(run, "name")?,
        info_url: get_string_or_empty(run, "url"),
        data_board_len: get_i32_in_range(run, "data_board_len", 3, 19)?,
        inputs_version: get_i32_in_range(run, "inputs_version", 3, 10)?,
        max_search_threads_allowed: get_i32_in_range(run, "max_search_threads_allowed", 1, 16384)?,
    })
}

/// A single multipart form field: name, plain-text value, optional file part.
type UploadField = (&'static str, String, Option<(String, String, Vec<u8>)>);

/// Build the multipart form fields for a game upload.
fn build_upload_fields(
    task: &Task,
    game_data: &FinishedGameData,
    pos_sample: Option<&PositionSample>,
    sgf_contents: &str,
    npz_contents: Option<&[u8]>,
    num_data_rows: i64,
) -> Result<Vec<UploadField>, ClientError> {
    let mut extra_metadata = serde_json::json!({
        "playout_doubling_advantage": game_data.playout_doubling_advantage,
        "playout_doubling_advantage_pla": kata_game::board::player_io::player_to_string(game_data.playout_doubling_advantage_pla),
        "draw_equivalent_wins_for_white": game_data.draw_equivalent_wins_for_white,
    });
    if let Some(sample) = pos_sample {
        if !sample.metadata.is_empty() {
            extra_metadata["pos_metadata"] = serde_json::Value::String(sample.metadata.clone());
        }
    }

    let game_uid = format!("{}", game_data.game_hash);
    let winner = winner_string(game_data.end_hist.winner, game_data.end_hist.is_no_result);
    let gametype = game_type_string(game_data.mode);
    let rules = game_data
        .start_hist
        .rules
        .to_json_string_no_komi_maybe_omit_stuff();

    let mut fields: Vec<UploadField> = vec![
        (
            "board_size_x",
            game_data.start_board.x_size.to_string(),
            None,
        ),
        (
            "board_size_y",
            game_data.start_board.y_size.to_string(),
            None,
        ),
        ("handicap", game_data.handicap_for_sgf.to_string(), None),
        ("komi", game_data.start_hist.rules.komi.to_string(), None),
        ("gametype", gametype.to_string(), None),
        ("rules", rules, None),
        ("extra_metadata", extra_metadata.to_string(), None),
        ("winner", winner.to_string(), None),
        (
            "score",
            game_data.end_hist.final_white_minus_black_score.to_string(),
            None,
        ),
        (
            "resigned",
            (if game_data.end_hist.is_resignation {
                "true"
            } else {
                "false"
            })
            .to_string(),
            None,
        ),
        (
            "game_length",
            game_data.end_hist.move_history.len().to_string(),
            None,
        ),
        ("kg_game_uid", game_uid.clone(), None),
        ("run", task.run_info_url.clone(), None),
        ("white_network", task.model_white.info_url.clone(), None),
        ("black_network", task.model_black.info_url.clone(), None),
        (
            "sgf_file",
            sgf_contents.to_string(),
            Some((
                format!("{game_uid}.sgf"),
                "text/plain".to_string(),
                sgf_contents.as_bytes().to_vec(),
            )),
        ),
    ];

    if let Some(npz) = npz_contents {
        fields.push((
            "training_data_file",
            String::new(),
            Some((
                format!("{game_uid}.npz"),
                "application/octet-stream".to_string(),
                npz.to_vec(),
            )),
        ));
        fields.push(("num_training_rows", num_data_rows.to_string(), None));
    }

    Ok(fields)
}

fn game_type_string(mode: i32) -> &'static str {
    match mode {
        0 => "normal",
        1 => "cleanup_training",
        2 => "fork",
        3 => "handicap",
        4 => "sgfpos",
        5 => "hintpos",
        6 => "hintfork",
        7 => "asym",
        _ => "unknown",
    }
}

fn winner_string(winner: Player, is_no_result: bool) -> &'static str {
    match winner {
        P_WHITE => "W",
        P_BLACK => "B",
        _ if is_no_result => "-",
        _ => "0",
    }
}

/// Errors that can occur in distributed client operations.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// URL parsing failed.
    #[error("URL error: {0}")]
    Url(#[from] StringError),
    /// HTTP client setup or request failed.
    #[error("HTTP error: {0}")]
    Http(String),
    /// Server returned an unexpected status code.
    #[error("Server returned status {status}: {body}")]
    Server { status: u16, body: String },
    /// Distributed training support is not yet implemented for this operation.
    #[error("Distributed training client is not yet implemented")]
    NotImplemented,
}

/// Connection to a distributed training server.
#[allow(dead_code)]
pub struct Connection {
    server_url: String,
    username: String,
    password: String,
    ca_certs_file: String,
    proxy_url: Url,
    model_download_mirror_base_url: String,
    mirror_use_proxy: bool,
    client_instance_id: String,
    logger: Arc<Logger>,
}

impl Connection {
    /// Create a new connection to the distributed server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        server_url: &str,
        username: &str,
        password: &str,
        ca_certs_file: &str,
        proxy_url: &Url,
        model_download_mirror_base_url: &str,
        mirror_use_proxy: bool,
        logger: Arc<Logger>,
    ) -> Self {
        let client_instance_id = format!(
            "{:016x}{:016x}",
            rand::random::<u64>(),
            rand::random::<u64>()
        );
        Self {
            server_url: server_url.to_string(),
            username: username.to_string(),
            password: password.to_string(),
            ca_certs_file: ca_certs_file.to_string(),
            proxy_url: proxy_url.clone(),
            model_download_mirror_base_url: model_download_mirror_base_url.to_string(),
            mirror_use_proxy,
            client_instance_id,
            logger,
        }
    }

    /// Test the connection to the server.
    pub fn test_connection(&self) -> Result<(), ClientError> {
        let response = self.http_get("/")?;
        let status = response.status();
        if status != 200 {
            let body = response.text().unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(())
    }

    /// Build an HTTP client configured for this connection.
    fn build_client(&self) -> Result<reqwest::blocking::Client, ClientError> {
        let mut builder = reqwest::blocking::Client::builder().timeout(Duration::from_secs(20));

        if !self.ca_certs_file.is_empty() {
            let pem = std::fs::read_to_string(&self.ca_certs_file).map_err(|e| {
                ClientError::Http(format!(
                    "Could not read CA certs file {}: {e}",
                    self.ca_certs_file
                ))
            })?;
            let cert = Certificate::from_pem(pem.as_bytes())
                .map_err(|e| ClientError::Http(format!("Could not parse CA certs file: {e}")))?;
            builder = builder.add_root_certificate(cert);
        }

        if !self.proxy_url.host.is_empty() {
            let proxy_str = format!(
                "{}://{}:{}",
                if self.proxy_url.is_ssl {
                    "https"
                } else {
                    "http"
                },
                self.proxy_url.host,
                self.proxy_url.port
            );
            let proxy = if self.proxy_url.is_ssl {
                Proxy::https(&proxy_str)
            } else {
                Proxy::http(&proxy_str)
            }
            .map_err(|e| ClientError::Http(format!("Could not configure proxy: {e}")))?;
            let proxy = if !self.proxy_url.username.is_empty() {
                proxy.basic_auth(&self.proxy_url.username, &self.proxy_url.password)
            } else {
                proxy
            };
            builder = builder.proxy(proxy);
        }

        builder
            .build()
            .map_err(|e| ClientError::Http(format!("Could not build HTTP client: {e}")))
    }

    /// Perform a GET request against the server.
    fn http_get(&self, path: &str) -> Result<reqwest::blocking::Response, ClientError> {
        let url = Url::parse(&self.server_url, false)?;
        let scheme = if url.is_ssl { "https" } else { "http" };
        let full_url = format!("{}://{}:{}{}", scheme, url.host, url.port, path);

        let client = self.build_client()?;
        let response = client
            .get(&full_url)
            .basic_auth(&self.username, Some(&self.password))
            .send()
            .map_err(|e| {
                ClientError::Http(format!(
                    "Could not connect to server at {}: {e}",
                    self.server_url
                ))
            })?;
        Ok(response)
    }

    /// Perform a multipart POST request against the server.
    fn http_post_multi(
        &self,
        path: &str,
        fields: &[(&str, String)],
    ) -> Result<reqwest::blocking::Response, ClientError> {
        let url = Url::parse(&self.server_url, false)?;
        let scheme = if url.is_ssl { "https" } else { "http" };
        let full_url = format!("{}://{}:{}{}", scheme, url.host, url.port, path);

        let mut form = reqwest::blocking::multipart::Form::new();
        for (name, value) in fields {
            form = form.text(name.to_string(), value.clone());
        }

        let client = self.build_client()?;
        let response = client
            .post(&full_url)
            .basic_auth(&self.username, Some(&self.password))
            .multipart(form)
            .send()
            .map_err(|e| {
                ClientError::Http(format!(
                    "Could not connect to server at {}: {e}",
                    self.server_url
                ))
            })?;
        Ok(response)
    }

    /// Retry a server operation with exponential backoff.
    ///
    /// Mirrors `Connection::retryLoop` in `cpp/distributed/client.cpp` with a
    /// simplified single failure mode. Returns `Ok(false)` if `should_stop`
    /// returns true before success, and `Ok(true)` on success.
    fn retry_loop(
        &self,
        error_label: &str,
        max_tries: i32,
        should_stop: &dyn Fn() -> bool,
        mut f: impl FnMut() -> Result<(), ClientError>,
    ) -> Result<bool, ClientError> {
        if should_stop() {
            return Ok(false);
        }

        let mut failure_interval = 5.0;
        for i in 0..max_tries {
            if should_stop() {
                return Ok(false);
            }
            match f() {
                Ok(()) => {
                    if i > 0 {
                        self.logger
                            .write(&format!("{error_label}: Connection to server is back!"));
                    }
                    return Ok(true);
                }
                Err(e) => {
                    if should_stop() {
                        return Ok(false);
                    }
                    if i >= max_tries - 1 {
                        return Err(e);
                    }
                    self.logger.write(&format!(
                        "{error_label}: Error connecting to server, possibly an internet blip, or possibly the server is down or temporarily misconfigured, waiting about {failure_interval:.0} seconds and trying again."
                    ));
                    self.logger.write(&format!("Error was:\n{e}"));

                    let jitter: f64 = rand::thread_rng().gen_range(0.95..1.05);
                    let mut interval_remaining = failure_interval * jitter;
                    while interval_remaining > 0.0 {
                        if should_stop() {
                            return Ok(false);
                        }
                        let sleep_time = interval_remaining.min(2.0);
                        std::thread::sleep(Duration::from_secs_f64(sleep_time));
                        interval_remaining -= 2.0;
                    }
                    failure_interval = (failure_interval * 1.3 + 1.0).round();
                    if failure_interval > 7200.0 {
                        failure_interval = 7200.0;
                    }
                }
            }
        }
        Ok(true)
    }

    /// Get parameters for the current run.
    pub fn get_run_parameters(&self) -> Result<RunParameters, ClientError> {
        let response = self.http_get("/api/runs/current_for_client/")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        let json: Value = response
            .json()
            .map_err(|e| ClientError::Http(format!("Could not parse run parameters JSON: {e}")))?;
        parse_run_parameters(&json).map_err(ClientError::from)
    }

    /// Get the next task from the server.
    #[allow(clippy::too_many_arguments)]
    pub fn get_next_task(
        &self,
        task: &mut Task,
        _base_dir: &str,
        retry_on_failure: bool,
        allow_selfplay_task: bool,
        allow_rating_task: bool,
        task_rep_factor: i32,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<bool, ClientError> {
        const DEFAULT_MAX_TRIES: i32 = 100;
        let max_tries = if retry_on_failure {
            DEFAULT_MAX_TRIES
        } else {
            1
        };

        let git_revision = format!("{}-rust", env!("CARGO_PKG_VERSION"));
        let client_instance_id = self.client_instance_id.clone();

        self.retry_loop("getNextTask", max_tries, should_stop, || {
            loop {
                let fields = vec![
                    ("git_revision", git_revision.clone()),
                    ("client_instance_id", client_instance_id.clone()),
                    ("task_rep_factor", task_rep_factor.to_string()),
                    (
                        "allow_selfplay_task",
                        (if allow_selfplay_task { "true" } else { "false" }).to_string(),
                    ),
                    (
                        "allow_rating_task",
                        (if allow_rating_task { "true" } else { "false" }).to_string(),
                    ),
                ];
                let response = self.http_post_multi("/api/tasks/", &fields)?;
                let status = response.status();
                let body = response.text().unwrap_or_default();

                if !allow_rating_task
                    && status == 400
                    && body.contains("server is only serving rating games right now")
                {
                    self.logger.write(
                        "Server is only serving rating games right now but we're full on how many we can accept, so we will sleep a while and then retry."
                    );
                    std::thread::sleep(Duration::from_secs(30));
                    return Err(ClientError::Http(
                        "Contacted server but rating games were full".to_string(),
                    ));
                }

                if !status.is_success() {
                    return Err(ClientError::Server {
                        status: status.as_u16(),
                        body,
                    });
                }

                let json: Value = serde_json::from_str(&body).map_err(|e| {
                    ClientError::Http(format!("Could not parse task JSON: {e}"))
                })?;

                let kind = json
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if kind == "rating" && !allow_rating_task {
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }

                parse_task(task, &json).map_err(ClientError::from)?;
                break;
            }
            Ok(())
        })
    }

    /// Upload a training game and its training data to the server.
    ///
    /// Mirrors `Connection::uploadTrainingGameAndData` in
    /// `cpp/distributed/client.cpp`.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_training_game_and_data(
        &self,
        task: &Task,
        game_data: &FinishedGameData,
        pos_sample: Option<&PositionSample>,
        sgf_file_path: &str,
        npz_file_path: &str,
        num_data_rows: i64,
        retry_on_failure: bool,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<bool, ClientError> {
        const DEFAULT_MAX_TRIES: i32 = 100;
        let max_tries = if retry_on_failure {
            DEFAULT_MAX_TRIES
        } else {
            1
        };

        let sgf_contents = std::fs::read_to_string(sgf_file_path).map_err(|e| {
            ClientError::Http(format!("Could not read sgf file {sgf_file_path}: {e}"))
        })?;
        let npz_contents = std::fs::read(npz_file_path).map_err(|e| {
            ClientError::Http(format!("Could not read npz file {npz_file_path}: {e}"))
        })?;

        self.retry_loop("uploadTrainingGameAndData", max_tries, should_stop, || {
            let fields = build_upload_fields(
                task,
                game_data,
                pos_sample,
                &sgf_contents,
                Some(&npz_contents),
                num_data_rows,
            )?;
            self.do_upload("/api/games/training/", sgf_file_path, &fields)?;
            Ok(())
        })
    }

    /// Upload a rating game to the server.
    ///
    /// Mirrors `Connection::uploadRatingGame` in `cpp/distributed/client.cpp`.
    pub fn upload_rating_game(
        &self,
        task: &Task,
        game_data: &FinishedGameData,
        sgf_file_path: &str,
        retry_on_failure: bool,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<bool, ClientError> {
        const DEFAULT_MAX_TRIES: i32 = 100;
        let max_tries = if retry_on_failure {
            DEFAULT_MAX_TRIES
        } else {
            1
        };

        let sgf_contents = std::fs::read_to_string(sgf_file_path).map_err(|e| {
            ClientError::Http(format!("Could not read sgf file {sgf_file_path}: {e}"))
        })?;

        self.retry_loop("uploadRatingGame", max_tries, should_stop, || {
            let fields = build_upload_fields(task, game_data, None, &sgf_contents, None, 0)?;
            self.do_upload("/api/games/rating/", sgf_file_path, &fields)?;
            Ok(())
        })
    }

    /// Execute a multipart upload and handle server-specific skip conditions.
    fn do_upload(
        &self,
        path: &str,
        file_path: &str,
        fields: &[UploadField],
    ) -> Result<(), ClientError> {
        let mut form = reqwest::blocking::multipart::Form::new();
        for (name, value, file) in fields {
            match file {
                Some((filename, mime, data)) => {
                    let part = reqwest::blocking::multipart::Part::bytes(data.clone())
                        .file_name(filename.clone())
                        .mime_str(mime)
                        .map_err(|e| ClientError::Http(format!("Invalid mime type: {e}")))?;
                    form = form.part(name.to_string(), part);
                }
                None => {
                    form = form.text(name.to_string(), value.clone());
                }
            }
        }

        let url = Url::parse(&self.server_url, false)?;
        let scheme = if url.is_ssl { "https" } else { "http" };
        let full_url = format!("{}://{}:{}{}", scheme, url.host, url.port, path);

        let client = self.build_client()?;
        let response = client
            .post(&full_url)
            .basic_auth(&self.username, Some(&self.password))
            .multipart(form)
            .send()
            .map_err(|e| {
                ClientError::Http(format!(
                    "Could not connect to server at {}: {e}",
                    self.server_url
                ))
            })?;

        let status = response.status();
        let body = response.text().unwrap_or_default();

        if status == 400 && body.contains("already exist") {
            self.logger.write(&format!(
                "Server returned 400 with 'already exist', data is probably uploaded already or has a key conflict, so skipping, response was: {body}"
            ));
            return Ok(());
        }
        if status == 400 && body.contains("no longer enabled for") {
            self.logger.write(&format!(
                "Server returned 400 with 'no longer enabled for', probably we've moved on from this network, so skipping: {body}"
            ));
            return Ok(());
        }
        if status != 200 && status != 201 && status != 202 {
            return Err(ClientError::Server {
                status: status.as_u16(),
                body: format!(
                    "When uploading {file_path} server gave response that was not status code 200 OK or 201 Created or 202 Accepted\n{body}"
                ),
            });
        }
        Ok(())
    }

    /// Compute the local path for a model.
    ///
    /// Mirrors `Connection::getModelPath` in `cpp/distributed/client.cpp`.
    pub fn get_model_path(model_info: &ModelInfo, model_dir: &str) -> String {
        if model_info.is_random {
            return "/dev/null".to_string();
        }
        let dir = model_dir.trim_end_matches('/');
        let suffix = if model_info.download_url.ends_with(".txt.gz") {
            ".txt.gz"
        } else {
            ".bin.gz"
        };
        format!("{}/{}{}", dir, model_info.name, suffix)
    }

    /// Download a model if it is not already present locally.
    ///
    /// Mirrors `Connection::downloadModelIfNotPresent` in
    /// `cpp/distributed/client.cpp`. This simplified implementation does not
    /// resume partial downloads or coordinate across threads, but it does verify
    /// file size and SHA-256.
    pub fn download_model_if_not_present(
        &self,
        model_info: &ModelInfo,
        model_dir: &str,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<bool, ClientError> {
        if model_info.is_random || self.is_model_present(model_info, model_dir) {
            return Ok(true);
        }
        if should_stop() {
            return Ok(false);
        }

        let model_path = Self::get_model_path(model_info, model_dir);
        let tmp_path = format!("{}.tmp", model_path);

        let full_url = self.resolve_model_download_url(&model_info.download_url)?;
        self.logger.write(&format!(
            "Beginning download of model {} from {}",
            model_info.name, full_url
        ));

        let client = self.build_client()?;
        let mut response = client
            .get(&full_url)
            .send()
            .map_err(|e| ClientError::Http(format!("Could not download model: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }

        let mut file = std::fs::File::create(&tmp_path)
            .map_err(|e| ClientError::Http(format!("Could not create temp model file: {e}")))?;
        response
            .copy_to(&mut file)
            .map_err(|e| ClientError::Http(format!("Could not write model file: {e}")))?;
        drop(file);

        let size = std::fs::metadata(&tmp_path)
            .map(|m| m.len() as usize)
            .map_err(|e| ClientError::Http(format!("Could not stat model file: {e}")))?;
        if size != model_info.bytes {
            return Err(ClientError::Http(format!(
                "Downloaded model size mismatch: expected {} bytes, got {}",
                model_info.bytes, size
            )));
        }

        let sha256 = Self::compute_sha256(&tmp_path)?;
        if sha256 != model_info.sha256 {
            return Err(ClientError::Http(format!(
                "Downloaded model SHA-256 mismatch: expected {}, got {}",
                model_info.sha256, sha256
            )));
        }

        std::fs::rename(&tmp_path, &model_path)
            .map_err(|e| ClientError::Http(format!("Could not rename temp model file: {e}")))?;
        self.logger.write(&format!(
            "Done downloading {} bytes for model: {}",
            size, model_info.name
        ));
        Ok(true)
    }

    /// Check whether a model is already present locally.
    pub fn is_model_present(&self, model_info: &ModelInfo, model_dir: &str) -> bool {
        if model_info.is_random {
            return true;
        }
        let path = Self::get_model_path(model_info, model_dir);
        std::fs::metadata(&path)
            .map(|m| m.is_file())
            .unwrap_or(false)
    }

    /// Query the server for the newest model and download it if needed.
    pub fn maybe_download_newest_model(
        &self,
        model_dir: &str,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<bool, ClientError> {
        let response = self.http_get("/api/networks/newest_training/")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        let json: Value = response
            .json()
            .map_err(|e| ClientError::Http(format!("Could not parse newest model JSON: {e}")))?;
        let model_info = parse_model_info(&json).map_err(ClientError::from)?;
        self.download_model_if_not_present(&model_info, model_dir, should_stop)
    }

    /// Resolve the actual download URL, applying the mirror if configured.
    fn resolve_model_download_url(&self, download_url: &str) -> Result<String, ClientError> {
        if self.model_download_mirror_base_url.is_empty() {
            let url = Url::parse(download_url, false)?;
            let scheme = if url.is_ssl { "https" } else { "http" };
            Ok(format!(
                "{}://{}:{}{}",
                scheme, url.host, url.port, url.path
            ))
        } else {
            let url_from_server = Url::parse(download_url, false)?;
            let mut mirror = Url::parse(&self.model_download_mirror_base_url, false)?;
            mirror.replace_path(&url_from_server.path);
            let scheme = if mirror.is_ssl { "https" } else { "http" };
            Ok(format!(
                "{}://{}:{}{}",
                scheme, mirror.host, mirror.port, mirror.path
            ))
        }
    }

    /// Compute the SHA-256 hex digest of a file.
    fn compute_sha256(path: &str) -> Result<String, ClientError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| ClientError::Http(format!("Could not open model file: {e}")))?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| ClientError::Http(format!("Could not read model file: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex::encode(hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_reports_url_error_for_invalid_server() {
        let logger = std::sync::Arc::new(Logger::new(
            kata_core::logger::LoggerOptions::default(),
            None,
        ));
        let conn = Connection::new("", "", "", "", &Url::default(), "", false, logger);
        assert!(matches!(conn.test_connection(), Err(ClientError::Url(_))));
    }

    #[test]
    fn url_parse_https_defaults() {
        let url = Url::parse("https://example.com/foo/bar", false).unwrap();
        assert!(url.is_ssl);
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 443);
        assert_eq!(url.path, "/foo/bar");
        assert!(url.username.is_empty());
        assert!(url.password.is_empty());
        assert_eq!(url.original_string, "https://example.com/foo/bar");
    }

    #[test]
    fn url_parse_http_port_and_auth() {
        let url = Url::parse("http://user:pass@example.com:8080/path", true).unwrap();
        assert!(!url.is_ssl);
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 8080);
        assert_eq!(url.path, "/path");
        assert_eq!(url.username, "user");
        assert_eq!(url.password, "pass");
    }

    #[test]
    fn url_parse_no_path() {
        let url = Url::parse("https://example.com", false).unwrap();
        assert_eq!(url.path, "/");
    }

    #[test]
    fn url_parse_rejects_missing_scheme() {
        assert!(Url::parse("example.com", false).is_err());
    }

    #[test]
    fn url_replace_path_rebuilds_original_string() {
        let mut url = Url::parse("https://user:pass@example.com:8443/old", true).unwrap();
        url.replace_path("/new/path");
        assert_eq!(url.path, "/new/path");
        assert_eq!(
            url.original_string,
            "https://user:pass@example.com:8443/new/path"
        );
    }

    fn sample_start_pose() -> Value {
        serde_json::json!({
            "xSize": 7,
            "ySize": 7,
            "board": "......./......./......./......./......./......./.......",
            "nextPla": "B",
            "moveLocs": [],
            "movePlas": [],
            "initialTurnNumber": 0,
            "hintLoc": "pass",
            "weight": 1.0,
        })
    }

    fn test_connection() -> Connection {
        Connection::new(
            "http://127.0.0.1:1",
            "",
            "",
            "",
            &Url::default(),
            "",
            false,
            Arc::new(Logger::new(
                kata_core::logger::LoggerOptions::default(),
                None,
            )),
        )
    }

    #[test]
    fn parse_task_selfplay() {
        let response = serde_json::json!({
            "kind": "selfplay",
            "network": {
                "name": "net-v1",
                "url": "https://example.com/net-v1",
                "model_file": "https://example.com/net-v1.bin.gz",
                "model_file_bytes": 12345,
                "model_file_sha256": "abcd",
                "is_random": false,
            },
            "run": {
                "name": "run-a",
                "url": "https://example.com/run-a",
            },
            "config": "maxVisits = 100",
            "start_poses": [sample_start_pose()],
            "overrides": ["foo=bar"],
        });
        let mut task = Task::default();
        parse_task(&mut task, &response).unwrap();
        assert_eq!(task.task_group, "net-v1");
        assert_eq!(task.run_name, "run-a");
        assert_eq!(task.model_black.name, "net-v1");
        assert_eq!(task.model_white.name, "net-v1");
        assert_eq!(task.start_poses.len(), 1);
        assert_eq!(task.overrides, vec!["foo=bar"]);
        assert!(task.do_write_training_data);
        assert!(!task.is_rating_game);
    }

    #[test]
    fn parse_task_rating() {
        let response = serde_json::json!({
            "kind": "rating",
            "black_network": {
                "name": "net-b",
                "created_at": "2024-01-01T00:00:00Z",
                "model_file_sha256": "0000",
                "model_file_bytes": 1,
                "is_random": false,
            },
            "white_network": {
                "name": "net-w",
                "created_at": "2024-02-01T00:00:00Z",
                "model_file_sha256": "1111",
                "model_file_bytes": 1,
                "is_random": false,
            },
            "run": { "name": "run-a", "url": "" },
            "config": "maxVisits = 50",
        });
        let mut task = Task::default();
        parse_task(&mut task, &response).unwrap();
        assert_eq!(task.task_group, "rating_net-w");
        assert_eq!(task.model_black.name, "net-b");
        assert_eq!(task.model_white.name, "net-w");
        assert!(!task.do_write_training_data);
        assert!(task.is_rating_game);
    }

    #[test]
    fn parse_task_rejects_unknown_kind() {
        let response = serde_json::json!({ "kind": "unknown" });
        let mut task = Task::default();
        assert!(parse_task(&mut task, &response).is_err());
    }

    #[test]
    fn parse_run_parameters_ok() {
        let run = serde_json::json!({
            "name": "run-a",
            "url": "https://example.com/run-a",
            "data_board_len": 19,
            "inputs_version": 8,
            "max_search_threads_allowed": 64,
        });
        let params = parse_run_parameters(&run).unwrap();
        assert_eq!(params.run_name, "run-a");
        assert_eq!(params.info_url, "https://example.com/run-a");
        assert_eq!(params.data_board_len, 19);
        assert_eq!(params.inputs_version, 8);
        assert_eq!(params.max_search_threads_allowed, 64);
    }

    #[test]
    fn parse_run_parameters_rejects_out_of_range() {
        let run = serde_json::json!({
            "name": "run-a",
            "data_board_len": 2,
            "inputs_version": 8,
            "max_search_threads_allowed": 64,
        });
        assert!(parse_run_parameters(&run).is_err());
    }

    #[test]
    fn get_run_parameters_fails_when_server_unreachable() {
        let logger = Arc::new(Logger::new(
            kata_core::logger::LoggerOptions::default(),
            None,
        ));
        // Use a port that is extremely unlikely to be open.
        let conn = Connection::new(
            "http://127.0.0.1:1",
            "user",
            "pass",
            "",
            &Url::default(),
            "",
            false,
            logger,
        );
        assert!(matches!(
            conn.get_run_parameters(),
            Err(ClientError::Http(_))
        ));
    }

    #[test]
    fn get_model_path_joins_dir_and_name() {
        let model = ModelInfo {
            name: "net-v1".to_string(),
            download_url: "https://example.com/net-v1.bin.gz".to_string(),
            ..Default::default()
        };
        assert_eq!(
            Connection::get_model_path(&model, "/tmp/models"),
            "/tmp/models/net-v1.bin.gz"
        );
        assert_eq!(
            Connection::get_model_path(&model, "/tmp/models/"),
            "/tmp/models/net-v1.bin.gz"
        );
    }

    #[test]
    fn get_model_path_uses_txt_gz_suffix() {
        let model = ModelInfo {
            name: "net-v2".to_string(),
            download_url: "https://example.com/net-v2.txt.gz".to_string(),
            ..Default::default()
        };
        assert_eq!(
            Connection::get_model_path(&model, "/tmp/models"),
            "/tmp/models/net-v2.txt.gz"
        );
    }

    #[test]
    fn get_model_path_returns_dev_null_for_random() {
        let model = ModelInfo {
            name: "random".to_string(),
            is_random: true,
            ..Default::default()
        };
        assert_eq!(
            Connection::get_model_path(&model, "/tmp/models"),
            "/dev/null"
        );
    }

    #[test]
    fn is_model_present_true_for_random() {
        let conn = test_connection();
        let model = ModelInfo {
            is_random: true,
            ..Default::default()
        };
        assert!(conn.is_model_present(&model, "/nonexistent/dir"));
    }

    #[test]
    fn is_model_present_detects_existing_file() {
        let conn = test_connection();
        let dir = std::env::temp_dir().join(format!("katago-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = ModelInfo {
            name: "present".to_string(),
            download_url: "https://example.com/present.bin.gz".to_string(),
            ..Default::default()
        };
        let path = Connection::get_model_path(&model, dir.to_str().unwrap());
        std::fs::write(&path, b"data").unwrap();
        assert!(conn.is_model_present(&model, dir.to_str().unwrap()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn compute_sha256_matches_known_digest() {
        let dir = std::env::temp_dir().join(format!("katago-sha-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hello.txt");
        std::fs::write(&path, b"hello").unwrap();
        let digest = Connection::compute_sha256(path.to_str().unwrap()).unwrap();
        assert_eq!(
            digest,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn resolve_model_download_url_passes_through_without_mirror() {
        let conn = test_connection();
        let url = conn
            .resolve_model_download_url("https://example.com/models/net.bin")
            .unwrap();
        assert_eq!(url, "https://example.com:443/models/net.bin");
    }

    #[test]
    fn resolve_model_download_url_uses_mirror_path() {
        let conn = Connection::new(
            "http://127.0.0.1:1",
            "",
            "",
            "",
            &Url::default(),
            "https://mirror.example.com:9000/base/",
            false,
            Arc::new(Logger::new(
                kata_core::logger::LoggerOptions::default(),
                None,
            )),
        );
        let url = conn
            .resolve_model_download_url("https://server.example.com/models/net.bin")
            .unwrap();
        assert_eq!(url, "https://mirror.example.com:9000/models/net.bin");
    }

    #[test]
    fn get_next_task_fails_when_server_unreachable() {
        let conn = test_connection();
        let mut task = Task::default();
        assert!(matches!(
            conn.get_next_task(&mut task, "/tmp", false, true, true, 1, &|| false),
            Err(ClientError::Http(_))
        ));
    }

    #[test]
    fn upload_training_game_fails_when_files_missing() {
        let conn = test_connection();
        let task = Task::default();
        let game_data = FinishedGameData::default();
        assert!(matches!(
            conn.upload_training_game_and_data(
                &task,
                &game_data,
                None,
                "/nonexistent/game.sgf",
                "/nonexistent/data.npz",
                0,
                false,
                &|| false,
            ),
            Err(ClientError::Http(_))
        ));
    }

    #[test]
    fn upload_rating_game_fails_when_sgf_missing() {
        let conn = test_connection();
        let task = Task::default();
        let game_data = FinishedGameData::default();
        assert!(matches!(
            conn.upload_rating_game(&task, &game_data, "/nonexistent/game.sgf", false, &|| false),
            Err(ClientError::Http(_))
        ));
    }

    #[test]
    fn build_upload_fields_with_defaults() {
        let task = Task::default();
        let game_data = FinishedGameData::default();
        let fields = build_upload_fields(&task, &game_data, None, "(;)", Some(b"npz"), 1).unwrap();
        let names: Vec<_> = fields.iter().map(|(n, _, _)| *n).collect();
        assert!(names.contains(&"sgf_file"));
        assert!(names.contains(&"training_data_file"));
        assert!(names.contains(&"num_training_rows"));
    }
}
