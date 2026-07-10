use std::{fs, path::PathBuf};

use anyhow::Context;
use matrix_sdk::{
    Client,
    authentication::matrix::MatrixSession,
    config::SyncSettings,
    store::RoomLoadSettings,
};

#[derive(Clone)]
pub struct MatrixAgentConfig {
    matrix_username: String,
    matrix_password: String,
    matrix_homeserver_url: String,
    matrix_store_dir: PathBuf,
    matrix_session_file: PathBuf,
    matrix_device_display_name: String,
}

#[derive(Clone)]
pub struct MatrixAgent {
    config: MatrixAgentConfig,
    pub client: Client,
}

impl MatrixAgentConfig {
    pub fn default_from_env() -> Result<Self, anyhow::Error> {
        let homeserver_url = std::env::var("MATRIX_HOMESERVER_URL")
            .unwrap_or_else(|_| "https://matrix.org".to_string());
        let username = std::env::var("MATRIX_USERNAME")
            .context("No MATRIX_USERNAME given as env parameter")?;
        let password = std::env::var("MATRIX_PASSWORD")
            .context("No MATRIX_PASSWORD given as env parameter")?;
        let store_dir = std::env::var("MATRIX_STORE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".matrix-store"));
        let session_file = std::env::var("MATRIX_SESSION_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| store_dir.join("session.json"));
        let device_display_name = std::env::var("MATRIX_DEVICE_DISPLAY_NAME")
            .unwrap_or_else(|_| "autojoin bot".to_string());

        Ok(Self {
            matrix_username: username,
            matrix_password: password,
            matrix_homeserver_url: homeserver_url,
            matrix_store_dir: store_dir,
            matrix_session_file: session_file,
            matrix_device_display_name: device_display_name,
        })
    }
}

impl MatrixAgent {
    pub async fn new(config: MatrixAgentConfig) -> Result<Self, anyhow::Error> {
        if let Some(parent) = config.matrix_session_file.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create session dir {}", parent.display()))?;
        }
        fs::create_dir_all(&config.matrix_store_dir).with_context(|| {
            format!(
                "failed to create matrix store dir {}",
                config.matrix_store_dir.display()
            )
        })?;

        let client = Client::builder()
            .homeserver_url(&config.matrix_homeserver_url)
            .sqlite_store(&config.matrix_store_dir, None)
            .build()
            .await?;
        Ok(Self {
            config: config,
            client: client,
        })
    }

    pub async fn connect_matrix(&self) -> Result<(), anyhow::Error> {
        if self.try_restore_session().await? {
            self.log_active_session();
            return Ok(());
        }

        self.client
            .matrix_auth()
            .login_username(&self.config.matrix_username, &self.config.matrix_password)
            .initial_device_display_name(&self.config.matrix_device_display_name)
            .send()
            .await?;

        self.persist_session()?;
        self.log_active_session();
        Ok(())
    }

    pub async fn sync(&self) -> Result<(), anyhow::Error> {
        self.client
            .sync(SyncSettings::default())
            .await
            .context("error syncing")
    }

    pub fn client(&self) -> Client {
        self.client.clone()
    }

    async fn try_restore_session(&self) -> Result<bool, anyhow::Error> {
        let serialized = match fs::read(&self.config.matrix_session_file) {
            Ok(contents) => contents,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "failed to read matrix session file {}",
                        self.config.matrix_session_file.display()
                    )
                });
            }
        };

        let session: MatrixSession = match serde_json::from_slice(&serialized) {
            Ok(session) => session,
            Err(err) => {
                eprintln!(
                    "Could not parse matrix session file {}: {err}",
                    self.config.matrix_session_file.display()
                );
                return Ok(false);
            }
        };

        if let Err(err) = self
            .client
            .matrix_auth()
            .restore_session(session, RoomLoadSettings::default())
            .await
        {
            eprintln!(
                "Could not restore matrix session from {}: {err}",
                self.config.matrix_session_file.display()
            );
            return Ok(false);
        }

        println!(
            "Restored matrix session from {}",
            self.config.matrix_session_file.display()
        );
        Ok(true)
    }

    fn persist_session(&self) -> Result<(), anyhow::Error> {
        let session = self
            .client
            .matrix_auth()
            .session()
            .context("matrix login succeeded but no session is available to persist")?;
        let session_json =
            serde_json::to_vec_pretty(&session).context("failed to serialize matrix session")?;
        fs::write(&self.config.matrix_session_file, session_json).with_context(|| {
            format!(
                "failed to write matrix session file {}",
                self.config.matrix_session_file.display()
            )
        })?;

        println!(
            "Saved matrix session to {}",
            self.config.matrix_session_file.display()
        );
        Ok(())
    }

    fn log_active_session(&self) {
        if let Some(session) = self.client.matrix_auth().session() {
            println!(
                "Matrix session active for {} on device {}",
                session.meta.user_id, session.meta.device_id
            );
        }
    }
}
