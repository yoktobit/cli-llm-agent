use anyhow::Context;
use matrix_sdk::{
    Client,
    config::SyncSettings,
};

#[derive(Clone)]
pub struct MatrixAgentConfig {
    matrix_username: String,
    matrix_password: String,
    matrix_homeserver_url: String,
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

        Ok(Self {
            matrix_username: username,
            matrix_password: password,
            matrix_homeserver_url: homeserver_url,
        })
    }
}

impl MatrixAgent {
    pub async fn new(config: MatrixAgentConfig) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .homeserver_url(&config.matrix_homeserver_url)
            .build()
            .await?;
        Ok(Self {
            config: config,
            client: client,
        })
    }

    pub async fn connect_matrix(&self) -> Result<(), anyhow::Error> {
        self.client
            .matrix_auth()
            .login_username(&self.config.matrix_username, &self.config.matrix_password)
            .initial_device_display_name("autojoin bot")
            .await?;
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
}