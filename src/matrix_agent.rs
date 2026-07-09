use std::{sync::Arc, time::Duration};

use adk_rust::types::Content;
use adk_rust::{
    Llm as _, LlmRequest, Part,
    futures::StreamExt as _,
    model::openai::{OpenAIResponsesClient, OpenAIResponsesConfig},
};
use anyhow::Context;
use matrix_sdk::ruma::events::SyncMessageLikeEvent;
use matrix_sdk::{
    Client, Room, RoomState,
    config::SyncSettings,
    ruma::events::room::{
        member::StrippedRoomMemberEvent,
        message::{MessageType, RoomMessageEventContent},
    },
    sleep::sleep,
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
    client: Client,
}

#[derive(Clone)]
pub struct AdkOpenAiAgentConfig {
    open_responses_model: String,
    open_responses_base_url: String,
    openai_api_key: String,
}

#[derive(Clone)]
pub struct AdkOpenAiAgent {
    config: AdkOpenAiAgentConfig,
    client: Arc<OpenAIResponsesClient>,
}


type MessageEvent = SyncMessageLikeEvent<RoomMessageEventContent>;


#[derive(Clone)]
pub struct MatrixAdkAgent {
    matrix_agent: Arc<MatrixAgent>,
    adk_agent: Arc<AdkOpenAiAgent>,
    auto_join: bool,
    // TODO: Weitere Handler erlauben
    // join_handlers: Vec<JoinHandler>,
    // message_handlers: Vec<MessageHandler>,
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

    async fn connect_matrix(&self) -> Result<(), anyhow::Error> {
        self.client
            .matrix_auth()
            .login_username(&self.config.matrix_username, &self.config.matrix_password)
            .initial_device_display_name("autojoin bot")
            .await?;
        Ok(())
    }

    async fn sync(&self) -> Result<(), anyhow::Error> {
        self.client
            .sync(SyncSettings::default())
            .await
            .context("error syncing")
    }

    fn client(&self) -> Client {
        self.client.clone()
    }
}

impl AdkOpenAiAgentConfig {
    pub fn default_from_env() -> Result<Self, anyhow::Error> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("no OpenAI API Key given by env parameter OPENAI_API_KEY")?;
        let base_url = std::env::var("OPEN_RESPONSES_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model =
            std::env::var("OPEN_RESPONSES_MODEL").unwrap_or_else(|_| "gpt-4.1-nano".to_string());
        Ok(Self {
            open_responses_model: model,
            open_responses_base_url: base_url,
            openai_api_key: api_key,
        })
    }
}

impl AdkOpenAiAgent {
    pub async fn new(config: AdkOpenAiAgentConfig) -> Result<Self, anyhow::Error> {
        let ai_config =
            OpenAIResponsesConfig::new(&config.openai_api_key, &config.open_responses_model)
                .with_open_responses_mode(true)
                .with_base_url(&config.open_responses_base_url);

        let client = OpenAIResponsesClient::new(ai_config)?;

        Ok(Self {
            config: config,
            client: Arc::new(client),
        })
    }

    async fn ask(&self, message: String) -> Result<String, anyhow::Error> {
        // Build a simple request
        let request = LlmRequest {
            model: self.config.open_responses_model.clone(),
            contents: vec![Content::new("user").with_text(message)],
            config: None,
            tools: Default::default(),
            previous_response_id: None,
        };

        // Send prompt and stream the response
        println!(
            "📤 Sending prompt to {}...",
            self.config.open_responses_base_url
        );
        println!();

        let mut stream = self.client.generate_content(request, true).await?;
        let mut received_content = false;

        let mut response_text = String::new();
        while let Some(response) = stream.next().await {
            match response {
                Ok(llm_response) => {
                    if let Some(content) = &llm_response.content {
                        for part in &content.parts {
                            if let Part::Text { text } = part {
                                if !text.is_empty() {
                                    print!("{text}");
                                    response_text.push_str(text);
                                    received_content = true;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    // Gracefully handle errors that may occur with third-party
                    // providers due to missing OpenAI-specific fields.
                    eprintln!("⚠️  Stream error (may be expected with some providers): {e}");
                }
            }
        }

        if received_content {
            println!();
        }
        Ok(response_text)
    }
}

impl MatrixAdkAgent {
    pub fn new(
        matrix_agent: MatrixAgent,
        adk_agent: AdkOpenAiAgent,
        auto_join: bool,
        // join_handlers: Vec<JoinHandler>,
        // message_handlers: Vec<MessageHandler>,
    ) -> Self {
        let matrix_adk_agent = Self {
            matrix_agent: Arc::new(matrix_agent),
            adk_agent: Arc::new(adk_agent),
            auto_join: auto_join,
            // join_handlers: join_handlers,
            // message_handlers: message_handlers,
        };
        matrix_adk_agent
    }

    pub async fn connect_matrix(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        self.matrix_agent().connect_matrix().await?;
        if self.auto_join {
            let agent = Arc::clone(self);
            self.matrix_agent.client().add_event_handler(
                move |room_member: StrippedRoomMemberEvent, client: Client, room: Room| {
                    let agent = Arc::clone(&agent);
                    async move {
                        agent.on_stripped_state_member(room_member, client, room).await
                    }
                },
            );
        }
        Ok(())
    }

    pub async fn run(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        self.add_default_message_handler();
        self.matrix_agent.sync().await
    }

    fn matrix_agent(&self) -> Arc<MatrixAgent> {
        self.matrix_agent.clone()
    }

    // pub fn add_message_handler<F, Fut>(&mut self, f: F)
    // where
    //     F: Fn(Arc<MatrixAdkAgent>, MessageEvent, Room) -> Fut + Send + Sync + 'static,
    //     Fut: Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    // {
    //     self.message_handlers
    //         .push(Arc::new(move |agent, event, room| {
    //             Box::pin(f(agent, event, room))
    //         }));
    // }

    pub fn add_default_message_handler_for_room(self: &Arc<Self>, room: Room) {
        let agent: Arc<MatrixAdkAgent> = Arc::clone(self);
        room.add_event_handler(|event, room| async move {
            agent.on_room_message(event, room).await;
        });
    }

    pub fn add_default_message_handler(self: &Arc<Self>) {
        let agent = Arc::clone(self);
        self.matrix_agent.client.add_event_handler(|event, room| async move {
            agent.on_room_message(event, room).await;
        });
    }

    // pub fn register_message_handlers_for_room(self: &Arc<Self>, room: Room) {
    //     for handler in self.message_handlers.iter().cloned() {
    //         let agent = Arc::clone(self);

    //         room.add_event_handler(move |event: MessageEvent, room: Room| {
    //             let handler = Arc::clone(&handler);
    //             let agent = Arc::clone(&agent);

    //             async move { if let Err(err) = handler(agent, event, room).await {} }
    //         });
    //     }
    // }

    async fn on_stripped_state_member(
        self: &Arc<Self>,
        room_member: StrippedRoomMemberEvent,
        client: Client,
        room: Room,
    ) {
        if room_member.state_key != client.user_id().unwrap() {
            return;
        }
        let agent = Arc::clone(self);
        tokio::spawn(async move {
            println!("Autojoining room {}", room.room_id());
            let mut delay = 2;

            while let Err(err) = room.join().await {
                // retry autojoin due to synapse sending invites, before the
                // invited user can join for more information see
                // https://github.com/matrix-org/synapse/issues/4345
                eprintln!(
                    "Failed to join room {} ({err:?}), retrying in {delay}s",
                    room.room_id()
                );

                sleep(Duration::from_secs(delay)).await;
                delay *= 2;

                if delay > 3600 {
                    eprintln!("Can't join room {} ({err:?})", room.room_id());
                    break;
                }
            }
            if delay <= 3600 {
                println!("Successfully joined room {}", room.room_id());
                agent.add_default_message_handler_for_room(room);
            }
        });
    }

    pub async fn on_room_message(&self, event: MessageEvent, room: Room) {
        if room.state() != RoomState::Joined {
            return;
        }

        match event {
            SyncMessageLikeEvent::Original(event) => {
                let MessageType::Text(text_content) = event.content.msgtype else {
                    return;
                };

                if text_content.body.contains("!party") {
                    let party_content = self
                        .adk_agent
                        .ask("Explain what a party is.".to_string())
                        .await
                        .unwrap();

                    let content = RoomMessageEventContent::text_plain(party_content);

                    println!("sending");

                    // send our message to the room we found the "!party" command in
                    room.send(content).await.unwrap();

                    println!("message sent");
                }
            }
            SyncMessageLikeEvent::Redacted(_redacted) => {},
        }
    }
}
