use adk_rust::{
    futures::StreamExt as _,
    model::openai::{OpenAIResponsesClient, OpenAIResponsesConfig},
    prelude::*,
    serde_json::json,
    Artifacts, CallbackContext, InvocationContext, Memory, ReadonlyContext, RunConfig, Session,
    State,
};

use anyhow::{Context, Result};
use std::{collections::HashMap, sync::atomic::{AtomicBool, Ordering}, sync::{Arc, Mutex}};

#[derive(Clone)]
pub struct AdkOpenAiAgentConfig {
    introduction: Option<String>,
    system_prompt: Option<String>,
    open_responses_model: String,
    open_responses_base_url: String,
    openai_api_key: String,
}

#[derive(Clone)]
pub struct AdkOpenAiAgent {
    config: AdkOpenAiAgentConfig,
    client: Arc<OpenAIResponsesClient>,
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
            introduction: None,
            system_prompt: None,
            open_responses_model: model,
            open_responses_base_url: base_url,
            openai_api_key: api_key,
        })
    }
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }
    pub fn with_introduction(mut self, introduction: String) -> Self {
        self.introduction = Some(introduction);
        self
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

    pub fn introduction(&self) -> String {
        if let Some(introduction) = &self.config.introduction {
            introduction.clone()
        } else {
            "Hello! I am an AI assistant. How can I help you today?".to_string()
        }
    }

    pub async fn ask(self: &Arc<Self>, message: String) -> Result<String, anyhow::Error> {
        let system_prompt = self.config.system_prompt.clone().unwrap_or( 
            "You are a helpful assistant. If you are asked to introduce yourself, use the introduction tool and return only its result, nothing more.".to_string()
        );

        let introduction = self.introduction();
        let introduction_tool = Arc::new(FunctionTool::new(
            "introduce_yourself",
            "Return the configured self-introduction string.",
            move |_ctx, _args| {
                let introduction = introduction.clone();
                async move { Ok(json!(introduction)) }
            },
        ));

        let llm_agent = Arc::new(
            LlmAgentBuilder::new("adk-openai-agent")
                .description("Agent wrapper around the OpenAI Responses client")
                .model(self.client.clone())
                .instruction(system_prompt)
                .tool(introduction_tool)
                .build()?,
        );

        let invocation_context = Arc::new(SimpleInvocationContext::new(llm_agent.clone(), message));

        // Send prompt and stream the response
        println!(
            "📤 Sending prompt to {}...",
            self.config.open_responses_base_url
        );
        println!();

        let mut stream = llm_agent.run(invocation_context).await?;
        let mut received_content = false;

        let mut response_text = String::new();
        while let Some(response) = stream.next().await {
            match response {
                Ok(event) => {
                    if let Some(content) = event.content() {
                        for part in &content.parts {
                            if let Part::Text { text } = part && !text.is_empty() {
                                print!("{text}");
                                response_text.push_str(text);
                                received_content = true;
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

struct SimpleState {
    values: Mutex<HashMap<String, adk_rust::serde_json::Value>>,
}

impl SimpleState {
    fn new() -> Self {
        Self {
            values: Mutex::new(HashMap::new()),
        }
    }
}

impl State for SimpleState {
    fn get(&self, key: &str) -> Option<adk_rust::serde_json::Value> {
        self.values.lock().ok().and_then(|values| values.get(key).cloned())
    }

    fn set(&mut self, key: String, value: adk_rust::serde_json::Value) {
        if let Ok(mut values) = self.values.lock() {
            values.insert(key, value);
        }
    }

    fn all(&self) -> HashMap<String, adk_rust::serde_json::Value> {
        self.values.lock().map(|values| values.clone()).unwrap_or_default()
    }
}

struct SimpleSession {
    state: SimpleState,
}

impl SimpleSession {
    fn new() -> Self {
        Self {
            state: SimpleState::new(),
        }
    }
}

impl Session for SimpleSession {
    fn id(&self) -> &str {
        "adk-openai-session"
    }

    fn app_name(&self) -> &str {
        "cli-llm-agent"
    }

    fn user_id(&self) -> &str {
        "cli-user"
    }

    fn state(&self) -> &dyn State {
        &self.state
    }

    fn conversation_history(&self) -> Vec<Content> {
        Vec::new()
    }
}

struct SimpleInvocationContext {
    agent: Arc<dyn Agent>,
    content: Content,
    config: RunConfig,
    session: SimpleSession,
    ended: AtomicBool,
}

impl SimpleInvocationContext {
    fn new(agent: Arc<dyn Agent>, message: String) -> Self {
        Self {
            agent,
            content: Content::new("user").with_text(message),
            config: RunConfig::default(),
            session: SimpleSession::new(),
            ended: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl ReadonlyContext for SimpleInvocationContext {
    fn invocation_id(&self) -> &str {
        "adk-openai-invocation"
    }

    fn agent_name(&self) -> &str {
        self.agent.name()
    }

    fn user_id(&self) -> &str {
        self.session.user_id()
    }

    fn app_name(&self) -> &str {
        self.session.app_name()
    }

    fn session_id(&self) -> &str {
        self.session.id()
    }

    fn branch(&self) -> &str {
        ""
    }

    fn user_content(&self) -> &Content {
        &self.content
    }
}

#[async_trait]
impl CallbackContext for SimpleInvocationContext {
    fn artifacts(&self) -> Option<Arc<dyn Artifacts>> {
        None
    }
}

#[async_trait]
impl InvocationContext for SimpleInvocationContext {
    fn agent(&self) -> Arc<dyn Agent> {
        self.agent.clone()
    }

    fn memory(&self) -> Option<Arc<dyn Memory>> {
        None
    }

    fn session(&self) -> &dyn Session {
        &self.session
    }

    fn run_config(&self) -> &RunConfig {
        &self.config
    }

    fn end_invocation(&self) {
        self.ended.store(true, Ordering::SeqCst);
    }

    fn ended(&self) -> bool {
        self.ended.load(Ordering::SeqCst)
    }
}