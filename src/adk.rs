use adk_rust::{
    futures::StreamExt as _,
    model::openai::{OpenAIResponsesClient, OpenAIResponsesConfig},
    prelude::*,
    serde_json::json,
    Artifacts, CallbackContext, InvocationContext, Memory, ReadonlyContext, RunConfig, Session,
    State,
};

use anyhow::{Context, Result};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const COLLABORATION_PROTOCOL: &str = "When you receive a chat message, first inspect the known participant introductions with the get_known_introductions tool before deciding whether to answer yourself or collaborate. Use those introductions to judge who is best suited for the task.\n\nIf you are asked to introduce yourself, use the introduce_yourself tool and return only its result, nothing more.\n\nYou may receive a task context block containing task_id, requester, sender, room_id, and the original message. Keep the task_id stable across all collaboration messages.\n\nIf another participant is better suited or collaboration is needed, respond with a message that includes `[Task: <task_id>]` and mention collaborators as `@[Agent-Name]`.\n\nIf you complete the task yourself, include `[TaskComplete: <task_id>]` in your reply. When a requester is provided in the context, also include `[Requester: <requester_user_id>]`.\n\nDo not invent collaborators. Base delegation decisions on known introductions.";

#[derive(Clone)]
pub struct AdkOpenAiAgentConfig {
    introduction: Option<String>,
    system_prompt: Option<String>,
    introductions_file: PathBuf,
    open_responses_model: String,
    open_responses_base_url: String,
    openai_api_key: String,
}

#[derive(Clone)]
pub struct AdkOpenAiAgent {
    config: AdkOpenAiAgentConfig,
    client: Arc<OpenAIResponsesClient>,
    introductions: Arc<Mutex<HashMap<String, String>>>,
}

impl AdkOpenAiAgentConfig {
    pub fn default_from_env() -> Result<Self, anyhow::Error> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("no OpenAI API Key given by env parameter OPENAI_API_KEY")?;
        let base_url = std::env::var("OPEN_RESPONSES_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model =
            std::env::var("OPEN_RESPONSES_MODEL").unwrap_or_else(|_| "gpt-4.1-nano".to_string());
        let introductions_file = std::env::var("INTRODUCTIONS_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".matrix-store/introductions.json"));
        Ok(Self {
            introduction: None,
            system_prompt: None,
            introductions_file,
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
        if let Some(parent) = config.introductions_file.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create introductions dir {}",
                    parent.display()
                )
            })?;
        }

        let ai_config =
            OpenAIResponsesConfig::new(&config.openai_api_key, &config.open_responses_model)
                .with_open_responses_mode(true)
                .with_reasoning_effort(adk_rust::model::ReasoningEffort::Low)
                .with_reasoning_summary(adk_rust::model::openai::ReasoningSummary::Concise)
                .with_base_url(&config.open_responses_base_url);

        let client = OpenAIResponsesClient::new(ai_config)?;
        let introductions = Self::load_introductions(&config.introductions_file)?;

        Ok(Self {
            config: config,
            client: Arc::new(client),
            introductions: Arc::new(Mutex::new(introductions)),
        })
    }

    pub fn remember_introduction(&self, name: String, introduction: String) {
        if let Ok(mut introductions) = self.introductions.lock() {
            introductions.insert(name, introduction);
            if let Err(err) = Self::persist_introductions(
                &self.config.introductions_file,
                &introductions,
            ) {
                eprintln!(
                    "Failed to persist introductions to {}: {err}",
                    self.config.introductions_file.display()
                );
            }
        }
    }

    pub fn find_helper_user_ids_by_name(&self, name: &str) -> Vec<String> {
        let requested = name.trim().to_lowercase();
        if requested.is_empty() {
            return Vec::new();
        }

        let Ok(introductions) = self.introductions.lock() else {
            return Vec::new();
        };

        let mut matches: Vec<String> = introductions
            .keys()
            .filter(|known_name| {
                let known = known_name.to_lowercase();
                known == requested || known.contains(&requested)
            })
            .cloned()
            .collect();
        matches.sort();
        matches.dedup();
        matches
    }

    pub fn introduction(&self) -> String {
        if let Some(introduction) = &self.config.introduction {
            introduction.clone()
        } else {
            "Hello! I am an AI assistant. How can I help you today?".to_string()
        }
    }

    pub async fn ask(self: &Arc<Self>, message: String) -> Result<String, anyhow::Error> {
        let mut system_prompt = self
            .config
            .system_prompt
            .clone()
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

        system_prompt.push_str("\n\n");
        system_prompt.push_str(COLLABORATION_PROTOCOL);

        let introduction = self.introduction();
        let introduction_tool = Arc::new(FunctionTool::new(
            "introduce_yourself",
            "Return the configured self-introduction string.",
            move |_ctx, _args| {
                let introduction = format!("[Introduction]: {}", introduction.clone());
                async move { Ok(json!(introduction)) }
            },
        ));

        let introductions = Arc::clone(&self.introductions);
        let known_introductions_tool = Arc::new(FunctionTool::new(
            "get_known_introductions",
            "Return all known introductions. Optional arg 'name' filters by helper name (supports case-insensitive partial match).",
            move |_ctx, args| {
                let introductions = Arc::clone(&introductions);
                async move {
                    let requested_name = args
                        .get("name")
                        .and_then(|value| value.as_str())
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                        .map(ToOwned::to_owned);

                    let introductions = match introductions.lock() {
                        Ok(introductions) => introductions,
                        Err(_) => {
                            return Ok(json!({
                                "requested_name": requested_name,
                                "matches": [],
                                "all": [],
                                "error": "failed to read introductions registry"
                            }));
                        }
                    };

                    let mut all_entries: Vec<(String, String)> = introductions
                        .iter()
                        .map(|(name, intro)| (name.clone(), intro.clone()))
                        .collect();
                    all_entries.sort_by(|a, b| a.0.cmp(&b.0));

                    let all_json: Vec<_> = all_entries
                        .iter()
                        .map(|(name, introduction)| {
                            json!({"name": name, "introduction": introduction})
                        })
                        .collect();

                    if let Some(requested_name) = requested_name {
                        let requested_lower = requested_name.to_lowercase();
                        let matches_json: Vec<_> = all_entries
                            .iter()
                            .filter(|(name, _)| {
                                name.to_lowercase() == requested_lower
                                    || name.to_lowercase().contains(&requested_lower)
                            })
                            .map(|(name, introduction)| {
                                json!({"name": name, "introduction": introduction})
                            })
                            .collect();

                        return Ok(json!({
                            "requested_name": requested_name,
                            "matches": matches_json,
                            "all": all_json,
                        }));
                    }

                    Ok(json!({ "all": all_json }))
                }
            },
        ));

        let llm_agent = Arc::new(
            LlmAgentBuilder::new("adk-openai-agent")
                .description("Agent wrapper around the OpenAI Responses client")
                .model(self.client.clone())
                .instruction(system_prompt)
                .tool(introduction_tool)
                .tool(known_introductions_tool)
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

    fn load_introductions(path: &PathBuf) -> Result<HashMap<String, String>, anyhow::Error> {
        let serialized = match fs::read(path) {
            Ok(contents) => contents,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("failed to read introductions file {}", path.display())
                });
            }
        };

        serde_json::from_slice(&serialized).with_context(|| {
            format!("failed to parse introductions file {}", path.display())
        })
    }

    fn persist_introductions(
        path: &PathBuf,
        introductions: &HashMap<String, String>,
    ) -> Result<(), anyhow::Error> {
        let serialized = serde_json::to_vec_pretty(introductions)
            .context("failed to serialize known introductions")?;
        fs::write(path, serialized)
            .with_context(|| format!("failed to write introductions file {}", path.display()))
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
        "matrix-adk-agent-rs"
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