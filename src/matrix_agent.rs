use std::{sync::Arc, sync::atomic::{AtomicU64, Ordering}, time::Duration};

use matrix_sdk::ruma::events::Mentions;
use matrix_sdk::ruma::events::SyncMessageLikeEvent;
use matrix_sdk::ruma::events::room::member::StrippedRoomMemberEvent;
use matrix_sdk::ruma::events::room::message::{MessageType, RoomMessageEventContent};
use matrix_sdk::ruma::{OwnedUserId, UserId};
use matrix_sdk::{Client, Room, RoomState};

use crate::adk::AdkOpenAiAgent;
use crate::matrix::MatrixAgent;

#[derive(Clone)]
pub struct MatrixAdkAgent {
    matrix_agent: Arc<MatrixAgent>,
    adk_agent: Arc<AdkOpenAiAgent>,
    auto_join: bool,
    task_counter: Arc<AtomicU64>,
    // TODO: Weitere Handler erlauben
    // join_handlers: Vec<JoinHandler>,
    // message_handlers: Vec<MessageHandler>,
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
            task_counter: Arc::new(AtomicU64::new(1)),
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
                        agent
                            .on_stripped_state_member(room_member, client, room)
                            .await
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
        self.matrix_agent
            .client
            .add_event_handler(|event, room| async move {
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

                matrix_sdk::sleep::sleep(Duration::from_secs(delay)).await;
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

    pub async fn on_room_message(
        &self,
        event: SyncMessageLikeEvent<RoomMessageEventContent>,
        room: Room,
    ) {
        if room.state() != RoomState::Joined {
            return;
        }

        match event {
            SyncMessageLikeEvent::Original(event) => {
                let MessageType::Text(text_content) = event.content.msgtype else {
                    return;
                };

                let message_body = text_content.body.clone();
                if let Some(introduction) = Self::extract_introduction(&message_body) {
                    let sender = event.sender.to_string();
                    self.adk_agent
                        .remember_introduction(sender.clone(), introduction.clone());
                    println!("Stored introduction from {sender}: {introduction}");
                }

                let mentioned = event.content.mentions.iter().any(|mention| {
                    if mention.room {
                        return true;
                    }
                    mention
                        .user_ids
                        .iter()
                        .any(|mention| mention == self.matrix_agent.client().user_id().unwrap())
                });
                if !mentioned {
                    println!("got message, but not mentioned");
                    return;
                }

                let task_context = self.build_task_context(&event.sender.to_string(), room.room_id().to_string(), &message_body);
                let question = task_context.to_prompt();
                println!("got message and was mentioned, asking llm: {question}");
                if let Err(_) = room.typing_notice(true).await {}
                let result_content = self
                    .adk_agent
                    .ask(question)
                    .await
                    .unwrap();
                println!("Result: {result_content}");
                
                if let Err(_) = room.typing_notice(false).await {}

                if !result_content.trim().is_empty() {
                    let response = Self::parse_agent_response(&result_content);
                    let mentioned_user_ids = self.resolve_mentions(&response, &task_context);
                    let outgoing_text = self.compose_outgoing_message(response, &task_context);

                    let mut content = RoomMessageEventContent::text_plain(outgoing_text);
                    if !mentioned_user_ids.is_empty() {
                        content = content.add_mentions(Mentions::with_user_ids(mentioned_user_ids));
                    }

                    println!("sending");
                    room.send(content).await.unwrap();
                    println!("message sent");
                }

            }
            SyncMessageLikeEvent::Redacted(_redacted) => {}
        }
    }

    fn extract_introduction(message: &str) -> Option<String> {
        let trimmed = message.trim_start();
        let introduction = trimmed
            .strip_prefix("[Introduction]")
            .or_else(|| trimmed.strip_prefix("[introduction]"))?
            .trim_start_matches(':')
            .trim();

        if introduction.is_empty() {
            return None;
        }

        Some(introduction.to_string())
    }

    fn extract_helper_tags(message: &str) -> (String, Vec<String>) {
        let mut cleaned = String::with_capacity(message.len());
        let mut helper_names = Vec::new();
        let mut remaining = message;

        while let Some(start) = remaining.find("@[") {
            cleaned.push_str(&remaining[..start]);

            let after_start = &remaining[start + 2..];
            let Some(end) = after_start.find(']') else {
                cleaned.push_str("@[");
                cleaned.push_str(after_start);
                return (cleaned, helper_names);
            };

            let helper_name = after_start[..end].trim();
            if !helper_name.is_empty() {
                helper_names.push(helper_name.to_string());
                cleaned.push('@');
                cleaned.push_str(helper_name);
            }

            remaining = &after_start[end + 1..];
        }

        cleaned.push_str(remaining);
        (cleaned, helper_names)
    }

    fn build_task_context(&self, sender: &str, room_id: String, message: &str) -> TaskContext {
        let parsed = TaskMetadata::parse(message);
        let task_id = parsed
            .task_id
            .unwrap_or_else(|| self.next_task_id());
        let requester = parsed
            .requester
            .unwrap_or_else(|| sender.to_string());

        TaskContext {
            task_id,
            requester,
            sender: sender.to_string(),
            room_id,
            original_message: parsed.body,
            has_existing_task_id: parsed.had_task_id,
        }
    }

    fn next_task_id(&self) -> String {
        let task_number = self.task_counter.fetch_add(1, Ordering::Relaxed);
        format!("task-{task_number}")
    }

    fn parse_agent_response(message: &str) -> AgentResponse {
        let completion_task_id = Self::extract_marker_value(message, "[TaskComplete:", ']');
        let requester = Self::extract_marker_value(message, "[Requester:", ']');
        let task_id = Self::extract_marker_value(message, "[Task:", ']');
        let without_internal_markers = Self::strip_marker(message, "[TaskComplete:");
        let without_requester_marker = Self::strip_marker(&without_internal_markers, "[Requester:");
        let without_task_marker = Self::strip_marker(&without_requester_marker, "[Task:");
        let (text, helper_names) = Self::extract_helper_tags(&without_task_marker);

        AgentResponse {
            text: text.trim().to_string(),
            helper_names,
            task_id,
            completion_task_id,
            requester,
        }
    }

    fn compose_outgoing_message(&self, response: AgentResponse, task_context: &TaskContext) -> String {
        let is_completion = response.is_completion();
        let mut text = response.text;
        let effective_task_id = response
            .task_id
            .clone()
            .or(response.completion_task_id.clone())
            .unwrap_or_else(|| task_context.task_id.clone());
        let should_prefix_task = !response.helper_names.is_empty()
            || response.completion_task_id.is_some()
            || task_context.has_existing_task_id;

        if should_prefix_task {
            text = format!("[Task: {effective_task_id}] {text}").trim().to_string();
        }

        if text.is_empty() && is_completion {
            text = format!("[Task: {effective_task_id}]");
        }

        text
    }

    fn resolve_mentions(&self, response: &AgentResponse, task_context: &TaskContext) -> Vec<OwnedUserId> {
        let mut user_ids = self.resolve_user_mentions(&response.helper_names);

        if response.is_completion() {
            let requester = response
                .requester
                .as_deref()
                .unwrap_or(&task_context.requester);
            if let Ok(user_id) = UserId::parse(requester) {
                user_ids.push(user_id);
            } else {
                eprintln!("Skipping invalid requester user id '{requester}'");
            }
        }

        user_ids.sort();
        user_ids.dedup();
        user_ids
    }

    fn extract_marker_value(message: &str, prefix: &str, suffix: char) -> Option<String> {
        let start = message.find(prefix)?;
        let after_prefix = &message[start + prefix.len()..];
        let end = after_prefix.find(suffix)?;
        let value = after_prefix[..end].trim();
        if value.is_empty() {
            return None;
        }
        Some(value.to_string())
    }

    fn strip_marker(message: &str, prefix: &str) -> String {
        let mut stripped = String::with_capacity(message.len());
        let mut remaining = message;

        while let Some(start) = remaining.find(prefix) {
            stripped.push_str(&remaining[..start]);
            let after_prefix = &remaining[start + prefix.len()..];
            let Some(end) = after_prefix.find(']') else {
                stripped.push_str(&remaining[start..]);
                return stripped;
            };
            remaining = &after_prefix[end + 1..];
        }

        stripped.push_str(remaining);
        stripped
    }

    fn resolve_user_mentions(&self, helper_names: &[String]) -> Vec<OwnedUserId> {
        let mut user_ids = Vec::new();

        for helper_name in helper_names {
            for user_id in self.adk_agent.find_helper_user_ids_by_name(helper_name) {
                match UserId::parse(user_id.clone()) {
                    Ok(owned) => user_ids.push(owned),
                    Err(_) => {
                        eprintln!("Skipping invalid helper user id '{user_id}' for '{helper_name}'");
                    }
                }
            }
        }

        user_ids.sort();
        user_ids.dedup();
        user_ids
    }
}

struct TaskMetadata {
    task_id: Option<String>,
    requester: Option<String>,
    body: String,
    had_task_id: bool,
}

impl TaskMetadata {
    fn parse(message: &str) -> Self {
        let task_id = MatrixAdkAgent::extract_marker_value(message, "[Task:", ']');
        let requester = MatrixAdkAgent::extract_marker_value(message, "[Requester:", ']');
        let without_requester = MatrixAdkAgent::strip_marker(message, "[Requester:");
        let body = MatrixAdkAgent::strip_marker(&without_requester, "[Task:")
            .trim()
            .to_string();

        Self {
            had_task_id: task_id.is_some(),
            task_id,
            requester,
            body,
        }
    }
}

struct TaskContext {
    task_id: String,
    requester: String,
    sender: String,
    room_id: String,
    original_message: String,
    has_existing_task_id: bool,
}

impl TaskContext {
    fn to_prompt(&self) -> String {
        format!(
            "[TaskContext]\ntask_id: {}\nrequester: {}\nsender: {}\nroom_id: {}\noriginal_message:\n{}",
            self.task_id, self.requester, self.sender, self.room_id, self.original_message
        )
    }
}

struct AgentResponse {
    text: String,
    helper_names: Vec<String>,
    task_id: Option<String>,
    completion_task_id: Option<String>,
    requester: Option<String>,
}

impl AgentResponse {
    fn is_completion(&self) -> bool {
        self.completion_task_id.is_some() || (self.helper_names.is_empty() && !self.text.is_empty())
    }
}
