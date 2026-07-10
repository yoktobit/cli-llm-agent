use std::{sync::Arc, time::Duration};

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
                
                let question = message_body;
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
                    let (outgoing_text, helper_names) = Self::extract_helper_tags(&result_content);
                    let mentioned_user_ids = self.resolve_user_mentions(&helper_names);

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
