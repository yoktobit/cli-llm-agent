//! OpenAI Responses API — Open Responses Mode example.
//!
//! Demonstrates provider-agnostic usage with a configurable endpoint. Open
//! Responses mode relaxes strict OpenAI field validation so you can connect to
//! any compatible third-party provider (LM Studio, Ollama, vLLM) without code
//! changes.
//!
//! # Running
//!
//! ```bash
//! # Using OpenAI directly (default):
//! export OPENAI_API_KEY=sk-...
//! cargo run --manifest-path examples/openai_open_responses/Cargo.toml
//!
//! # Using a local provider (e.g., LM Studio):
//! export OPEN_RESPONSES_BASE_URL=http://localhost:1234/v1
//! export OPEN_RESPONSES_MODEL=local-model
//! cargo run --manifest-path examples/openai_open_responses/Cargo.toml
//! ```

use std::time::Duration;

use adk_rust::{futures::StreamExt as _, model::openai::{OpenAIResponsesClient, OpenAIResponsesConfig}, prelude::*};
use matrix_sdk::{Client, Room, RoomState, config::SyncSettings, ruma::events::room::{member::StrippedRoomMemberEvent, message::{MessageType, OriginalSyncRoomMessageEvent, RoomMessageEventContent}}, sleep::sleep};

async fn on_room_message(event: OriginalSyncRoomMessageEvent, room: Room) {
    if room.state() != RoomState::Joined {
        return;
    }
    let MessageType::Text(text_content) = event.content.msgtype else {
        return;
    };

    if text_content.body.contains("!party") {

        let party_content = call_ai("Explain what a party is.".to_string()).await.unwrap();

        let content = RoomMessageEventContent::text_plain(party_content);

        println!("sending");

        // send our message to the room we found the "!party" command in
        room.send(content).await.unwrap();

        println!("message sent");
    }
}

async fn on_stripped_state_member(
    room_member: StrippedRoomMemberEvent,
    client: Client,
    room: Room,
) {
    if room_member.state_key != client.user_id().unwrap() {
        return;
    }

    tokio::spawn(async move {
        println!("Autojoining room {}", room.room_id());
        let mut delay = 2;

        while let Err(err) = room.join().await {
            // retry autojoin due to synapse sending invites, before the
            // invited user can join for more information see
            // https://github.com/matrix-org/synapse/issues/4345
            eprintln!("Failed to join room {} ({err:?}), retrying in {delay}s", room.room_id());

            sleep(Duration::from_secs(delay)).await;
            delay *= 2;

            if delay > 3600 {
                eprintln!("Can't join room {} ({err:?})", room.room_id());
                break;
            }
        }
        if delay <= 3600 {
            println!("Successfully joined room {}", room.room_id());
            client.add_event_handler(on_room_message);
            let banana_desc = call_ai("Explain what a banana is.".to_string()).await.unwrap();
            println!("AI response: {}", banana_desc);
            let message = RoomMessageEventContent::text_plain(banana_desc);
            room.send(message).await.unwrap();
        }
    });
}

async fn login_and_sync(
    homeserver_url: String,
    username: &str,
    password: &str,
) -> anyhow::Result<()> {
    // Note that when encryption is enabled, you should use a persistent store to be
    // able to restore the session with a working encryption setup.
    // See the `persist_session` example.
    let client = Client::builder().homeserver_url(homeserver_url).build().await?;

    client
        .matrix_auth()
        .login_username(username, password)
        .initial_device_display_name("autojoin bot")
        .await?;

    println!("logged in as {username}");

    client.add_event_handler(on_stripped_state_member);

    client.sync(SyncSettings::default()).await?;

    Ok(())
}

async fn call_ai(request_str: String) -> anyhow::Result<String, anyhow::Error> {
    println!("═══════════════════════════════════════════════════");
    println!("  OpenAI Open Responses — Provider-Agnostic Example");
    println!("═══════════════════════════════════════════════════");
    println!();

    // Read configuration from environment with sensible defaults.
    // OPENAI_API_KEY uses unwrap_or_default() because third-party providers
    // may not require an API key at all.
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let base_url = std::env::var("OPEN_RESPONSES_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let model = std::env::var("OPEN_RESPONSES_MODEL")
        .unwrap_or_else(|_| "gpt-4.1-nano".to_string());

    println!("🔧 Configuration:");
    println!("   Base URL: {base_url}");
    println!("   Model:    {model}");
    println!(
        "   API Key:  {}",
        if api_key.is_empty() {
            "(not set — using third-party provider)"
        } else {
            "(set)"
        }
    );
    println!();

    // Configure the client with Open Responses mode enabled and custom base URL.
    let config = OpenAIResponsesConfig::new(&api_key, &model)
        .with_open_responses_mode(true)
        .with_base_url(&base_url);

    let client = OpenAIResponsesClient::new(config)?;

    // Build a simple request
    let request = LlmRequest {
        model: model.clone(),
        contents: vec![Content::new("user").with_text(
            request_str,
        )],
        config: None,
        tools: Default::default(),
        previous_response_id: None,
    };

    // Send prompt and stream the response
    println!("📤 Sending prompt to {base_url}...");
    println!();

    let mut stream = client.generate_content(request, true).await?;
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

    println!();
    println!("✅ Example completed successfully.");

    Ok(response_text)

}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    
    let homeserver_url = std::env::var("MATRIX_HOMESERVER_URL")
        .unwrap_or_else(|_| "https://matrix.org".to_string());
    let username = std::env::var("MATRIX_USERNAME").expect("No MATRIX_USERNAME given as env parameter");
    let password = std::env::var("MATRIX_PASSWORD").expect("No MATRIX_PASSWORD given as env parameter");

    login_and_sync(homeserver_url, &username, &password).await?;


    
    Ok(())
}