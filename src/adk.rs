use adk_rust::{futures::StreamExt as _, model::openai::{OpenAIResponsesClient, OpenAIResponsesConfig}, prelude::*};

pub async fn call_ai(request_str: String) -> anyhow::Result<String, anyhow::Error> {
    println!("═══════════════════════════════════════════════════");
    println!("  OpenAI Open Responses — Provider-Agnostic Example");
    println!("═══════════════════════════════════════════════════");
    println!();

    // Read configuration from environment with sensible defaults.
    // OPENAI_API_KEY uses unwrap_or_default() because third-party providers
    // may not require an API key at all.
    

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