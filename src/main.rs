use adk_rust::Launcher;
use adk_rust::prelude::*;
use adk_rust::serde_json::json;
use std::sync::Arc;

const DESC: &str = "Ollama-powered local assistant";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let description_tool = FunctionTool::new(
        "get_description",
        "Get description of the current ai agent (own description, introduction)",
        |_ctx, _args| async move { Ok(json!(DESC.to_string())) },
    );

    // No API key needed!
    let model = OllamaModel::new(OllamaConfig::new("llama3.2"))?;

    let agent = LlmAgentBuilder::new("ollama_assistant")
        .description(DESC)
        .instruction("You are a helpful assistant running locally via Ollama. Be concise.")
        .model(Arc::new(model))
        .tool(Arc::new(description_tool))
        .build()?;

    // Run interactive session
    Launcher::new(Arc::new(agent)).run().await?;

    Ok(())
}
