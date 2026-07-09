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

use std::sync::Arc;

use crate::{adk::{AdkOpenAiAgent, AdkOpenAiAgentConfig}, matrix::{MatrixAgent, MatrixAgentConfig}, matrix_agent::MatrixAdkAgent};

pub mod adk;
pub mod matrix;
pub mod matrix_agent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    
    let ai_agent = AdkOpenAiAgent::new(AdkOpenAiAgentConfig::default_from_env()?).await?;
    let matrix_agent = MatrixAgent::new(MatrixAgentConfig::default_from_env()?).await?;
    let matrix_adk_agent = Arc::new(MatrixAdkAgent::new(matrix_agent, ai_agent, true));
    matrix_adk_agent.connect_matrix().await?;
    matrix_adk_agent.run().await?;
    
    Ok(())
}