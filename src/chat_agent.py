import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio

# Load environment variables from .env file
load_dotenv()

# Configure the AI model provider from environment variables
# Configure the AI model provider for OpenRouter
model_choice = os.getenv("MODEL_CHOICE", "gpt-4o") # Default model
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1") # Default OpenRouter base URL

if not api_key:
    raise ValueError("API key for OpenRouter (OPENROUTER_API_KEY) not found in .env file.")

# Configure the OpenAI provider for OpenRouter
provider_instance = OpenAIProvider(
    api_key=api_key,
    base_url=base_url
)

# Configure the OpenAI model
# Note: For OpenRouter, the model name doesn't need the 'openrouter:' prefix
# when using OpenAIModel and OpenAIProvider explicitly.
model_instance = OpenAIModel(
    model_choice,
    provider=provider_instance
)

# Configure the MCP Email Server connection
email_server = MCPServerStdio(
    'python', # Command to run the server script
    args=['src/email_mcp_server.py'] # Arguments for the command
)

# Create the agent instance with MCP server configuration
agent = Agent(
    model_instance,
    system_prompt='Be a helpful chat assistant. You have a tool called `send_email_summary` available via an MCP server to send email summaries.',
    mcp_servers=[email_server] # Add the server config here
)

message_history: list[ModelMessage] = []

async def chat(user_input: str) -> str:
    """Runs the chat agent asynchronously with the given input and maintains history."""
    global message_history
    # Use await agent.run for async execution
    result = await agent.run(user_input, message_history=message_history)
    message_history = result.all_messages()
    # Access the content from the last message in the history
    # Check if the last part is text before accessing content
    last_message = result.all_messages()[-1]
    if last_message.parts and hasattr(last_message.parts[0], 'content'):
         return last_message.parts[0].content
    elif hasattr(last_message, 'content') and isinstance(last_message.content, str): # Handle potential direct content
         return last_message.content
    else:
         # Handle cases where the last message might be a tool call/result without simple text
         # You might want more sophisticated logic here depending on expected tool interactions
         return "[Agent action completed]"


async def main():
    """Main asynchronous function to run the chat loop."""
    print("Chat Agent Initialized. Type 'quit' to exit.")
    # Manage MCP server lifecycle
    async with agent.run_mcp_servers():
        print("MCP Email Server started.")
        while True:
            # Use asyncio.to_thread for synchronous input in async context
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() == 'quit':
                break
            response = await chat(user_input)
            print(f"Agent: {response}")
        print("Exiting chat.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat interrupted by user.")