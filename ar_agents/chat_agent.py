import os
import asyncio
import pathlib
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Sequence
import logging

# Pydantic AI Imports
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart, UserPromptPart, TextPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import Tool as PydanticTool

# LangGraph Imports
from langgraph.graph import StateGraph, END

# Local Imports
import mcp_client # Import the MCP client wrapper

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the .env file relative to the script directory
DOTENV_PATH = SCRIPT_DIR / "../.env"
# Define the path to the mcp_config.json file relative to the script directory
MCP_CONFIG_PATH = SCRIPT_DIR / "../mcp_config.json"

load_dotenv(dotenv_path=DOTENV_PATH)

# --- LLM Configuration ---
def get_llm_model() -> OpenAIModel:
    """Configures and returns the Pydantic AI model wrapper for OpenRouter."""
    model_choice = os.getenv("MODEL_CHOICE", "gpt-4o") # Default model
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("BASE_URL", "https://openrouter.ai/api/v1") # Default OpenRouter base URL

    if not api_key:
        raise ValueError("API key for OpenRouter (OPENROUTER_API_KEY) not found in .env file.")

    provider_instance = OpenAIProvider(api_key=api_key, base_url=base_url)
    model_instance = OpenAIModel(model_choice, provider=provider_instance)
    logging.info(f"LLM Model configured: {model_choice} via OpenRouter")
    return model_instance

# --- Global Variables (initialized in main) ---
mcp_instance: mcp_client.MCPClient | None = None
pydantic_agent: Agent | None = None
mcp_tools_dict: dict[str, PydanticTool] = {} # Store tools by name for easy lookup

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Represents the state of our graph."""
    messages: Annotated[Sequence[ModelMessage], operator.add]

# --- LangGraph Nodes ---
async def call_model(state: AgentState) -> dict:
    """Calls the Pydantic AI agent."""
    if not pydantic_agent:
        raise RuntimeError("Pydantic Agent not initialized.")
    messages = state['messages']
    logging.info(f"Calling model with {len(messages)} messages.")
    # The agent internally handles history and tool calls based on its configuration
    result = await pydantic_agent.run(messages[-1].parts[0].content, message_history=messages[:-1]) # Pass last message as input, rest as history
    new_messages = result.new_messages() # Get only the messages added in this run
    logging.info(f"Model returned {len(new_messages)} new messages.")
    return {"messages": new_messages}

async def call_tool(state: AgentState) -> dict:
    """Executes tools based on the model's request."""
    last_message = state['messages'][-1]
    tool_return_messages = []

    # Check if the last message contains tool calls
    if isinstance(last_message, ModelMessage) and last_message.parts:
        for part in last_message.parts:
            if isinstance(part, ToolCallPart):
                tool_name = part.tool_name
                tool_args = part.tool_args
                logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                if tool_name in mcp_tools_dict:
                    tool_to_call = mcp_tools_dict[tool_name]
                    try:
                        # Execute the tool function (which calls the MCP server via mcp_client)
                        observation = await tool_to_call.func(**tool_args)
                        logging.info(f"Tool '{tool_name}' executed. Observation: {str(observation)[:100]}...")
                        # Create a ToolReturnPart message
                        tool_return_messages.append(ModelMessage(parts=[ToolReturnPart(tool_call_id=part.tool_call_id, content=str(observation))]))
                    except Exception as e:
                        logging.error(f"Error executing tool {tool_name}: {e}")
                        tool_return_messages.append(ModelMessage(parts=[ToolReturnPart(tool_call_id=part.tool_call_id, content=f"Error executing tool: {e}")]))
                else:
                    logging.warning(f"Tool '{tool_name}' requested but not found.")
                    tool_return_messages.append(ModelMessage(parts=[ToolReturnPart(tool_call_id=part.tool_call_id, content=f"Error: Tool '{tool_name}' not available.")]))

    if not tool_return_messages:
         # This case should ideally not be reached if routing is correct
         logging.warning("call_tool node reached without ToolCallPart in the last message.")
         return {"messages": []} # Return empty list if no tool was called

    return {"messages": tool_return_messages}


# --- LangGraph Conditional Edge ---
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the last message."""
    last_message = state['messages'][-1]
    # Check if any part of the last message is a ToolCallPart
    if isinstance(last_message, ModelMessage) and any(isinstance(part, ToolCallPart) for part in last_message.parts):
        logging.info("Routing: Tool call detected, routing to 'call_tool'.")
        return "call_tool"
    else:
        logging.info("Routing: No tool call detected, routing to END.")
        return END

# --- Build the Graph ---
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Build graph
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "action",
        END: END,
    },
)
workflow.add_edge("action", "agent") # After action, go back to agent to process tool result

# Compile the graph
app = workflow.compile()

# --- Main Execution Logic ---
async def main():
    global mcp_instance, pydantic_agent, mcp_tools_dict # Allow modification

    print("Initializing LangGraph Chat Agent with MCP...")
    mcp_instance = mcp_client.MCPClient()

    try:
        # Load MCP config and start servers/tools
        mcp_instance.load_servers(str(MCP_CONFIG_PATH))
        mcp_pydantic_tools = await mcp_instance.start()

        if not mcp_pydantic_tools:
             print("Error: Failed to retrieve tools from MCP servers. Exiting.")
             logging.error("MCP Client failed to start or retrieve tools.")
             return

        # Store tools in a dictionary for easy lookup in the tool node
        mcp_tools_dict = {tool.name: tool for tool in mcp_pydantic_tools}
        logging.info(f"Initialized MCP tools: {list(mcp_tools_dict.keys())}")

        # Initialize Pydantic Agent
        llm_model = get_llm_model()
        pydantic_agent = Agent(
            model=llm_model,
            tools=mcp_pydantic_tools, # Provide the tools to the agent
            system_prompt="You are a helpful assistant. Use available tools when necessary, like 'send_email_summary' to send emails."
        )
        logging.info("Pydantic Agent initialized.")
        print("Agent Initialized. Type 'quit' to exit.")

        current_messages: List[ModelMessage] = []

        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() == 'quit':
                break

            # Append user message to the state
            user_message = ModelMessage(parts=[UserPromptPart(content=user_input)])
            current_messages.append(user_message)

            # Invoke the graph
            graph_input = {"messages": current_messages}
            final_state = await app.ainvoke(graph_input)

            # Update message history from the final state
            current_messages = final_state['messages']

            # Print the last assistant message
            last_assistant_message = current_messages[-1]
            if last_assistant_message.parts and isinstance(last_assistant_message.parts[0], TextPart):
                 print(f"Agent: {last_assistant_message.parts[0].content}")
            else:
                 # Handle cases where the last message might be a tool return or something else
                 logging.debug(f"Last message was not simple text: {last_assistant_message}")
                 print("Agent: [Action completed or complex response]")


    except FileNotFoundError as e:
         print(f"Error: {e}")
         logging.error(f"Initialization failed: {e}")
    except ValueError as e:
         print(f"Configuration Error: {e}")
         logging.error(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.exception("An unexpected error occurred during main execution.")
    finally:
        if mcp_instance:
            print("Cleaning up MCP connections...")
            logging.info("Initiating MCPClient cleanup.")
            await mcp_instance.cleanup()
            print("Cleanup finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat agent interrupted by user.")
        # Cleanup should happen in the finally block of main