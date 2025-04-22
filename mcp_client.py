from pydantic_ai import RunContext, Tool as PydanticTool
from pydantic_ai.tools import ToolDefinition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool, ToolCallResult
from contextlib import AsyncExitStack
from typing import Any, List, Dict
import asyncio
import logging
import shutil
import json
import os

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - MCPClient - %(levelname)s - %(message)s"
)

class MCPClient:
    """Manages connections to one or more MCP servers based on a config file."""

    def __init__(self) -> None:
        self.servers: Dict[str, MCPServer] = {}
        self.config: dict[str, Any] = {}
        self.pydantic_tools: List[PydanticTool] = []
        self.exit_stack = AsyncExitStack()
        self._is_started = False

    def load_servers(self, config_path: str) -> None:
        """Load server configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file.
        """
        if not os.path.exists(config_path):
             logging.error(f"MCP config file not found at {config_path}")
             raise FileNotFoundError(f"MCP config file not found at {config_path}")

        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)
            logging.info(f"Loaded MCP config from {config_path}")

        if "mcpServers" not in self.config:
             logging.error("MCP config file missing 'mcpServers' key.")
             raise ValueError("Invalid MCP config file format.")

        self.servers = {
            name: MCPServer(name, config, self.exit_stack)
            for name, config in self.config["mcpServers"].items()
        }
        logging.info(f"Configured {len(self.servers)} MCP server(s): {list(self.servers.keys())}")

    async def start(self) -> List[PydanticTool]:
        """Starts each configured MCP server, retrieves their tools,
           and converts them into Pydantic AI compatible tools.

        Returns:
            A list of Pydantic AI Tool instances.

        Raises:
            RuntimeError: If called again after already started without cleanup.
            Exception: If any server fails to initialize.
        """
        if self._is_started:
            raise RuntimeError("MCPClient already started. Call cleanup() before starting again.")

        logging.info("Starting MCP servers...")
        self.pydantic_tools = []
        all_tools = []
        try:
            # Initialize all servers concurrently
            init_tasks = [server.initialize() for server in self.servers.values()]
            await asyncio.gather(*init_tasks)
            logging.info("All MCP servers initialized.")

            # Get tools from all servers concurrently
            tool_tasks = [server.create_pydantic_ai_tools() for server in self.servers.values()]
            results = await asyncio.gather(*tool_tasks)
            for tool_list in results:
                all_tools.extend(tool_list)

            self.pydantic_tools = all_tools
            self._is_started = True
            logging.info(f"Successfully retrieved {len(self.pydantic_tools)} Pydantic AI tools from MCP servers.")
            return self.pydantic_tools
        except Exception as e:
            logging.error(f"Failed to start MCP servers or retrieve tools: {e}")
            # Attempt cleanup if startup failed
            await self.cleanup()
            raise # Re-raise the exception after cleanup attempt

    async def cleanup(self) -> None:
        """Cleans up all MCP server connections and resources."""
        if not self._is_started and not self.exit_stack._exit_callbacks:
             logging.info("MCPClient cleanup called but not started or already cleaned.")
             return

        logging.info("Cleaning up MCPClient resources...")
        try:
            # The exit stack handles cleaning up individual servers started via initialize
            await self.exit_stack.aclose()
            logging.info("AsyncExitStack closed successfully.")
        except Exception as e:
            logging.warning(f"Warning during MCPClient cleanup: {e}")
        finally:
             # Reset state
             self.servers = {}
             self.pydantic_tools = []
             self.exit_stack = AsyncExitStack() # Recreate for potential restart
             self._is_started = False
             logging.info("MCPClient cleanup finished.")


class MCPServer:
    """Represents and manages a connection to a single MCP server."""

    def __init__(self, name: str, config: dict[str, Any], exit_stack: AsyncExitStack) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = exit_stack # Use shared stack from MCPClient
        self._initialized = False
        logging.info(f"MCPServer instance created for '{self.name}'")

    async def initialize(self) -> None:
        """Initializes the connection to the MCP server via stdio."""
        if self._initialized:
            logging.warning(f"Server '{self.name}' already initialized.")
            return

        logging.info(f"Initializing MCP server '{self.name}'...")
        command_name = self.config.get("command")
        if not command_name:
             raise ValueError(f"Missing 'command' in config for server '{self.name}'")

        # Resolve command path if necessary (e.g., for 'npx', 'python')
        command_path = shutil.which(command_name)
        if not command_path:
            # If shutil.which fails, try using the command name directly
            # This might be needed if the command is an alias or in a non-standard path
            logging.warning(f"Could not resolve path for command '{command_name}', using it directly.")
            command_path = command_name
            # Consider adding a check here if os.path.exists(command_path) if needed

        server_params = StdioServerParameters(
            command=command_path,
            args=self.config.get("args", []),
            env=self.config.get("env"), # Pass None if not present
            cwd=self.config.get("cwd") # Add CWD if specified
        )
        logging.info(f"Server '{self.name}' params: command='{server_params.command}', args={server_params.args}, env={'present' if server_params.env else 'not present'}, cwd={server_params.cwd}")

        try:
            # Enter the stdio_client and ClientSession contexts using the shared exit_stack
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            self._initialized = True
            logging.info(f"Successfully initialized MCP server '{self.name}'.")
        except Exception as e:
            logging.error(f"Error initializing server '{self.name}': {e}")
            # No need to call self.cleanup() here, AsyncExitStack handles it on error
            raise # Re-raise to signal failure to MCPClient

    async def create_pydantic_ai_tools(self) -> List[PydanticTool]:
        """Fetches tools from the connected MCP server and converts them to Pydantic AI Tools."""
        if not self.session or not self._initialized:
            raise RuntimeError(f"Server '{self.name}' is not initialized. Call initialize() first.")

        logging.info(f"Fetching tools from MCP server '{self.name}'...")
        try:
            mcp_tools_response = await self.session.list_tools()
            mcp_tools = mcp_tools_response.tools
            logging.info(f"Received {len(mcp_tools)} tools from '{self.name}'.")

            pydantic_tools = []
            for tool in mcp_tools:
                # Example filter: Skip tools without descriptions or specific names if needed
                # if not tool.description or tool.name == 'internal_tool':
                #     logging.info(f"Skipping tool '{tool.name}' from '{self.name}' due to filter.")
                #     continue
                pydantic_tools.append(self._create_tool_instance(tool))
            logging.info(f"Converted {len(pydantic_tools)} tools from '{self.name}' to Pydantic AI format.")
            return pydantic_tools
        except Exception as e:
            logging.error(f"Error fetching or converting tools from server '{self.name}': {e}")
            raise

    def _create_tool_instance(self, tool: MCPTool) -> PydanticTool:
        """Creates a Pydantic AI Tool instance that wraps an MCP tool call."""
        if not self.session:
             # This should ideally not happen if called after successful initialize
             raise RuntimeError(f"Session not available for tool '{tool.name}' on server '{self.name}'.")

        # Capture session for the closure
        current_session = self.session

        async def execute_tool(**kwargs: Any) -> Any:
            """Dynamically created function to execute a specific MCP tool."""
            logging.info(f"Executing MCP tool '{tool.name}' on server '{self.name}' with args: {kwargs}")
            try:
                result: ToolCallResult = await current_session.call_tool(tool.name, arguments=kwargs)
                # Process result - often the text content is needed
                # Handle potential variations in ToolCallResult structure
                if result.content and result.content[0] and hasattr(result.content[0], 'text'):
                     tool_output = result.content[0].text
                     logging.info(f"Tool '{tool.name}' executed successfully. Output: {tool_output[:100]}...") # Log truncated output
                     return tool_output # Return the primary text content
                else:
                     logging.warning(f"Tool '{tool.name}' result structure unexpected or missing text content.")
                     return str(result.content) # Fallback to string representation
            except Exception as e:
                logging.error(f"Error executing MCP tool '{tool.name}': {e}")
                # Return error information to the agent
                return f"Error executing tool '{tool.name}': {str(e)}"

        async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
            """Dynamically created function to prepare the tool definition."""
            # Set the JSON schema for input parameters
            tool_def.parameters_json_schema = tool.inputSchema
            logging.debug(f"Prepared tool '{tool.name}' with schema: {tool.inputSchema}")
            return tool_def

        # Create the Pydantic AI Tool
        return PydanticTool(
            func=execute_tool, # The async function that calls the MCP tool
            name=tool.name,
            description=tool.description or f"Executes the '{tool.name}' tool via MCP.", # Provide default description
            takes_ctx=False, # This tool wrapper doesn't need the RunContext directly
            prepare=prepare_tool # Function to set the input schema
        )

    # Note: No explicit cleanup method needed here anymore,
    # as the shared AsyncExitStack in MCPClient handles resource release.