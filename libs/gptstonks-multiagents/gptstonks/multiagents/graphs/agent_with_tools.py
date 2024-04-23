import asyncio
from typing import Any, Sequence

from langchain.agents.output_parsers.openai_tools import (
    parse_ai_message_to_openai_tool_action,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models.llms import LLM
from langchain_core.messages import ToolMessage
from langchain_core.runnables.base import RunnableLike
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..states import BaseAgentState


class GraphAgentWithTools(BaseModel):
    """Factory class for Agent with tools using LangGraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(
        description="LangChain LLM to use in the graph. Tools should not be binded yet."
    )
    tools: Sequence[Any] = Field(description="LangChain tools the agent can call")
    prompt_main_agent: Any = Field(
        description="LangChain Prompt object to use with the main agent"
    )

    @computed_field
    @property
    def tool_executor(self) -> ToolExecutor:
        return ToolExecutor(self.tools)

    @computed_field
    @property
    def llm_with_tools(self) -> LLM:
        return self.prompt_main_agent | self.model.bind_tools(self.tools)

    def _should_continue(self, state: BaseAgentState):
        """Function that determines whether to continue or not."""
        last_message = parse_ai_message_to_openai_tool_action(state["context_messages"][-1])
        # If there is no function call, then we finish
        if isinstance(last_message, AgentFinish):
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def _call_model(self, state: BaseAgentState):
        """Function that calls the model."""
        response = await self.llm_with_tools.ainvoke(
            {"input": state["input"], "agent_scratchpad": state["context_messages"]}
        )
        # We return a list, because this will get added to the existing list
        return {"context_messages": [response]}

    async def _call_openai_tool(
        self,
        agent_action: AgentAction,
    ) -> ToolMessage | tuple[ToolMessage, list[dict], str]:
        """Executes a LLM function and returns the response."""
        action = ToolInvocation(
            tool=agent_action.tool,
            tool_input=agent_action.tool_input,
        )
        # We call the tool_executor and get back a response
        response = await self.tool_executor.ainvoke(action)

        return ToolMessage(
            content=str(response), name=action.tool, tool_call_id=agent_action.tool_call_id
        )

    async def _call_tool_with_openai(self, state: BaseAgentState):
        """Function to execute tools."""
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = parse_ai_message_to_openai_tool_action(state["context_messages"][-1])
        new_tool_outputs = await asyncio.gather(
            *[self._call_openai_tool(agent_action) for agent_action in last_message]
        )
        if len(new_tool_outputs) == 0:
            return {"context_messages": ["The tool returned nothing. You must try other tool."]}
        return {
            "context_messages": (
                new_tool_outputs
                if new_tool_outputs
                else ["The tool returned nothing. You must try other tool."]
            ),
        }

    def _define_basic_workflow(
        self, alternative_node: str = END, alternative_node_func: RunnableLike | None = None
    ) -> StateGraph:
        """Defines basic workflow with agent and action nodes.

        The nodes are called:
        - "agent": calls the LLM to generate a response or a tool call.
        - "action": executes the tool when a tool call is generated by "action".

        The graph is as follows:
        agent -> action | alternative_node; action -> agent

        Args:
            alternative_node (`str`): state to go to when no action is called.
            alternative_node_func (`RunnableLike | None`):
                LangGraph node implementation. None if `alternative_node` is left as default.

        Returns:
            `StateGraph`: StateGraph with action and agent nodes included.
        """
        # Define a new graph
        workflow = StateGraph(BaseAgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", self._call_tool_with_openai)

        # We now add a conditional edge
        if alternative_node != END and alternative_node_func is not None:
            workflow.add_node(alternative_node, alternative_node_func)
        elif alternative_node != END:
            raise ValueError(
                "`alternative_node_func` cannot be None when `alternative_node` is not `END`"
            )
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                # If there is a function call, then we call the action node.
                "continue": "action",
                # Otherwise we finish.
                "end": alternative_node,
            },
        )

        workflow.add_edge("action", "agent")
        return workflow

    def define_basic_graph(self) -> CompiledGraph:
        """Defines basic LangGraph graph."""

        workflow = self._define_basic_workflow()
        workflow.set_entry_point("agent")

        return workflow.compile()
