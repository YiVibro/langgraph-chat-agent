# from typing import Annotated,get_type_hints,get_args
# import json
# from typing_extensions import TypedDict
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
# from langchain_tavily import TavilySearch
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.types import Command, interrupt
# from langchain_core.tools import tool

# memory = InMemorySaver()

# load_dotenv()


# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]

# class State(TypedDict):
#     messages:Annotated[list,add_messages]

# class BasicToolNode:
#     """A node that runs the tools requested in the last AIMessage."""

#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, inputs: dict):
#         if messages := inputs.get("messages", []):
#             message = messages[-1]
#         else:
#             raise ValueError("No message found in input")
#         outputs = []
#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                 tool_call["args"]
#             )
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}


# graph_builder = StateGraph(State)

# tool = TavilySearch(max_results=2)
# tools = [tool,human_assistance]

# tool_node = BasicToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)

# #adding  a node chatbot

# import os 
# from langchain_google_genai import ChatGoogleGenerativeAI
# import getpass

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=os.environ["GOOGLE_API_KEY"],
# )

# #Tell the llm which tools it can call
# llm_with_tools=llm.bind_tools(tools)

# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}

# def route_tools(
#     state: State,
# ):
#     """
#     Use in the conditional_edge to route to the ToolNode if the last message
#     has tool calls. Otherwise, route to the end.
#     """
#     if isinstance(state, list):
#         ai_message = state[-1]
#     elif messages := state.get("messages", []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(f"No messages found in input state to tool_edge: {state}")
#     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
#         return "tools"
#     return END


# # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# # it is fine directly responding. This conditional routing defines the main agent loop.
# graph_builder.add_conditional_edges(
#     "chatbot",
#     route_tools,
#     # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
#     # It defaults to the identity function, but if you
#     # want to use a node named something else apart from "tools",
#     # You can update the value of the dictionary to something else
#     # e.g., "tools": "my_tools"
#     {"tools": "tools", END: END},
# )
# # Any time a tool is called, we return to the chatbot to decide the next step
# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")
# graph = graph_builder.compile(checkpointer=memory)


# # graph_builder.add_node("chatbot", chatbot)

# # graph_builder.add_edge(START,"chatbot")

# # graph_builder.add_edge("chatbot",END)

# # graph=graph_builder.compile()

# # from IPython.display import Image,display

# # try:
# #     display(Image(graph.get_graph().draw_mermaid_png()))
# # except Exception as e:
# #     print("Error displaying graph:", str(e))    
# config = {
#     "thread_id": "my-conversation-thread-001"
# }
# def stream_graph_updates(user_input: str):
#     system_prompt = SystemMessage(
#     content="You are a helpful assistant that answers any user queries"
#         )
#     user_message = HumanMessage(content=user_input)
#     for event in graph.stream({"messages": [system_prompt, user_message]},config=config):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)


# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

import os
import json
import getpass
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

# --- Setup ---
load_dotenv()
memory = InMemorySaver()

# --- Tool Definition ---
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# --- State Definition ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Tool Node ---
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in input")

        last_message = messages[-1]
        outputs = []

        for tool_call in last_message.tool_calls:
            tool = self.tools_by_name.get(tool_call["name"])
            if not tool:
                raise ValueError(f"Tool '{tool_call['name']}' not found")

            result = tool.invoke(tool_call["args"])
            outputs.append(ToolMessage(
                content=json.dumps(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))

        return {"messages": outputs}

# --- LLM Setup ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.environ["GOOGLE_API_KEY"],
)

# --- Tools and LLM ---
search_tool = TavilySearch(max_results=2)
tools = [search_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

# --- Nodes ---
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(state: State):
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages found in state")

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    return "tools" if tool_calls else END

# --- Build the Graph ---
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", BasicToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph = graph_builder.compile(checkpointer=memory)

# --- Run Function ---
def stream_graph_updates(user_input: str):
    system_prompt = SystemMessage(content="You are a helpful assistant that answers any user queries")
    user_message = HumanMessage(content=user_input)

    config = {"thread_id": "my-thread-001"}  # required for checkpointing
    for event in graph.stream({"messages": [system_prompt, user_message]}, config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# --- CLI Loop ---
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
