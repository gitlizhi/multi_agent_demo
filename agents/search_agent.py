from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from config import Config
from langchain_community.chat_models import ChatZhipuAI
from langchain_tavily import TavilySearch       # https://www.tavily.com/ 申请API Key
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


class SearchAgent:
    """基于TavilySearch的搜索Agent"""
    def __init__(self):
        self.llm = ChatZhipuAI(**Config.get_llm_config())
        self.memory = MemorySaver()
        # 构建图
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        self.save_graph_visualization()

    def chatbot(self, state: State):
        tavily_search = TavilySearch(max_results=2)
        llm_with_tools = self.llm.bind_tools([tavily_search])
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def _build_graph(self) -> StateGraph:
        tavily_search = TavilySearch(max_results=2)     # 设置每次最多搜2条
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        tool_node = ToolNode([tavily_search])
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges(
            "chatbot",
            self.route_tools,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        return graph_builder

    def route_tools(self, state: State):
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    def call(self, user_input: str, config: dict):
        input_message = {"messages": [HumanMessage(content=user_input)]}
        result = self.app.invoke(input_message, config)
        print(f'TavilySearch搜索结果：\n{result}')
        return result

    def save_graph_visualization(self, filename: str = "graph_search.png") -> None:
        """保存状态图的可视化表示。"""
        try:
            with open(filename, "wb") as f:
                f.write(self.app.get_graph().draw_mermaid_png())
        except Exception as e:
            print(f"Failed to save graph visualization: {e}")