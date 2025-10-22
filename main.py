from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from config import Config
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from agents.search_agent import SearchAgent
import uuid
from datetime import datetime
from agents.db_search_agent import VectorDBAgent
from langgraph.prebuilt import create_react_agent


class State(TypedDict):
    messages: Annotated[list, add_messages]


class MultiAgentMapSystem:
    """多智能体调用，此类为总协调，根据用户输入来区分调用不同的智能体工作。"""
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=Config.BASE_URL,
            api_key=Config.DASHSCOPE_API_KEY,
            model=Config.MODEL_NAME,
            temperature=Config.MODEL_TEMPERATURE,
            timeout=30,  # 添加超时配置（秒）
            max_retries=2  # 添加重试次数
        )
        # self.llm = ChatZhipuAI(**Config.get_llm_config())
        self.memory = MemorySaver()
        # 初始化向量数据库智能体
        self.vector_agent = VectorDBAgent()
        # 提前初始化分类提示模板
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能专家。你的任务是：
                        1. 分析用户的需求，针对用户的提问来区分属于哪一类问题。
                        2. 必须在给定的答案中选择其一进行回答，不要回答其他内容，使用中文回答，只回答选项，针对无法区分的问题选择其他。
                        3. 选项如下：数据库检索，web搜索，聊天
                        """),
            ("human", "{input}")
        ])
        # 构建图
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        self.save_graph_visualization()

    def _build_graph(self) -> StateGraph:
        """构建LangGraph"""
        # 构建图
        graph_builder = StateGraph(State)

        # 添加节点
        graph_builder.add_node("classifier", self.classifier_node)
        graph_builder.add_node("search", self.search_handler)
        graph_builder.add_node("vector_search", self.vector_search_handler)
        graph_builder.add_node("chat", self.chat_handler)

        # 设置入口点
        graph_builder.set_entry_point("classifier")

        # 添加条件边
        graph_builder.add_conditional_edges(
            "classifier",
            self.route_classification,  # 使用路由方法
            {
                "vector_search": "vector_search",
                "search": "search",
                "chat": "chat",
            }
        )

        # 所有处理节点都指向结束
        graph_builder.add_edge("chat", END)
        graph_builder.add_edge("vector_search", END)
        graph_builder.add_edge("search", END)

        return graph_builder

    def classifier_node(self, state: State) -> State:
        """分类节点 - 只是传递状态，不做实际分类"""
        return state

    def classifierByAI(self, user_input: str) -> str:
        """使用AI进行问题分类"""
        chain = self.classifier_prompt | self.llm
        response = chain.invoke({"input": user_input})
        print(f'智谱AI分类问题为：{response.content}')
        return response.content

    def route_classification(self, state: State) -> Literal["vector_search", "search", "chat"]:
        """根据查询内容分类并路由"""
        messages = state['messages']
        if not messages:
            return "chat"
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_input = last_message.content
        else:
            user_input = str(last_message)

        choose = self.classifierByAI(user_input)
        if '数据库检索' in choose:
            return 'vector_search'
        elif 'web搜索' in choose:
            return 'search'
        else:
            return 'chat'

    def search_handler(self, state: State) -> State:
        print("正在调用web搜索智能体...")
        messages = state['messages']
        last_message = messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        thread_id = self.generate_improved_thread_id()
        result = SearchAgent().call(user_input, {"configurable": {"thread_id": thread_id, "user_id": "uid1"}})
        return result

    def vector_search_handler(self, state: State) -> State:
        print("正在调用向量数据库搜索智能体...")
        messages = state['messages']
        last_message = messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # 搜索向量数据库
        search_results = self.vector_agent.get_search_results_formatted(user_input, top_n=1)

        # 基于检索结果生成回答
        if search_results:
            context = "\n".join([f"{i + 1}. {result['content']}" for i, result in enumerate(search_results)])
            prompt = f"""基于以下从向量数据库中检索到的信息回答用户问题：

                    检索到的相关信息：
                    {context}
                
                    用户问题：{user_input}
                
                    请根据上述信息给出准确回答："""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        else:
            response_content = "未找到相关信息，请尝试其他查询词。"

        # 将回答添加到消息列表
        from langchain_core.messages import AIMessage
        state['messages'].append(AIMessage(content=response_content))

        return state

    def chat_handler(self, state: State) -> State:
        print("正在调用chat智能体...")
        agent = create_react_agent(self.llm, [])
        # 传递完整的消息历史，而不是只传递最后一条
        result = agent.invoke(state)
        return result

    def save_graph_visualization(self, filename: str = "graph.png") -> None:
        """保存状态图的可视化表示。"""
        try:
            with open(filename, "wb") as f:
                f.write(self.app.get_graph().draw_mermaid_png())
        except Exception as e:
            print(f"Failed to save graph visualization: {e}")

    def generate_improved_thread_id(self, user_input="", user_id=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        if user_id:
            base_id = f"user_{user_id}"
        elif user_input:
            # 从输入中提取一些特征作为标识
            input_hash = hash(user_input[:20]) % 10000
            base_id = f"input_{abs(input_hash)}"
        else:
            base_id = f"session_{uuid.uuid4().hex[:8]}"

        return f"{base_id}_{timestamp}"


def main():
    sys = MultiAgentMapSystem()
    thread_id = sys.generate_improved_thread_id("", user_id="uid1")
    conf = {"configurable": {"thread_id": thread_id, "user_id": "uid1"}}
    print("开始对话（输入'退出','exit' 或 'quit' 结束聊天）:")
    while True:
        userInput = input('用户：')
        if userInput.lower() in ['退出', 'exit', 'quit']:
            break

        input_message = {"messages": [HumanMessage(content=userInput)]}
        try:
            result = sys.app.invoke(input_message, config=conf)
            # 打印结果
            if 'messages' in result and result['messages']:
                # 找到最后一条AI消息
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'type') and msg.type == 'ai' and hasattr(msg, 'content'):
                        print(f"助手: {msg.content}")
                        break
                else:
                    # 如果没有找到AI消息，打印最后一条消息
                    last_msg = result['messages'][-1]
                    if hasattr(last_msg, 'content'):
                        print(f"助手: {last_msg.content}")
                    else:
                        print(f"助手: {last_msg}")
            else:
                print(f"处理结果: {result}")

        except Exception as e:
            print(f"错误: {e}")
            import traceback

            traceback.print_exc()


if __name__ == '__main__':
    main()