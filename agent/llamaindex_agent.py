# -*- coding: utf-8 -*-
import sys
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.llama_pack.base import BaseLlamaPack
from typing import List, Dict, Optional
import chromadb
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.postprocessor import SimilarityPostprocessor


HOTEL_COLLECTION_NAME = "hotel"
COMMON_COLLECTION_NAME = "common"
DATA_DIR = "E:\\work\\LLMs\\agent\\data\\"
API_KEY = "sk-013c0ff07bd84d14a7db70f77e090f23"
MODEL = "llama3.1:70b"


# Define a simple Python function
def pow(a: float, b: float) -> float:
    """a to the exponent b and return the result."""
    return a ** b


def query_member_points(member_id: str) -> str:
    """依据 member_id, 查询会员积分, 返回会员积分信息 """
    return f"会员 {member_id} 的积分 50"


# Agent 从RAG 到 Agent
class Agent:
    def __init__(self, chroma_db: str = DATA_DIR + "chroma_db", is_local_llm: bool = True, is_init_load: bool = False):
        self.is_local_llm = is_local_llm
        self.chroma_db = chroma_db
        self.is_init_load = is_init_load
        self.llm = self._init_llm()
        # 设置llama_index的模型
        self._llama_index_model_setting()
        self.hotel_collection = self._init_chroma_db(collection_name=HOTEL_COLLECTION_NAME)
        self.common_collection = self._init_chroma_db(collection_name=COMMON_COLLECTION_NAME)
        self.react_agent = self._init_agent()

    # _init_llm 初始化 llm
    def _init_llm(self):
        if self.is_local_llm:
            llm = Ollama(model=MODEL, request_timeout=10240.0)
        else:
            llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=API_KEY)
        return llm

    # _llama_index_model_setting  设置 llama_index 的框架模型
    def _llama_index_model_setting(self):
        Settings.embed_model = OllamaEmbedding(
            model_name="quentinz/bge-large-zh-v1.5:latest",
            base_url="http://localhost:11434",
            # ollama_additional_kwargs={"mirostat": 0},
        )
        Settings.llm = self.llm

    # _init_chroma_db 初始化 chroma_db
    def _init_chroma_db(self, collection_name: str):
        db = chromadb.PersistentClient(path=self.chroma_db)
        collection = db.get_or_create_collection(name=collection_name)
        return collection

    # create_auto_query_engine 创建自动多租户的rag 查询引擎
    def create_auto_query_engine(self, input_files: List[str], meta_data: List[Dict[str, str]]):
        assert len(input_files) == len(meta_data), "input files and meta data must be the same length"
        if not self.is_init_load:
            vector_store = ChromaVectorStore(chroma_collection=self.hotel_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=Settings.embed_model,
            )
        else:
            meta_dicts = dict(zip(input_files, meta_data))
            meta_func = lambda file: meta_dicts[file]
            docs = SimpleDirectoryReader(input_files=input_files, file_metadata=meta_func).load_data()
            vector_store = ChromaVectorStore(chroma_collection=self.hotel_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        vector_store_info = VectorStoreInfo(
            content_info="酒店基础设施信息",
            metadata_info=[
                MetadataInfo(
                    name="hotel_id",
                    type="str",
                    description=("各个对应酒店基础设施介绍，取值为 ['0010002', '0010003'] 中某一个"),
                ),
            ],
        )
        vector_auto_retriever = VectorIndexAutoRetriever(index=index, vector_store_info=vector_store_info)
        response_synthesizer = get_response_synthesizer()
        auto_query_engine = RetrieverQueryEngine(
            retriever=vector_auto_retriever,
            response_synthesizer=response_synthesizer,
            # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )
        return auto_query_engine

    def create_query_engine(self, input_files: List[str], meta_data: List[Dict[str, str]]):
        assert len(input_files) == len(meta_data), "input files and meta data must be the same length"
        if not self.is_init_load:
            vector_store = ChromaVectorStore(chroma_collection=self.common_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=Settings.embed_model,
            )
        else:
            meta_dicts = dict(zip(input_files, meta_data))
            meta_func = lambda file: meta_dicts[file]
            docs = SimpleDirectoryReader(input_files=input_files, file_metadata=meta_func).load_data()
            vector_store = ChromaVectorStore(chroma_collection=self.common_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        engine = index.as_query_engine(similarity_top_k=5)
        return engine

    # 创建agent 工具
    def _create_tools(self):
        auto_query_engine = self.create_auto_query_engine(
            input_files=[DATA_DIR + "hotel_0010002.txt", DATA_DIR + "hotel_0010003.txt"],
            meta_data=[{"hotel_id": "0010002"}, {"hotel_id": "0010003"}]
        )
        engine = self.create_query_engine(
            input_files=[DATA_DIR + "携程调价模型数据分析_V2.pdf", DATA_DIR + "DeepSeekMath.pdf"],
            meta_data=[{"content_type": "携程调价结果分析"}, {"content_type": "DeepSeekMath"}]
        )
        pow_tool = FunctionTool.from_defaults(fn=pow)
        query_member_points_tool = FunctionTool.from_defaults(fn=query_member_points)
        tools = [
            QueryEngineTool(
                query_engine=auto_query_engine,
                metadata=ToolMetadata(name="hotel_id", description=("各个酒店基础设施文档介绍")),
            ),
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(name="content_type", description=("DeepSeek 论文和携程调价模型结果分析数据文档")),
            ),
            pow_tool,
            query_member_points_tool,
        ]
        return tools

    def _init_agent(self):
        tools = self._create_tools()
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            max_iterations=10,
            verbose=True,
            context="你是一个酒店智能客服，必要情况下可以调用工具回答问题，在工具没能解决的情况下，你可以依据自身模型进行思考回答"
        )
        return agent

    # query 查询
    def query(self, question: str):
        return self.react_agent.query(question)

    # chat 聊天
    def stream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
        return self.react_agent.stream_chat(message, chat_history)

    def reset(self):
        return self.react_agent.reset()


class Capturing(list):
    """To capture the stdout from ReActAgent.chat with verbose=True. Taken from
    https://stackoverflow.com/questions/16571150/
    how-to-capture-stdout-output-from-a-python-function-call.
    """

    def __enter__(self) -> Any:
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args) -> None:
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class GradioAgentServer:
    """Gradio chatbot to chat with a ReActAgent ."""

    def __init__(self) -> None:
        """Init params."""
        try:
            from ansi2html import Ansi2HTMLConverter
        except ImportError:
            raise ImportError("Please install ansi2html via `pip install ansi2html`")

        self.agent = Agent()
        self.conv = Ansi2HTMLConverter()

    def _handle_user_message(self, user_message, history):
        """Handle the user submitted message. Clear message box, and append
        to the history.
        """
        return "", [*history, (user_message, "")]

    def _generate_response(
            self, chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Generate the response from agent, and capture the stdout of the ReActAgent's thoughts."""
        with Capturing() as output:
            response = self.agent.stream_chat(chat_history[-1][0])
        ansi = "\n========\n".join(output)
        html_output = self.conv.convert(ansi)
        for token in response.response_gen:
            if token.find("Answer: ") > 0:
                token = token.split("Answer: ")[1]
            chat_history[-1][1] += token
            yield chat_history, str(html_output)

    def _reset_chat(self) -> Tuple[str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        # clear agent history
        self.agent.reset()
        return "", "", ""  # clear textboxes

    def run(self) -> Any:
        """Run the pipeline."""
        import gradio as gr

        demo = gr.Blocks(
            theme=gr.themes.Default(),
            css="#box { height: 420px; overflow-y: scroll !important}",
        )
        with demo:
            gr.Markdown(
                "# AI ReActAgent Powered by LlamaIndex 🦙\n"
                "- This Application  is powered by LlamaIndex's `ReActAgent` with Qwen-Max \n"
                "- If  Qwen-Max cloud model is not available , Llama3.1:70b will serve for you \n"
            )
            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Message History",
                    scale=3,
                )
                console = gr.HTML(elem_id="box")
            with gr.Row():
                message = gr.Textbox(label="Your question", scale=5)
                clear = gr.ClearButton()

            message.submit(
                self._handle_user_message,
                [message, chat_window],
                [message, chat_window],
                queue=False,
            ).then(
                self._generate_response,
                chat_window,
                [chat_window, console],
            )
            clear.click(self._reset_chat, None, [message, chat_window, console])

        demo.launch(show_error=True)


if __name__ == "__main__":
    # # 让Agent完成任务
    # agent = Agent()
    # y = agent.query("计算3 的 0.5 次方")
    # y = agent.query("member_id x100_y, 我的会员积分是多少？")
    # y = agent.query("介绍一下广州")
    # y = agent.query("DeepSeekMath 成功的关键因素有哪些？")
    # y = agent.query("hotel_id 0010003, wifi 密码是多少？")
    # print(y)
    GradioAgentServer().run()


