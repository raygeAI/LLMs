# -*- coding: utf-8 -*-
import kuzu
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.kuzu import KuzuGraphStore
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

from llama_index.core import StorageContext
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import SimpleDirectoryReader
from pyvis.network import Network
from llama_index.core import PropertyGraphIndex

API_KEY = "sk-013c0ff07bd84d14a7db70f77e090f23"
MODEL = "llama3.1:latest"
DATA_DIR = "E:\\work\\LLMs\\GraphRag\\data\\"


class GraphRAG:
    def __init__(self, is_local_llm: bool = True, local_graph_store: str = DATA_DIR + "graph_store"):
        self.is_local_llm = is_local_llm
        self.local_graph_store = local_graph_store
        self.llm = self._init_llm()
        self._llama_index_model_setting()
        self.graph_store = self._init_graph_db()
        self.index = self.build_knowledge_graph()
        self.engine = self._init_engine()

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
            model_name="bge-m3:latest",
            base_url="http://localhost:11434",
        )
        Settings.llm = self.llm
        Settings.chunk_size = 512

    def _init_graph_db(self):
        db = kuzu.Database(self.local_graph_store)
        graph_store = KuzuGraphStore(db)
        return graph_store

    def build_knowledge_graph(self):
        docs = SimpleDirectoryReader(input_files=[DATA_DIR + "dossen.txt"]).load_data()
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        index = KnowledgeGraphIndex.from_documents(
            documents=docs,
            show_progress=True,
            max_triplets_per_chunk=15,
            storage_context=storage_context,
            include_embeddings=True,
        )
        storage_context.persist(self.local_graph_store)
        return index

    def _init_engine(self):
        return self.index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,
            verbose=True,
        )

    # 可视化知识图谱
    def visualize_graph(self):
        g = self.index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show(DATA_DIR + "vis.html")

    def query(self, text: str):
        return self.engine.query(text)


if __name__ == "__main__":
    graph_rag = GraphRAG(is_local_llm=False)
    result = graph_rag.query("东呈集团价值观是什么？")
    print(result)
    graph_rag.visualize_graph()
