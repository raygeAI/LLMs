# -*- coding: utf-8 -*-
import neo4j
from tqdm import tqdm
from typing import Any, Dict, List, Set, Optional, Text
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs.networkx_graph import parse_triples
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA

KG_MODEL = "llama3.1:70b"
QA_MODEL = "qwen2:latest"

KG_TRIPLE_DELIMITER = "<|>"
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property."
    "if the extracted text is in Chinese, the triples will be in Chinese. "
    f"If the extracted text is in English, the triples will be in English. the triples  in the following {KG_TRIPLE_DELIMITER}-separated format\n\n"
    "EXAMPLE\n"
    "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to 10"
    "keywords or entity from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "if the extracted text is in Chinese, the keywords will be in Chinese. If the extracted text is in English, the keywords will be in English. \n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)

DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template=DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL
)


# Neo4jGraphStore 图存储库
class Neo4jGraphStore:
    def __init__(
            self,
            username: str,
            password: str,
            url: str,
            database: str = "neo4j",
            node_label: str = "Entity",
    ) -> None:
        self.node_label = node_label
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))

        self._database = database
        self.upsert_session = self._driver.session(database=self._database)
        # Verify connection
        try:
            with self._driver as driver:
                driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )

        # Create constraint for faster insert and retrieval
        try:  # Using Neo4j 5
            self.query(
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:%s) REQUIRE n.id IS UNIQUE;
                """
                % (self.node_label)
            )
        except Exception:  # Using Neo4j <5
            self.query(
                """
                CREATE CONSTRAINT IF NOT EXISTS ON (n:%s) ASSERT n.id IS UNIQUE;
                """
                % (self.node_label)
            )

    @property
    def client(self) -> Any:
        return self._driver

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r]->(n2:%s)
            WHERE n1.id  contains($subj)
            RETURN  n1.id, type(r), n2.id;
        """
        prepared_statement = query % (self.node_label, self.node_label)
        with self._driver.session(database=self._database) as session:
            data = session.run(prepared_statement, {"subj": subj})
            return [record.values() for record in data]

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        query = """
            MERGE (n1:`%s` {id:$subj})
            MERGE (n2:`%s` {id:$obj})
            MERGE (n1)-[:`%s`]->(n2)
        """
        prepared_statement = query % (
            self.node_label,
            self.node_label,
            rel.replace(" ", "_").upper(),
        )

        self.upsert_session.run(prepared_statement, {"subj": subj, "obj": obj})
        # with self._driver.session(database=self._database) as session:
        #     session.run(prepared_statement, {"subj": subj, "obj": obj})

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [d.data() for d in result]


# VectorGraphRAG 混合检索 VectorRAG、GraphRAG
class VectorGraphRAG:
    def __init__(self, is_init: bool = True):
        self.is_init = is_init
        self.query_keyword_extract_template = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
        self.extract_knowledge_template = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
        self.qa_template = PromptTemplate(
            input_variables=["question", "context"],
            template="Answer the {question} according to the context:\n {context} \n Answer:",
        )
        # 初始化语言模型
        self.kg_llm = Ollama(model=KG_MODEL)
        self.qa_llm = Ollama(model=QA_MODEL)
        self.embedding = OllamaEmbeddings(model="bge-m3:latest")
        self.neo4j_graph = Neo4jGraphStore(
            url="bolt://localhost:7687",
            username="neo4j",
            password="20191121gr",
            database="neo4j",
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        self.chroma = Chroma(persist_directory="./data/chroma", embedding_function=self.embedding)
        # 初始化构建vector index， graph index;
        if self.is_init:
            self.build_index()

    def build_index(
            self,
            data_prefix: str = "./data/docs/",
            docs: List[str] = ["text/dossen.txt", "pdf/DeepSeekMath.pdf"]):
        documents = DirectoryLoader(path=data_prefix, glob=docs).load()
        chunked_documents = self.text_splitter.split_documents(documents)
        for doc in tqdm(chunked_documents):
            self.extract_knowledge_graph(doc.page_content)
        # 构建向量知识库文档
        self.chroma.add_documents(
            documents=chunked_documents,  # 以分块的文档
            embedding=self.embedding,
            show_progress=True,
        )
        self.chroma.persist()

    # 抽取知识图谱
    def extract_knowledge_graph(self, text: str):
        triples = []
        chain = LLMChain(llm=self.kg_llm, prompt=self.extract_knowledge_template)
        output = chain.predict(text=text)  # 使用LLM链对文本进行预测
        knowledge = parse_triples(output)  # 解析预测输出得到的三元组
        for triple in knowledge:  # 遍历所有的三元组
            triples.append(triple)  # 将三元组添加到列表中
            # 写入neo4j
            if self.is_init:
                self.neo4j_graph.upsert_triplet(triple.subject, triple.predicate, triple.object_)
        self.upsert_session.close()
        return triples

    def _get_keywords(self, query_str: str) -> List[str]:
        chain = LLMChain(llm=self.qa_llm, prompt=self.query_keyword_extract_template)
        response = chain.predict(question=query_str)
        keywords = self.extract_keywords_given_response(response, start_token="KEYWORDS:", lowercase=False)
        return list(keywords)

    def extract_keywords_given_response(
            self,
            response: str,
            lowercase: bool = True,
            start_token: str = ""
    ) -> Set[str]:
        results = []
        response = response.strip()
        if response.startswith(start_token):
            response = response[len(start_token):]

        keywords = response.split(",")
        for k in keywords:
            rk = k.strip('"')
            if lowercase:
                rk = rk.lower()
            results.append(rk.strip())
        return list(set(results))

    # graph_query 图检索
    def graph_query(self, text: str) -> str:
        graph_context = ""
        keywords = self._get_keywords(text)
        for keyword in keywords:
            triples = self.neo4j_graph.get(keyword)
            if len(triples) > 0:
                for triple in triples:
                    graph_context += " -> ".join(triple) + "\n"

        chain = LLMChain(llm=self.qa_llm, prompt=self.qa_template)
        output = chain.predict(question=text, context=graph_context)
        return output

    # vector_query 向量检索
    def vector_query(self, text: str):
        qa = RetrievalQA.from_chain_type(llm=self.qa_llm, chain_type="stuff", retriever=self.chroma.as_retriever())
        output = qa.run(text)
        return output

    def query(self, text: Text):
        graph_context = self.graph_query(text)
        vector_context = self.vector_query(text)
        context = graph_context + "\n" + vector_context
        chain = LLMChain(llm=self.qa_llm, prompt=self.qa_template)
        output = chain.predict(question=text, context=context)
        return output


# match (n) detach delete n 删除所有数据
if __name__ == "__main__":
    graphrag = VectorGraphRAG(is_init=False)
    out = graphrag.vector_query("DeepSeekMath 成功的关键有哪些？")
    print(out)
