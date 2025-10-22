from typing import List, Dict, Any
import os
import logging
import uuid
import chromadb
from openai import OpenAI
from utils import pdfSplitTest_Ch, pdfSplitTest_En

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorDBAgent:
    """向量数据库智能体 - 负责PDF文档的向量化存储和检索"""

    def __init__(
            self,
            collection_name: str = "demo001",
            chromadb_directory: str = "chromaDB",
            text_language: str = "Chinese",
            input_pdf: str = "input/健康档案.pdf",
            page_numbers: List[int] = None
    ):
        """
        初始化向量数据库智能体

        Args:
            collection_name: 向量数据库集合名称
            chromadb_directory: 向量数据库存储目录
            text_language: 文本语言 (Chinese/English)
            input_pdf: PDF文件路径
            page_numbers: 要处理的页码列表，None表示全部页码
        """
        self.collection_name = collection_name
        self.chromadb_directory = chromadb_directory
        self.text_language = text_language
        self.input_pdf = input_pdf
        self.page_numbers = page_numbers

        # 阿里通义千问配置
        self.qwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.qwen_embedding_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.qwen_embedding_model = "text-embedding-v1"

        # 初始化向量数据库连接
        self.vector_db = self._initialize_vector_db()

        # 初始化时自动灌库
        self._initialize_documents()

    def _initialize_vector_db(self) -> chromadb.Collection:
        """初始化向量数据库连接"""
        try:
            chroma_client = chromadb.PersistentClient(path=self.chromadb_directory)
            collection = chroma_client.get_or_create_collection(name=self.collection_name)
            logger.info(f"向量数据库初始化成功: {self.collection_name}")
            return collection
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            raise

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        try:
            client = OpenAI(
                base_url=self.qwen_api_base,
                api_key=self.qwen_embedding_api_key
            )
            data = client.embeddings.create(input=texts, model=self.qwen_embedding_model).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.error(f"生成向量时出错: {e}")
            return []

    def _generate_vectors(self, data: List[str], max_batch_size: int = 25) -> List[List[float]]:
        """批量生成向量"""
        results = []
        for i in range(0, len(data), max_batch_size):
            batch = data[i:i + max_batch_size]
            response = self._get_embeddings(batch)
            results.extend(response)
        return results

    def _extract_pdf_paragraphs(self) -> List[str]:
        """提取PDF文本段落"""
        try:
            if self.text_language == 'Chinese':
                paragraphs = pdfSplitTest_Ch.getParagraphs(
                    filename=self.input_pdf,
                    page_numbers=self.page_numbers,
                    min_line_length=1
                )
            elif self.text_language == 'English':
                paragraphs = pdfSplitTest_En.getParagraphs(
                    filename=self.input_pdf,
                    page_numbers=self.page_numbers,
                    min_line_length=1
                )
            else:
                raise ValueError(f"不支持的语言类型: {self.text_language}")

            logger.info(f"成功提取 {len(paragraphs)} 个文本段落")
            return paragraphs
        except Exception as e:
            logger.error(f"PDF文本提取失败: {e}")
            return []

    def _initialize_documents(self):
        """初始化文档到向量数据库"""
        try:
            # 检查集合是否已有文档
            existing_count = self.vector_db.count()
            if existing_count > 0:
                logger.info(f"向量数据库已有 {existing_count} 个文档，跳过初始化")
                return

            # 提取PDF文本
            paragraphs = self._extract_pdf_paragraphs()
            if not paragraphs:
                logger.warning("未提取到任何文本段落")
                return

            # 生成向量并添加到数据库
            embeddings = self._generate_vectors(paragraphs)

            self.vector_db.add(
                embeddings=embeddings,
                documents=paragraphs,
                ids=[str(uuid.uuid4()) for _ in range(len(paragraphs))]
            )

            logger.info(f"成功初始化 {len(paragraphs)} 个文档到向量数据库")

        except Exception as e:
            logger.error(f"文档初始化失败: {e}")
            raise

    def search(self, query: str, top_n: int = 5) -> Dict[str, Any]:
        """
        搜索向量数据库

        Args:
            query: 查询文本
            top_n: 返回最相似的前n个结果

        Returns:
            包含搜索结果和元数据的字典
        """
        try:
            # 生成查询向量
            query_embedding = self._get_embeddings([query])
            if not query_embedding:
                return {"documents": [], "metadatas": [], "distances": []}

            # 执行搜索
            results = self.vector_db.query(
                query_embeddings=query_embedding,
                n_results=top_n,
                include=["documents", "metadatas", "distances"]
            )

            logger.info(f"搜索成功，返回 {len(results['documents'][0])} 个结果")
            return results

        except Exception as e:
            logger.error(f"向量数据库搜索失败: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def get_search_results_formatted(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取格式化的搜索结果"""
        raw_results = self.search(query, top_n)

        formatted_results = []
        for i, (doc, distance) in enumerate(zip(
                raw_results.get('documents', [[]])[0],
                raw_results.get('distances', [[]])[0]
        )):
            formatted_results.append({
                "rank": i + 1,
                "content": doc,
                "similarity_score": 1 - distance,  # 将距离转换为相似度分数
                "metadata": raw_results.get('metadatas', [[]])[0][i] if raw_results.get('metadatas') else {}
            })

        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.vector_db.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "chromadb_directory": self.chromadb_directory
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}


if __name__ == "__main__":
    # 初始化向量数据库智能体
    agent = VectorDBAgent(
        collection_name="health_records",
        chromadb_directory="../chromaDB",
        text_language="Chinese",
        input_pdf="../input/健康档案.pdf"
    )

    # 测试搜索
    results = agent.get_search_results_formatted("张三九的基本信息", top_n=3)
    for result in results:
        print(f"Rank {result['rank']}: {result['content'][:100]}...")
        print(f"Similarity: {result['similarity_score']:.4f}\n")

    # 查看集合信息
    info = agent.get_collection_info()
    print(f"集合信息: {info}")