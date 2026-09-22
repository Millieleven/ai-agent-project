"""
Day 3: 使用Chroma的真实RAG系统

功能：
1. 从文本文件加载知识库
2. 自动切分、向量化、存储
3. 支持多轮对话
4. 展示检索到的相关文档

面试时怎么讲：
"我用Chroma做了一个RAG系统，支持上传自定义文档，
自动切分成chunk，存入向量数据库，
用户提问时检索top-k相关片段，然后让AI基于这些片段回答。
我还做了chunk size的实验，发现300-500字符效果最好。"
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


# ==================== 配置 ====================

# Chroma需要一个embedding function
class SiliconFlowEmbeddingFunction:
    def __init__(self):
        self.client = client

    def __call__(self, input: list) -> list:
        """Chroma会调用这个方法 - 用于批量向量化"""
        embeddings = []
        for text in input:
            response = self.client.embeddings.create(
                model="BAAI/bge-large-zh-v1.5",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, input: list) -> list:
        """用于查询时的向量化"""
        return self.__call__(input)

    def name(self) -> str:
        """返回embedding function的名称"""
        return "siliconflow-bge-large-zh"
# ==================== RAG类 ====================

class RAGSystem:
    """
    RAG系统封装

    工程化思维：
    1. 初始化时创建collection
    2. add_documents()批量添加文档
    3. query()检索+生成
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        # 初始化Chroma客户端（本地模式）
        self.chroma_client = chromadb.Client()

        # 创建或获取collection（类似数据库的表）
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SiliconFlowEmbeddingFunction()
        )

        print(f"✅ RAG系统初始化完成")
        print(f"   Collection: {collection_name}")

    def add_documents(self, documents: list, chunk_size: int = 300):
        """
        添加文档到知识库

        参数：
            documents: 文档列表，每个元素是{"title": "", "content": ""}
            chunk_size: 切分大小
        """
        print(f"\n📚 开始添加文档...")

        all_chunks = []
        all_metadatas = []
        all_ids = []

        chunk_id = 0

        for doc in documents:
            title = doc.get("title", "未命名")
            content = doc.get("content", "")

            # 切分文档
            chunks = self._chunk_text(content, chunk_size)

            print(f"   - {title}: {len(chunks)} 个chunk")

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "title": title,
                    "chunk_index": i,
                    "source": doc.get("source", "unknown")
                })
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        # 批量添加到Chroma
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )

        print(f"✅ 添加完成，共 {len(all_chunks)} 个chunk")

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> list:
        """文本切分"""
        chunks = []
        text = text.strip()

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += (chunk_size - overlap)

        return chunks

    def query(self, question: str, top_k: int = 3) -> str:
        """
        RAG查询

        参数：
            question: 用户问题
            top_k: 返回最相关的k个chunk
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 用户问题：{question}")
        print('=' * 60)

        # 步骤1：检索
        print(f"\n1️⃣  检索相关文档（top {top_k}）...")

        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )

        if not results['documents'][0]:
            return "知识库中没有找到相关内容"

        # 显示检索结果
        print("\n   检索到的相关内容：")
        retrieved_docs = []
        for i, (doc, metadata) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
        ), 1):
            print(f"\n   [{i}] 来源：《{metadata['title']}》")
            print(f"       内容：{doc[:100]}...")
            retrieved_docs.append(doc)

        # 步骤2：构建增强prompt
        print("\n2️⃣  构建增强prompt...")

        context = "\n\n".join([
            f"文档{i + 1}:\n{doc}"
            for i, doc in enumerate(retrieved_docs)
        ])

        enhanced_prompt = f"""请基于以下文档回答问题。

参考文档：
{context}

用户问题：{question}

要求：
1. 只基于上述文档回答
2. 如果文档中没有相关信息，明确说"文档中没有提到"
3. 引用文档时可以说"根据文档X"
4. 不要编造信息"""

        # 步骤3：调用AI
        print("\n3️⃣  调用AI生成回答...")

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        print(f"\n4️⃣  AI回答：")
        print(f"\n{answer}\n")

        return answer

    def clear(self):
        """清空知识库"""
        self.chroma_client.delete_collection(self.collection.name)
        print("✅ 知识库已清空")


# ==================== 测试 ====================

def test_rag():
    """测试RAG系统"""

    # 准备测试文档
    documents = [
        {
            "title": "Python简介",
            "content": """Python是一种高级编程语言，由Guido van Rossum于1991年创建。
            Python的设计哲学强调代码的可读性和简洁性。
            Python使用缩进来表示代码块，而不是使用大括号。
            Python支持多种编程范式，包括面向对象编程、命令式编程和函数式编程。
            Python拥有丰富的标准库和第三方库，如NumPy、Pandas、Django、Flask等。
            Python广泛应用于Web开发、数据科学、机器学习、自动化等领域。""",
            "source": "编程语言文档"
        },
        {
            "title": "机器学习基础",
            "content": """机器学习是人工智能的一个分支，让计算机通过数据学习规律。
            机器学习主要分为三类：监督学习、无监督学习和强化学习。
            监督学习使用标注数据训练模型，如分类和回归任务。
            无监督学习从无标注数据中发现模式，如聚类和降维。
            强化学习通过与环境交互学习最优策略。
            常用的机器学习算法包括线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络等。
            Python是机器学习最流行的编程语言，主要库有scikit-learn、TensorFlow、PyTorch。""",
            "source": "AI技术文档"
        },
        {
            "title": "RAG技术详解",
            "content": """RAG（Retrieval-Augmented Generation）是检索增强生成技术。
            RAG的核心思想是将信息检索和文本生成结合起来。
            RAG的工作流程：首先从知识库中检索相关文档，然后将检索结果作为上下文提供给大语言模型生成回答。
            RAG的优势：减少模型幻觉，提供可验证的答案，支持私有知识库。
            RAG的关键技术：文本切分（chunking）、向量化（embedding）、向量数据库、相似度检索。
            常用的向量数据库包括Chroma、Pinecone、Milvus、Weaviate等。
            RAG特别适合企业知识管理、客服问答、文档助手等场景。""",
            "source": "AI技术文档"
        }
    ]

    # 初始化RAG系统
    rag = RAGSystem(collection_name="test_kb")

    # 添加文档
    rag.add_documents(documents, chunk_size=200)

    # 测试查询
    print("\n" + "=" * 60)
    print("开始测试RAG查询")
    print("=" * 60)

    # 测试1：文档中有明确答案
    rag.query("Python是谁创建的？")

    # 测试2：需要综合信息
    rag.query("Python在机器学习中的应用")

    # 测试3：文档中没有的信息
    rag.query("Java的特点是什么？")

    # 测试4：RAG相关
    rag.query("什么是RAG？它解决什么问题？")


if __name__ == "__main__":
    test_rag()