"""
Day 3: RAG核心原理

RAG = Retrieval Augmented Generation
检索增强生成

5个步骤（面试必问）：
1. 文档加载（Load）
2. 文本切分（Chunk）
3. 向量化（Embed）
4. 存储（Store）
5. 检索+生成（Retrieve & Generate）
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# ==================== 步骤1：准备文档 ====================

# 模拟知识库文档
KNOWLEDGE_BASE = [
    {
        "id": 1,
        "title": "Python基础",
        "content": """Python是一种高级编程语言，由Guido van Rossum于1991年创建。
        Python强调代码可读性，使用缩进来定义代码块。
        Python支持多种编程范式，包括面向对象、命令式、函数式编程。
        常用的Python库包括NumPy、Pandas、Django、Flask等。"""
    },
    {
        "id": 2,
        "title": "机器学习简介",
        "content": """机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。
        主要分为三类：监督学习、无监督学习、强化学习。
        常用算法包括线性回归、决策树、神经网络等。
        Python是机器学习最常用的编程语言，主要库有scikit-learn、TensorFlow、PyTorch。"""
    },
    {
        "id": 3,
        "title": "RAG技术",
        "content": """RAG（检索增强生成）是一种结合信息检索和文本生成的技术。
        RAG先从知识库中检索相关内容，然后将检索结果作为上下文提供给大语言模型。
        这种方法可以减少模型幻觉，提供更准确和可验证的回答。
        RAG特别适合需要基于特定文档回答问题的场景。"""
    },
    {
        "id": 4,
        "title": "向量数据库",
        "content": """向量数据库用于存储和检索向量数据。
        常见的向量数据库包括Chroma、Pinecone、Milvus、Weaviate等。
        向量数据库通过相似度搜索来找到最相关的文档。
        相似度计算常用的方法是余弦相似度和欧氏距离。"""
    }
]


# ==================== 步骤2：文本切分（Chunking）====================

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list:
    """
    将长文本切分成小块

    为什么要切分？
    1. AI的上下文窗口有限
    2. 检索时更精确（只取相关部分）
    3. 成本更低（只传相关内容给AI）

    参数：
        chunk_size: 每块的大小（字符数）
        overlap: 块之间的重叠（避免信息断裂）

    面试要点：
    - chunk_size太小：上下文不完整
    - chunk_size太大：检索不精确，成本高
    - 一般：200-500字符
    """
    chunks = []
    text = text.strip()

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


# 测试切分
print("=" * 60)
print("步骤2：文本切分演示")
print("=" * 60)

test_text = KNOWLEDGE_BASE[0]["content"]
chunks = chunk_text(test_text, chunk_size=80, overlap=20)

print(f"\n原文长度：{len(test_text)} 字符")
print(f"切分后：{len(chunks)} 块\n")

for i, chunk in enumerate(chunks, 1):
    print(f"块{i}：{chunk[:50]}...")


# ==================== 步骤3：向量化（Embedding）====================

def get_embedding(text: str) -> list:
    """
    将文本转换为向量

    什么是向量？
    一串数字，表示文本的"语义"

    例如：
    "Python编程" → [0.1, 0.8, 0.3, ...]
    "编程语言"   → [0.2, 0.7, 0.4, ...]  ← 相似，向量也相似
    "今天天气"   → [0.9, 0.1, 0.2, ...]  ← 不相似，向量差异大

    面试要点：
    - 用embedding模型将文本转成向量
    - 相似的文本，向量也相似
    - 通过计算向量距离来检索相关文档
    """
    try:
        response = client.embeddings.create(
            model="BAAI/bge-large-zh-v1.5",  # 中文向量模型
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"向量化失败: {str(e)}")
        return None


# 测试向量化
print("\n" + "=" * 60)
print("步骤3：向量化演示")
print("=" * 60)

text1 = "Python是一种编程语言"
text2 = "编程语言Python"
text3 = "今天天气很好"

vec1 = get_embedding(text1)
vec2 = get_embedding(text2)
vec3 = get_embedding(text3)

if vec1:
    print(f"\n文本1：{text1}")
    print(f"向量维度：{len(vec1)}")
    print(f"向量前5个值：{vec1[:5]}")


# ==================== 步骤4：计算相似度 ====================

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    计算余弦相似度

    返回值范围：-1 到 1
    - 1：完全相同
    - 0：无关
    - -1：完全相反

    面试要点：
    余弦相似度是最常用的向量相似度计算方法
    """
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# 测试相似度
if vec1 and vec2 and vec3:
    sim_12 = cosine_similarity(vec1, vec2)
    sim_13 = cosine_similarity(vec1, vec3)

    print("\n" + "=" * 60)
    print("步骤4：相似度计算")
    print("=" * 60)
    print(f"\n文本1：{text1}")
    print(f"文本2：{text2}")
    print(f"相似度：{sim_12:.4f}  ← 语义相似，分数高\n")

    print(f"文本1：{text1}")
    print(f"文本3：{text3}")
    print(f"相似度：{sim_13:.4f}  ← 语义不同，分数低")


# ==================== 步骤5：简单的RAG流程 ====================

def simple_rag(question: str):
    """
    简化版RAG流程演示

    完整流程：
    1. 用户提问
    2. 问题向量化
    3. 在知识库中找最相关的文档
    4. 把文档+问题一起给AI
    5. AI基于文档回答
    """
    print("\n" + "=" * 60)
    print(f"用户问题：{question}")
    print("=" * 60)

    # 步骤1：问题向量化
    print("\n1️⃣  问题向量化...")
    question_vec = get_embedding(question)

    if not question_vec:
        return "向量化失败"

    # 步骤2：检索相关文档
    print("2️⃣  检索相关文档...")

    doc_scores = []
    for doc in KNOWLEDGE_BASE:
        doc_vec = get_embedding(doc["content"])
        if doc_vec:
            similarity = cosine_similarity(question_vec, doc_vec)
            doc_scores.append({
                "doc": doc,
                "score": similarity
            })

    # 排序，取最相关的
    doc_scores.sort(key=lambda x: x["score"], reverse=True)
    top_doc = doc_scores[0]

    print(f"\n   最相关文档：《{top_doc['doc']['title']}》")
    print(f"   相似度分数：{top_doc['score']:.4f}")

    # 步骤3：构建prompt
    print("\n3️⃣  构建增强prompt...")

    context = top_doc['doc']['content']

    enhanced_prompt = f"""请基于以下文档回答问题。

文档内容：
{context}

用户问题：{question}

要求：
1. 只基于文档内容回答
2. 如果文档中没有相关信息，说"文档中没有提到"
3. 不要编造信息"""

    print(f"\n   增强后的prompt长度：{len(enhanced_prompt)} 字符")

    # 步骤4：调用AI
    print("\n4️⃣  调用AI生成回答...")

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "user", "content": enhanced_prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    print(f"\n5️⃣  AI回答：")
    print(f"\n{answer}")

    return answer


# ==================== 测试RAG ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG完整流程测试")
    print("=" * 60)

    # 测试1：文档中有答案
    simple_rag("Python是谁创建的？")

    # 测试2：文档中没有答案
    simple_rag("Java是谁发明的？")

    # 测试3：需要综合理解
    simple_rag("什么是RAG？")