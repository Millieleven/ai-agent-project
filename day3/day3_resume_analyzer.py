"""
Day 3 核心项目：AI简历分析助手

功能：
1. 上传简历文本
2. 上传岗位JD
3. AI分析匹配度
4. 给出优化建议
5. 生成面试问题

技术栈：
- RAG（匹配简历和JD）
- Chroma（向量存储）
- Function Calling（结构化输出）

面试亮点：
"我做了一个简历分析助手，核心创新点是用RAG技术将简历和JD都向量化，
通过相似度匹配找出简历中与岗位最相关和最不相关的部分，
然后让AI针对性地给出优化建议。相比直接让AI分析整份简历，
这种方法更精准，能指出具体哪一段需要调整。"
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


# ==================== Embedding Function ====================

class SiliconFlowEmbedding:
    def __call__(self, input: list) -> list:
        embeddings = []
        for text in input:
            response = client.embeddings.create(
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

# ==================== 简历分析器 ====================

class ResumeAnalyzer:
    """
    简历分析器

    设计思路：
    1. 将简历按section切分（教育、经历、技能等）
    2. 将JD按要求切分（职责、要求等）
    3. 计算每个简历section和JD的匹配度
    4. 基于匹配度给出针对性建议
    """

    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.embedding_fn = SiliconFlowEmbedding()

        # 创建两个collection
        self.resume_collection = self.chroma_client.get_or_create_collection(
            name="resume",
            embedding_function=self.embedding_fn
        )

        self.jd_collection = self.chroma_client.get_or_create_collection(
            name="job_description",
            embedding_function=self.embedding_fn
        )

        print("✅ 简历分析器初始化完成\n")

    def load_resume(self, resume_text: str):
        """
        加载简历

        简历格式示例：
        # 基本信息
        张三 | Python开发工程师 | 3年经验

        # 工作经历
        2021-2024：XX公司 Python工程师
        ...

        # 项目经验
        项目1：...

        # 技能
        Python、Django、MySQL...
        """
        print("📄 加载简历...")

        # 按section切分
        sections = self._parse_resume(resume_text)

        # 清空旧数据
        try:
            self.chroma_client.delete_collection("resume")
            self.resume_collection = self.chroma_client.create_collection(
                name="resume",
                embedding_function=self.embedding_fn
            )
        except:
            pass

        # 存储每个section
        for i, section in enumerate(sections):
            self.resume_collection.add(
                documents=[section['content']],
                metadatas=[{"type": section['type']}],
                ids=[f"resume_section_{i}"]
            )

        print(f"✅ 简历已加载，共{len(sections)}个部分\n")

        return sections

    def load_jd(self, jd_text: str):
        """加载岗位JD"""
        print("📋 加载岗位JD...")

        # 按section切分
        sections = self._parse_jd(jd_text)

        # 清空旧数据
        try:
            self.chroma_client.delete_collection("job_description")
            self.jd_collection = self.chroma_client.create_collection(
                name="job_description",
                embedding_function=self.embedding_fn
            )
        except:
            pass

        # 存储
        for i, section in enumerate(sections):
            self.jd_collection.add(
                documents=[section['content']],
                metadatas=[{"type": section['type']}],
                ids=[f"jd_section_{i}"]
            )

        print(f"✅ JD已加载，共{len(sections)}个部分\n")

        return sections

    def _parse_resume(self, text: str) -> list:
        """
        解析简历

        简单版本：按#分割
        生产版本：需要更复杂的解析逻辑
        """
        sections = []
        lines = text.strip().split('\n')

        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 如果是标题行（以#开头）
            if line.startswith('#'):
                # 保存上一个section
                if current_section and current_content:
                    sections.append({
                        "type": current_section,
                        "content": '\n'.join(current_content)
                    })

                # 开始新section
                current_section = line.replace('#', '').strip()
                current_content = []
            else:
                current_content.append(line)

        # 保存最后一个section
        if current_section and current_content:
            sections.append({
                "type": current_section,
                "content": '\n'.join(current_content)
            })

        return sections

    def _parse_jd(self, text: str) -> list:
        """解析JD"""
        return self._parse_resume(text)  # 相同的解析逻辑

    def analyze(self) -> dict:
        """
        分析匹配度

        流程：
        1. 获取所有JD要求
        2. 对每个要求，在简历中检索最匹配的内容
        3. 计算匹配分数
        4. 识别优势和劣势
        """
        print("=" * 60)
        print("🔍 开始分析...")
        print("=" * 60)

        # 获取JD中的所有要求
        jd_sections = self.jd_collection.get()

        if not jd_sections['documents']:
            return {"error": "请先加载JD"}

        analysis_results = []

        for i, jd_doc in enumerate(jd_sections['documents']):
            jd_type = jd_sections['metadatas'][i]['type']

            print(f"\n分析 JD要求：《{jd_type}》")

            # 在简历中检索匹配内容
            results = self.resume_collection.query(
                query_texts=[jd_doc],
                n_results=2
            )

            if results['documents'][0]:
                best_match = results['documents'][0][0]
                match_type = results['metadatas'][0][0]['type']

                # 计算匹配度（这里简化，实际应该用距离）
                # Chroma返回的results没有直接给距离，我们用AI判断

                analysis_results.append({
                    "jd_section": jd_type,
                    "jd_content": jd_doc[:100] + "...",
                    "matched_resume_section": match_type,
                    "matched_content": best_match[:100] + "..."
                })

                print(f"   ✓ 匹配到简历部分：《{match_type}》")

        # 调用AI进行深度分析
        analysis_text = self._generate_analysis(analysis_results)

        return {
            "matching_details": analysis_results,
            "analysis": analysis_text
        }

    def _generate_analysis(self, matching_results: list) -> str:
        """调用AI生成分析报告"""

        # 构建prompt
        prompt = f"""你是一个资深的招聘顾问，请分析以下简历与岗位的匹配情况。

匹配详情：
{json.dumps(matching_results, ensure_ascii=False, indent=2)}

请提供：
1. 匹配度评分（1-10分）
2. 核心优势（3条）
3. 明显短板（3条）
4. 优化建议（5条具体建议）
5. 可能的面试问题（3个）

要求：
- 具体、可操作
- 不要泛泛而谈
- 基于实际匹配情况"""

        print("\n🤖 AI正在生成深度分析...")

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    def optimize_section(self, section_name: str) -> str:
        """针对某个section给出优化建议"""

        # 获取简历中的这个section
        results = self.resume_collection.get(
            where={"type": section_name}
        )

        if not results['documents']:
            return f"简历中没有找到《{section_name}》部分"

        section_content = results['documents'][0]

        # 获取JD要求
        jd_docs = self.jd_collection.get()
        jd_text = '\n'.join(jd_docs['documents'])

        prompt = f"""你是简历优化专家。

岗位要求：
{jd_text}

简历中的《{section_name}》部分：
{section_content}

请给出具体的优化建议：
1. 这部分目前的问题（3个）
2. 应该添加的内容（3条）
3. 应该删除/淡化的内容（2条）
4. 优化后的示例（重写这一段）

要求：
- 针对岗位要求
- 具体可操作
- 展示改前改后对比"""

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content


# ==================== 交互界面 ====================

def main():
    """主程序"""

    print("=" * 60)
    print("🎯 AI简历分析助手")
    print("=" * 60)
    print()

    analyzer = ResumeAnalyzer()

    # 示例简历
    sample_resume = """# 基本信息
张三 | Python后端开发 | 3年经验 | 本科
邮箱：zhangsan@example.com | 电话：138****8888

# 工作经历
2021.06 - 2024.01：某互联网公司 | Python开发工程师
- 负责电商平台后端API开发
- 使用Django框架开发RESTful API
- 参与数据库设计和优化
- 日常bug修复和功能迭代

# 项目经验
项目一：电商订单系统（2022.03-2022.10）
- 独立负责订单模块的设计和开发
- 使用Redis实现订单缓存，提升查询性能30%
- 使用Celery处理异步任务
- 项目用户量10万+

# 技能清单
编程语言：Python、SQL
框架：Django、Flask
数据库：MySQL、Redis
工具：Git、Docker

# 教育背景
2017-2021：某大学 | 计算机科学与技术 | 本科"""

    # 示例JD
    sample_jd = """# 岗位职责
1. 负责AI Agent应用的后端开发
2. 设计和实现RAG系统
3. 集成大语言模型API
4. 优化系统性能和稳定性

# 任职要求
1. 3年以上Python开发经验
2. 熟悉LangChain、LlamaIndex等AI框架
3. 了解向量数据库（Chroma、Pinecone等）
4. 有大模型应用开发经验优先
5. 熟悉FastAPI、异步编程

# 技能要求
- Python精通
- AI框架使用经验
- 向量数据库
- API设计
- 性能优化"""

    print("使用示例数据进行演示...\n")

    # 加载数据
    analyzer.load_resume(sample_resume)
    analyzer.load_jd(sample_jd)

    # 分析
    result = analyzer.analyze()

    print("\n" + "=" * 60)
    print("📊 分析报告")
    print("=" * 60)
    print(result['analysis'])

    # 针对性优化
    print("\n" + "=" * 60)
    print("💡 针对《项目经验》的优化建议")
    print("=" * 60)

    optimization = analyzer.optimize_section("项目经验")
    print(optimization)

    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()