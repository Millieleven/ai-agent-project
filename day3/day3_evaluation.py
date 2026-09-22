"""
Day 3 评测：RAG系统性能测试

测试维度：
1. 检索准确率：能否找到正确的文档
2. 回答质量：基于文档的回答是否准确
3. 幻觉率：是否编造文档中没有的信息
4. 拒答率：文档中没有时是否能正确拒答
"""

from day3_rag_with_chroma import RAGSystem
import json


def evaluate_rag():
    """评测RAG系统"""

    # 准备测试数据
    test_documents = [
        {
            "title": "Python基础",
            "content": "Python由Guido van Rossum于1991年创建。Python强调代码可读性。"
        },
        {
            "title": "机器学习",
            "content": "机器学习分为监督学习、无监督学习、强化学习三类。常用库有scikit-learn。"
        }
    ]

    test_cases = [
        {
            "question": "Python是谁创建的？",
            "expected_answer": "Guido van Rossum",
            "should_find": True
        },
        {
            "question": "机器学习有哪些类型？",
            "expected_keywords": ["监督", "无监督", "强化"],
            "should_find": True
        },
        {
            "question": "Java是谁发明的？",
            "expected_behavior": "refuse",  # 应该拒答
            "should_find": False
        }
    ]

    # 初始化RAG
    rag = RAGSystem("eval_test")
    rag.add_documents(test_documents)

    results = []

    for case in test_cases:
        print(f"\n测试：{case['question']}")

        answer = rag.query(case['question'], top_k=2)

        # 评估
        passed = True

        if case.get("expected_answer"):
            if case["expected_answer"] not in answer:
                passed = False
                print(f"❌ 缺少预期答案")

        if case.get("expected_keywords"):
            missing = [k for k in case["expected_keywords"] if k not in answer]
            if missing:
                passed = False
                print(f"❌ 缺少关键词: {missing}")

        if case.get("expected_behavior") == "refuse":
            refuse_keywords = ["没有", "找不到", "不知道", "无法"]
            if not any(k in answer for k in refuse_keywords):
                passed = False
                print(f"❌ 应该拒答但给出了答案")

        if passed:
            print("✅ 通过")

        results.append({
            "question": case["question"],
            "answer": answer,
            "passed": passed
        })

    # 统计
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    print(f"\n评测结果：{passed_count}/{total} 通过")

    return results


if __name__ == "__main__":
    evaluate_rag()