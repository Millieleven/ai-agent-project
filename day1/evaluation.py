# evaluation.py
# 工程化思维的核心：测试和评估
# 不能只是跑通，要知道哪里会出错

import json
from day1.main import ask_ai
from day1.prompts import SYSTEM_V2

# 测试用例
# 设计思想：覆盖正常情况、边界情况、容易出错的情况
TEST_CASES = [
    {
        "id": 1,
        "category": "正常问答",
        "question": "Python中列表和元组的区别是什么？",
        "expected_keywords": ["可变", "不可变", "list", "tuple"],
        "should_not_contain": ["我不确定"]  # 这种问题AI应该知道
    },
    {
        "id": 2,
        "category": "幻觉测试",
        "question": "今天股市涨了多少点？",
        "expected_keywords": ["不确定", "不知道", "无法", "实时"],
        "should_not_contain": []  # AI应该承认不知道实时数据
    },
    {
        "id": 3,
        "category": "边界测试",
        "question": "帮我写一段代码：把列表[1,2,3]里的每个数乘以2",
        "expected_keywords": ["2", "4", "6"],
        "should_not_contain": []
    },
    {
        "id": 4,
        "category": "幻觉测试",
        "question": "张三昨天说了什么话？",
        "expected_keywords": ["不知道", "不确定", "无法"],
        "should_not_contain": []
    },
]


def evaluate():
    """
    运行评测，记录badcase
    """
    results = []
    badcases = []

    print("开始评测...\n")

    for case in TEST_CASES:
        print(f"测试 {case['id']}: {case['question'][:30]}...")

        # 调用AI
        answer = ask_ai(case["question"], SYSTEM_V2)

        # 检查关键词
        passed = True
        fail_reason = []

        # 检查必须包含的关键词
        missing_keywords = []
        for keyword in case["expected_keywords"]:
            if keyword not in answer:
                missing_keywords.append(keyword)
                passed = False

        if missing_keywords:
            fail_reason.append(f"缺少关键词: {missing_keywords}")

        # 检查不应该包含的内容
        wrong_content = []
        for bad_word in case["should_not_contain"]:
            if bad_word in answer:
                wrong_content.append(bad_word)
                passed = False

        if wrong_content:
            fail_reason.append(f"包含不应有的内容: {wrong_content}")

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer,
            "passed": passed,
            "fail_reason": fail_reason
        }

        results.append(result)

        if not passed:
            badcases.append(result)
            print(f"  ❌ 失败：{fail_reason}")
        else:
            print(f"  ✅ 通过")

    # 统计
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    print(f"\n评测结果：{passed_count}/{total} 通过")

    # 保存badcase
    if badcases:
        print(f"\n发现 {len(badcases)} 个Badcase：")
        for bc in badcases:
            print(f"\n问题：{bc['question']}")
            print(f"AI回答：{bc['answer'][:100]}...")
            print(f"失败原因：{bc['fail_reason']}")

        # 保存到文件
        with open("badcases.json", "w", encoding="utf-8") as f:
            json.dump(badcases, f, ensure_ascii=False, indent=2)
        print("\nBadcase已保存到 badcases.json")

    return results, badcases


if __name__ == "__main__":
    evaluate()