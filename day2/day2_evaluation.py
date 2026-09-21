"""
Day 2 评测系统

工程化思维：
1. 测试工具调用是否正确
2. 测试异常处理是否到位
3. 记录所有badcase
4. 为明天的优化提供数据
"""

import json
from datetime import datetime
from day2_smart_assistant import SmartAssistant

# 测试用例
TEST_CASES = [
    {
        "id": 1,
        "name": "正常计算",
        "question": "帮我算一下 25 乘以 8",
        "expected_tool": "calculator",
        "expected_keywords": ["200"],
        "should_call_tool": True
    },
    {
        "id": 2,
        "name": "除以0",
        "question": "10除以0等于多少？",
        "expected_tool": "calculator",
        "expected_keywords": ["不能", "错误", "无法"],
        "should_call_tool": True
    },
    {
        "id": 3,
        "name": "天气查询",
        "question": "北京今天天气怎么样？",
        "expected_tool": "get_weather",
        "expected_keywords": ["北京", "度"],
        "should_call_tool": True
    },
    {
        "id": 4,
        "name": "不存在的城市",
        "question": "火星的天气",
        "expected_tool": "get_weather",
        "expected_keywords": ["暂无", "数据", "无法"],
        "should_call_tool": True
    },
    {
        "id": 5,
        "name": "搜索查询",
        "question": "什么是LangChain？",
        "expected_tool": "search_web",
        "expected_keywords": ["LangChain", "框架"],
        "should_call_tool": True
    },
    {
        "id": 6,
        "name": "常识问答（不需要工具）",
        "question": "Python是什么？",
        "expected_tool": None,
        "expected_keywords": ["编程", "语言"],
        "should_call_tool": False
    },
    {
        "id": 7,
        "name": "多步骤任务",
        "question": "先告诉我北京的天气，然后计算100加200",
        "expected_tool": ["get_weather", "calculator"],
        "expected_keywords": ["北京", "300"],
        "should_call_tool": True
    }
]


def run_evaluation():
    """运行评测"""
    print("=" * 70)
    print("🧪 Day 2 评测系统")
    print("=" * 70)

    results = []
    badcases = []

    assistant = SmartAssistant(max_history=10)

    for case in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"测试 #{case['id']}: {case['name']}")
        print(f"问题: {case['question']}")
        print('=' * 70)

        # 记录初始状态
        initial_tool_count = assistant.tool_call_count

        # 执行测试
        try:
            answer = assistant.chat(case['question'])

            # 检查是否调用了工具
            tool_called = assistant.tool_call_count > initial_tool_count

            # 评估结果
            passed = True
            fail_reasons = []

            # 检查1：是否正确调用了工具
            if case['should_call_tool'] and not tool_called:
                passed = False
                fail_reasons.append("应该调用工具但没有调用")

            if not case['should_call_tool'] and tool_called:
                passed = False
                fail_reasons.append("不应该调用工具但调用了")

            # 检查2：关键词是否出现
            missing_keywords = []
            for keyword in case['expected_keywords']:
                if keyword not in answer:
                    missing_keywords.append(keyword)

            if missing_keywords:
                passed = False
                fail_reasons.append(f"缺少关键词: {missing_keywords}")

            # 记录结果
            result = {
                "id": case['id'],
                "name": case['name'],
                "question": case['question'],
                "answer": answer,
                "tool_called": tool_called,
                "passed": passed,
                "fail_reasons": fail_reasons,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

            if passed:
                print(f"\n✅ 通过")
            else:
                print(f"\n❌ 失败: {fail_reasons}")
                badcases.append(result)

        except Exception as e:
            print(f"\n❌ 测试异常: {str(e)}")
            badcases.append({
                "id": case['id'],
                "name": case['name'],
                "question": case['question'],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # 统计
    total = len(results)
    passed = sum(1 for r in results if r['passed'])

    print(f"\n{'=' * 70}")
    print(f"📊 评测结果")
    print(f"{'=' * 70}")
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed / total * 100:.1f}%")

    # 保存badcase
    if badcases:
        print(f"\n❌ 发现 {len(badcases)} 个Badcase")

        with open("day2_badcases.json", "w", encoding="utf-8") as f:
            json.dump(badcases, f, ensure_ascii=False, indent=2)

        print("\nBadcase详情：")
        for bc in badcases:
            print(f"\n#{bc['id']} - {bc['name']}")
            print(f"问题: {bc['question']}")
            if 'fail_reasons' in bc:
                print(f"失败原因: {bc['fail_reasons']}")
            if 'error' in bc:
                print(f"错误: {bc['error']}")

        print(f"\nBadcase已保存到: day2_badcases.json")
    else:
        print("\n🎉 没有发现Badcase！")

    # 保存完整结果
    with open("day2_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完整结果已保存到: day2_evaluation_results.json")

    return results, badcases


if __name__ == "__main__":
    run_evaluation()