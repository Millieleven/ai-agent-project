"""
Day 2 改进版：解决两个Badcase
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 改进版System Prompt
IMPROVED_SYSTEM_PROMPT = """你是一个智能助手。

核心规则：
1. 涉及计算的问题，必须调用calculator工具
   - 包括简单运算和特殊情况
   - 示例："10除以0" 必须调用calculator
   - 原因：保证结果的可验证性和一致性

2. 工具返回结果后，严格基于结果回答
   - 工具说"暂无"，你就说"暂无"
   - 不要补充工具结果之外的信息
   - 不要解释为什么会这样

示例对比：

❌ 错误示例1：
用户："火星天气"
工具返回："暂无火星的天气数据"
你的回答："暂无火星的天气数据。火星是太阳系第四颗行星..."
↑ 不要补充额外知识！

✅ 正确示例1：
用户："火星天气"
工具返回："暂无火星的天气数据"
你的回答："暂无火星的天气数据"

❌ 错误示例2：
用户："10除以0"
你的想法："我知道答案，直接说"
你的回答："数学上10除以0是未定义的"
↑ 应该调用calculator！

✅ 正确示例2：
用户："10除以0"
你的行动：调用calculator("10/0")
工具返回："除数不能为0"
你的回答："除数不能为0，无法计算"
"""

# 改进版工具定义
tools_improved = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": """执行数学计算。

⚠️ 重要：所有涉及数值运算的问题都必须通过此工具，包括：
- 简单运算（如2+3）
- 复杂运算（如123*456）
- 特殊情况（如除以0、开方负数等）

不要凭记忆或常识直接回答计算结果。""",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python可执行的数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气。返回结果后，不要补充额外的地理或气象知识。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


def calculator(expression: str) -> dict:
    """计算器"""
    try:
        result = eval(expression)
        return {
            "success": True,
            "result": str(result),
            "error": None
        }
    except ZeroDivisionError:
        return {
            "success": False,
            "result": None,
            "error": "除数不能为0"
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": f"计算错误: {str(e)}"
        }


def get_weather(city: str) -> dict:
    """天气查询"""
    weather_db = {
        "北京": {"temp": 25, "condition": "晴天"},
        "上海": {"temp": 22, "condition": "多云"}
    }

    if city in weather_db:
        data = weather_db[city]
        return {
            "success": True,
            "result": f"{city}天气：{data['condition']}，温度{data['temp']}度",
            "error": None
        }
    else:
        return {
            "success": False,
            "result": None,
            "error": f"暂无{city}的天气数据"
        }


TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather
}


def test_improved():
    """测试改进效果"""

    test_cases = [
        ("计算10除以0", "calculator", ["除数", "不能", "0"]),
        ("火星的天气", "get_weather", ["暂无"])
    ]

    for question, expected_tool, keywords in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试：{question}")
        print('=' * 60)

        messages = [{"role": "user", "content": question}]

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                         {"role": "system", "content": IMPROVED_SYSTEM_PROMPT}
                     ] + messages,
            tools=tools_improved,
            tool_choice="auto",
            temperature=0.3  # 降低温度，减少发挥
        )

        response_msg = response.choices[0].message

        if response_msg.tool_calls:
            print(f"✅ 调用了工具: {response_msg.tool_calls[0].function.name}")

            # 执行工具
            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                result = TOOLS[func_name](**func_args)
                print(f"工具返回: {result}")

                messages.append(response_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            # 生成最终回答
            final = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                             {"role": "system", "content": IMPROVED_SYSTEM_PROMPT}
                         ] + messages,
                temperature=0.3
            )

            answer = final.choices[0].message.content
            print(f"\nAI回答: {answer}")

            # 检查关键词
            missing = [k for k in keywords if k not in answer]
            if missing:
                print(f"❌ 缺少关键词: {missing}")
            else:
                print(f"✅ 包含所有关键词")

            # 检查是否过度发挥
            if len(answer) > 100:
                print(f"⚠️  回答过长({len(answer)}字)，可能过度发挥")
        else:
            print(f"❌ 没有调用工具")
            print(f"直接回答: {response_msg.content}")


if __name__ == "__main__":
    test_improved()