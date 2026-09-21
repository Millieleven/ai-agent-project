"""
Function Calling 核心原理演示

什么是Function Calling？
→ 让AI能够"调用工具"
→ AI不直接回答，而是告诉你"我需要调用XX工具，参数是YY"
→ 你执行工具，把结果返回给AI
→ AI根据工具结果给出最终回答

这解决了什么问题？
→ AI不知道实时数据（天气、股票）
→ AI不能做精确计算
→ AI不能访问你的数据库、API

面试必问：Function Calling的流程是什么？
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


# 定义工具：计算器
def calculator(expression: str) -> str:
    """
    执行数学计算

    这是一个真实的Python函数
    AI不会执行它，AI只会告诉你"调用这个函数，参数是XX"
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


# 定义工具：获取天气（模拟）
def get_weather(city: str) -> str:
    """
    获取城市天气

    实际项目中这里应该调用真实天气API
    现在用模拟数据
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，25度",
        "上海": "多云，22度",
        "深圳": "小雨，28度"
    }
    return weather_data.get(city, f"{city}的天气数据暂无")


# 关键：告诉AI有哪些工具可用
# 这个格式是OpenAI规定的
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，输入数学表达式，返回计算结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如: '2+3' 或 '10*5'"
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
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如: '北京' 或 '上海'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


def chat_with_tools(user_question: str):
    """
    带工具调用的对话流程

    面试重点：这个流程要能讲清楚
    """
    print(f"\n用户问题：{user_question}")
    print("=" * 50)

    # 第一步：把问题和工具列表发给AI
    messages = [{"role": "user", "content": user_question}]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
        tools=tools,  # 关键：告诉AI有哪些工具
        tool_choice="auto"  # 让AI自己决定要不要调用工具
    )

    response_message = response.choices[0].message

    # 第二步：检查AI是否要调用工具
    if response_message.tool_calls:
        print("\n🔧 AI决定调用工具：")

        # 把AI的回复加入对话历史
        messages.append(response_message)

        # 第三步：执行AI要求的工具调用
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  函数名: {function_name}")
            print(f"  参数: {function_args}")

            # 根据函数名执行对应的函数
            if function_name == "calculator":
                function_response = calculator(function_args["expression"])
            elif function_name == "get_weather":
                function_response = get_weather(function_args["city"])
            else:
                function_response = "未知函数"

            print(f"  工具返回: {function_response}")

            # 第四步：把工具执行结果告诉AI
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response
            })

        # 第五步：AI根据工具结果给出最终回答
        print("\n🤖 AI根据工具结果生成最终回答：")
        final_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages
        )

        final_answer = final_response.choices[0].message.content
        print(f"\n最终回答：{final_answer}")

    else:
        # AI认为不需要调用工具，直接回答
        print("\n💬 AI直接回答（未调用工具）：")
        print(response_message.content)


# 测试
if __name__ == "__main__":
    # 测试1：需要计算器
    chat_with_tools("帮我计算 123 乘以 456 等于多少？")

    # 测试2：需要天气工具
    chat_with_tools("北京今天天气怎么样？")

    # 测试3：不需要工具
    chat_with_tools("什么是Python？")