"""
工具调用的异常处理

面试重点：
- 工具调用失败怎么办？
- 工具返回格式错误怎么办？
- 工具执行超时怎么办？
- AI调用了不存在的工具怎么办？
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


def calculator(expression: str) -> dict:
    """
    改进版计算器
    返回格式统一：{"success": bool, "result": str, "error": str}
    """
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


def get_weather_api(city: str) -> dict:
    """
    模拟调用天气API
    可能失败：API超时、城市不存在、接口限流
    """
    # 模拟API延迟
    time.sleep(0.5)

    # 模拟不同情况
    if city == "火星":
        return {
            "success": False,
            "result": None,
            "error": "城市不存在"
        }

    # 模拟API限流
    if city == "限流测试":
        return {
            "success": False,
            "result": None,
            "error": "API请求过于频繁，请稍后重试"
        }

    # 正常情况
    weather_data = {
        "北京": "晴天，25度",
        "上海": "多云，22度"
    }

    return {
        "success": True,
        "result": weather_data.get(city, f"{city}天气数据暂无"),
        "error": None
    }


# 工具注册表（方便管理）
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather_api
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式"
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
            "description": "获取城市天气",
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


def execute_tool_safely(function_name: str, function_args: dict) -> str:
    """
    安全执行工具调用

    工程化思维：
    1. 检查工具是否存在
    2. 捕获所有异常
    3. 返回统一格式
    4. 记录日志（生产环境需要）
    """
    print(f"\n  → 准备执行工具: {function_name}")
    print(f"  → 参数: {function_args}")

    # 检查1：工具是否存在
    if function_name not in TOOL_FUNCTIONS:
        error_msg = f"工具 '{function_name}' 不存在"
        print(f"  ❌ {error_msg}")
        return json.dumps({
            "success": False,
            "error": error_msg
        }, ensure_ascii=False)

    try:
        # 检查2：参数是否完整
        function = TOOL_FUNCTIONS[function_name]

        # 执行工具
        result = function(**function_args)

        # 检查3：返回格式是否正确
        if not isinstance(result, dict) or "success" not in result:
            return json.dumps({
                "success": False,
                "error": "工具返回格式错误"
            }, ensure_ascii=False)

        if result["success"]:
            print(f"  ✅ 执行成功: {result['result']}")
        else:
            print(f"  ❌ 执行失败: {result['error']}")

        return json.dumps(result, ensure_ascii=False)

    except TypeError as e:
        # 参数错误
        error_msg = f"参数错误: {str(e)}"
        print(f"  ❌ {error_msg}")
        return json.dumps({
            "success": False,
            "error": error_msg
        }, ensure_ascii=False)

    except Exception as e:
        # 未知错误
        error_msg = f"未知错误: {str(e)}"
        print(f"  ❌ {error_msg}")
        return json.dumps({
            "success": False,
            "error": error_msg
        }, ensure_ascii=False)


def chat_with_robust_tools(user_question: str):
    """
    健壮的工具调用流程
    """
    print(f"\n{'=' * 50}")
    print(f"用户问题：{user_question}")
    print('=' * 50)

    messages = [{"role": "user", "content": user_question}]

    # 最多重试3次
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"\n第 {attempt} 次尝试...")

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=10.0  # 设置超时
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                print("\n🔧 AI要调用工具：")
                messages.append(response_message)

                all_tools_success = True

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # 安全执行工具
                    function_response = execute_tool_safely(
                        function_name,
                        function_args
                    )

                    # 检查工具执行结果
                    result_dict = json.loads(function_response)
                    if not result_dict.get("success"):
                        all_tools_success = False

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })

                # 如果所有工具都失败，且还有重试机会
                if not all_tools_success and attempt < max_attempts:
                    print("\n⚠️  工具执行失败，准备重试...")
                    continue

                # 生成最终回答
                print("\n🤖 生成最终回答...")
                final_response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=messages
                )

                print(f"\n最终回答：{final_response.choices[0].message.content}")
                return

            else:
                # 不需要工具
                print(f"\n💬 直接回答：{response_message.content}")
                return

        except Exception as e:
            print(f"\n❌ API调用失败: {str(e)}")
            if attempt < max_attempts:
                print("准备重试...")
                time.sleep(1)
            else:
                print("已达最大重试次数，放弃")
                return


# 测试各种异常情况
if __name__ == "__main__":
    # 测试1：正常调用
    chat_with_robust_tools("计算 100 + 200")

    # 测试2：除以0错误
    chat_with_robust_tools("计算 10 除以 0")

    # 测试3：城市不存在
    chat_with_robust_tools("火星的天气怎么样？")

    # 测试4：API限流
    chat_with_robust_tools("限流测试的天气")