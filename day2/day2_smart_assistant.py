"""
Day 2 核心项目：智能助手

功能：
1. 多轮对话 + 记忆
2. 工具调用（计算、天气、搜索）
3. 异常处理
4. 对话历史管理（防止token爆炸）

面试时怎么讲这个项目：
"我做了一个带工具调用的Agent，解决了三个核心问题：
1. 工具调用失败的兜底策略（重试+降级）
2. 对话历史管理（避免token超限）
3. 错误信息对用户友好化"
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from datetime import datetime

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


# ==================== 工具函数 ====================

def calculator(expression: str) -> dict:
    """计算器工具"""
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
    """天气查询工具（模拟）"""
    # 实际项目中这里调用真实天气API
    time.sleep(0.3)  # 模拟API延迟

    weather_db = {
        "北京": {"temp": 25, "condition": "晴天", "humidity": "45%"},
        "上海": {"temp": 22, "condition": "多云", "humidity": "60%"},
        "深圳": {"temp": 28, "condition": "小雨", "humidity": "75%"},
        "广州": {"temp": 27, "condition": "阴天", "humidity": "70%"}
    }

    if city in weather_db:
        data = weather_db[city]
        return {
            "success": True,
            "result": f"{city}天气：{data['condition']}，温度{data['temp']}度，湿度{data['humidity']}",
            "error": None
        }
    else:
        return {
            "success": False,
            "result": None,
            "error": f"暂无{city}的天气数据"
        }


def search_web(query: str) -> dict:
    """网络搜索工具（模拟）"""
    # 实际项目中这里调用搜索API（如DuckDuckGo、Bing）
    time.sleep(0.5)

    # 模拟搜索结果
    mock_results = {
        "python": "Python是一种高级编程语言，由Guido van Rossum创建，强调代码可读性。",
        "ai agent": "AI Agent是能够感知环境、做出决策并执行任务的智能系统，通常基于大语言模型。",
        "langchain": "LangChain是一个用于开发LLM应用的框架，提供了工具调用、RAG等功能。"
    }

    for key in mock_results:
        if key in query.lower():
            return {
                "success": True,
                "result": mock_results[key],
                "error": None
            }

    return {
        "success": True,
        "result": f"关于'{query}'的搜索结果：这是一个模拟的搜索结果。实际项目中会调用真实搜索API。",
        "error": None
    }


# 工具注册表
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web": search_web
}

# OpenAI格式的工具定义
tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。支持加减乘除、幂运算等。例如：'2+3', '10*5', '2**8'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式"
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
            "description": "查询指定城市的当前天气情况，包括温度、天气状况和湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、深圳"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网络信息，用于回答需要最新信息或外部知识的问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ==================== 核心逻辑 ====================

class SmartAssistant:
    """
    智能助手类

    工程化思维：
    1. 用类封装，方便管理状态
    2. 对话历史单独管理
    3. 错误统一处理
    """

    def __init__(self, max_history: int = 10):
        self.conversation_history = []
        self.max_history = max_history  # 最多保留多少轮对话
        self.tool_call_count = 0  # 统计工具调用次数
        self.error_count = 0  # 统计错误次数

        # System Prompt
        self.system_prompt = """你是一个智能助手。

你的能力：
1. 可以调用工具来获取信息或执行任务
2. 遇到不确定的信息，优先使用工具而不是瞎编
3. 工具调用失败时，如实告知用户

回答规则：
1. 简洁明了，直击重点
2. 工具返回错误时，向用户解释原因
3. 不编造事实
4. 不确定时说"我不确定"或使用搜索工具"""

    def _manage_history(self):
        """
        管理对话历史，防止token爆炸

        策略：
        1. 只保留最近N轮对话
        2. System prompt永远保留
        3. 工具调用的结果要保留
        """
        if len(self.conversation_history) > self.max_history * 2:
            # 保留最近的对话
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            print("\n⚠️  对话历史已截断，保留最近10轮")

    def _execute_tool(self, function_name: str, function_args: dict) -> str:
        """安全执行工具"""
        print(f"  🔧 调用工具: {function_name}")
        print(f"     参数: {function_args}")

        self.tool_call_count += 1

        if function_name not in AVAILABLE_TOOLS:
            self.error_count += 1
            return json.dumps({
                "success": False,
                "error": f"工具 '{function_name}' 不存在"
            }, ensure_ascii=False)

        try:
            result = AVAILABLE_TOOLS[function_name](**function_args)

            if result["success"]:
                print(f"     ✅ 成功: {result['result'][:50]}...")
            else:
                print(f"     ❌ 失败: {result['error']}")
                self.error_count += 1

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            self.error_count += 1
            error_msg = f"工具执行异常: {str(e)}"
            print(f"     ❌ {error_msg}")
            return json.dumps({
                "success": False,
                "error": error_msg
            }, ensure_ascii=False)

    def chat(self, user_message: str) -> str:
        """
        处理一轮对话

        返回AI的回复
        """
        print(f"\n{'=' * 60}")
        print(f"👤 用户: {user_message}")
        print('=' * 60)

        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 构建完整消息
        messages = [
                       {"role": "system", "content": self.system_prompt}
                   ] + self.conversation_history

        max_iterations = 3  # 最多迭代3次（防止死循环）
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                # 调用AI
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=messages,
                    tools=tools_definition,
                    tool_choice="auto",
                    temperature=0.7,
                    timeout=15.0
                )

                response_message = response.choices[0].message

                # 检查是否需要调用工具
                if response_message.tool_calls:
                    print(f"\n🤖 AI决定调用 {len(response_message.tool_calls)} 个工具")

                    # 把AI的决策加入历史
                    self.conversation_history.append(response_message)

                    # 执行所有工具调用
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        # 执行工具
                        tool_result = self._execute_tool(function_name, function_args)

                        # 把工具结果加入历史
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_result
                        })

                    # 继续下一轮，让AI根据工具结果生成回答
                    messages = [
                                   {"role": "system", "content": self.system_prompt}
                               ] + self.conversation_history
                    continue

                else:
                    # AI直接回答，不需要工具
                    final_answer = response_message.content

                    # 把AI回复加入历史
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_answer
                    })

                    print(f"\n🤖 AI: {final_answer}")

                    # 管理历史长度
                    self._manage_history()

                    return final_answer

            except Exception as e:
                error_msg = f"对话出错: {str(e)}"
                print(f"\n❌ {error_msg}")
                self.error_count += 1

                if iteration < max_iterations:
                    print("正在重试...")
                    time.sleep(1)
                else:
                    return "抱歉，系统出现问题，请稍后再试。"

        # 达到最大迭代次数
        print("\n⚠️  达到最大迭代次数，可能陷入循环")
        return "抱歉，处理您的请求时遇到了问题。"

    def get_statistics(self):
        """获取统计信息"""
        return {
            "对话轮数": len(self.conversation_history) // 2,
            "工具调用次数": self.tool_call_count,
            "错误次数": self.error_count
        }


# ==================== 交互界面 ====================

def main():
    """主程序"""
    print("=" * 60)
    print("🤖 智能助手 v2.0 - 带工具调用")
    print("=" * 60)
    print("\n可用命令：")
    print("  - 直接输入问题开始对话")
    print("  - 输入 '统计' 查看使用统计")
    print("  - 输入 '清空' 清除对话历史")
    print("  - 输入 '退出' 结束对话")
    print("\n提示：我可以帮你计算、查天气、搜索信息\n")

    assistant = SmartAssistant(max_history=10)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if not user_input:
                continue

            if user_input == "退出":
                print("\n👋 再见！")
                stats = assistant.get_statistics()
                print(f"\n本次会话统计：")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                break

            if user_input == "统计":
                stats = assistant.get_statistics()
                print("\n📊 使用统计：")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            if user_input == "清空":
                assistant.conversation_history = []
                assistant.tool_call_count = 0
                assistant.error_count = 0
                print("\n✅ 对话历史已清空")
                continue

            # 处理对话
            assistant.chat(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 程序错误: {str(e)}")


if __name__ == "__main__":
    main()