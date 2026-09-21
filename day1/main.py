# main.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from prompts import SYSTEM_V1, SYSTEM_V2, SYSTEM_V3

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)


def ask_ai(question: str, system_prompt: str) -> str:
    """单次问答，不带记忆"""
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def compare_prompts(question: str):
    """对比三种 system prompt 的效果"""
    print(f"\n问题：{question}")
    print("=" * 50)

    print("\n【版本1 - 无约束】")
    print(ask_ai(question, SYSTEM_V1))

    print("\n【版本2 - 有约束】")
    print(ask_ai(question, SYSTEM_V2))

    print("\n【版本3 - 结构化】")
    print(ask_ai(question, SYSTEM_V3))


def chat_with_memory():
    """带记忆的多轮对话"""
    history = []  # 对话历史（不含 system）

    print("\n开始对话（输入 '退出' 结束，输入 '清空' 清除记忆）\n")

    while True:
        user_input = input("你：").strip()

        if user_input == "退出":
            print("再见！")
            break

        if user_input == "清空":
            history = []
            print("记忆已清空\n")
            continue

        if not user_input:
            continue

        # 1. 把用户输入追加到历史
        history.append({"role": "user", "content": user_input})

        # 2. 构建完整消息：system + 全部历史
        messages = [{"role": "system", "content": SYSTEM_V2}] + history

        # 3. 请求
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        ai_reply = response.choices[0].message.content

        # 4. 把 AI 回复也追加到历史，形成闭环
        history.append({"role": "assistant", "content": ai_reply})

        print(f"AI：{ai_reply}\n")

        # 5. 历史过长提醒
        if len(history) > 20:
            print("⚠️  对话历史较长，建议输入 '清空' 重置\n")


if __name__ == "__main__":
    # 二选一：想测 prompt 就开这个
    # compare_prompts("我叫张三")
    # compare_prompts("我叫什么名字")

    # 想测记忆就开这个
    chat_with_memory()