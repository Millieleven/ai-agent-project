# AI Agent 学习项目

## Day 1: Prompt工程与对话系统
- ✅ 调通API
- ✅ 实现带System Prompt的对话
- ✅ 对话记忆管理
- ✅ 评测系统

**学到的核心概念**：
- System Prompt控制AI行为
- Temperature参数的作用
- 如何减少幻觉

## Day 2: Function Calling与工具调用 ⭐
- ✅ 理解Function Calling原理
- ✅ 实现3个工具（计算器、天气、搜索）
- ✅ 完善的异常处理
- ✅ 对话历史管理
- ✅ 评测系统

**核心能力**：
```python
# AI决定调用工具
AI: "我需要调用calculator，参数是'123*456'"

# 我们执行
result = calculator("123*456")

# AI根据结果回答
AI: "计算结果是56088"
