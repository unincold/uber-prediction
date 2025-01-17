import os
import openai
import asyncio
from openai import AsyncOpenAI

# 设置 API 密钥
api_key = ""
openai.api_key = api_key

async def test_agent_run():
    """
    测试 agent.run(prompt) 功能。
    """
    # 构造 prompt
    prompt = (
        "以下是一些特征和目标变量的数据样本：\n\n"
        "feature_1, feature_2, target\n"
        "1, 10, 100\n"
        "2, 20, 200\n"
        "3, 30, 300\n"
        "4, 40, 400\n"
        "5, 50, 500\n\n"
        "请分析这些特征与目标变量之间的潜在关系，并建议可能的隐藏特征（如交互项、非线性变换等）。"
    )

    # 使用 OpenAI API 发送请求
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    # 处理响应
    try:
        text = response.choices[0].message.content.strip()
        usage = response.usage
        
        if "total_tokens" not in usage:
            token_usage = int(usage["prompt_tokens"] + usage["completion_tokens"])
        else:
            token_usage = int(usage["total_tokens"])
        
        print("Response Text:", text)
        print("Token Usage:", token_usage)
    except Exception as e:
        print(f"Error in agent.run: {e}. Please check the response: {response}")

if __name__ == "__main__":
    asyncio.run(test_agent_run())