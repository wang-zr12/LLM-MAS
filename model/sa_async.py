import json
import os
import aiohttp
import asyncio

from openai import APIError
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from IO import ResultWriter, load_file

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("API key not found. Set DEEPSEEK_API_KEY in your environment or .env file")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


headers = {
    "Content-Type": "application/json",
    'Accept': 'application/json',
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

# 输入输出文件配置
input_data_path = 'CharacterEval/data/test_data.jsonl'
character_profiles_path = 'CharacterEval/data/character_profiles.json'
output_file = 'middle_results/generation_sa_r1.jsonl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 加载角色信息
try:
    role_information = load_file(character_profiles_path)
except Exception as e:
    print(f"加载角色信息失败: {str(e)}")
    exit(1)


def make_inputs(context):
    """准备对话输入"""
    if not context:
        return []
    return [{"from": part.split("：")[0], "value": "：".join(part.split("：")[1:])}
            for part in context.split('\n') if "：" in part]


def prepare_messages_for_api(conversations, role, system_message):
    """准备API消息"""
    messages = [{"role": "system", "content": system_message}]
    for conv in conversations:
        role_type = "assistant" if conv.get('from') == role else "user"
        messages.append({"role": role_type, "content": f"{conv.get('from', '')}：{conv.get('value', '')}"})
    return messages[:-1] if messages[-1]["role"] == "assistant" else messages


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_response_async(session, data):
    """异步获取API响应"""
    try:
        context = data.get('context', '')
        role = data.get('role', '')
        role_info = role_information.get(role, "")

        messages = prepare_messages_for_api(
            make_inputs(context),
            role,
            f"你是一个角色扮演专家。根据信息扮演{role}进行对话。\n== 角色设定 ==\n{role_info} "
        )

        if not messages:
            raise ValueError("无法生成有效消息")

        async with session.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json={
                    "model": 'deepseek-chat',
                    "messages": messages,
                    "max_tokens": 1024
                }
        ) as response:
            result = await response.json()
            if not result.get("choices"):
                raise ValueError(f"API响应格式异常: {result}")
            if not result["choices"][0].get("message", {}).get("content"):
                raise ValueError("API返回内容为空")
            data["model_output"] = result["choices"][0]["message"]["content"]
            return data

    except json.JSONDecodeError as e:
        raise ValueError(f"API响应JSON解析失败: {str(e)}")


async def process_batch(session, writer, batch):
    """处理一批数据并写入结果"""
    tasks = []
    for data in batch:
        try:
            if not isinstance(data, dict):
                raise ValueError("数据格式非字典")
            tasks.append(get_response_async(session, data))
        except Exception as e:
            print(f"跳过无效数据 {data.get('id')}: {str(e)}")

    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            await writer.write_result(result)
        except APIError as e:
            print(f"[API错误] {str(e)}")
        except Exception as e:
            print(f"{type(e).__name__}: {str(e)}")


async def main():
    # 加载数据
    datas = load_file(input_data_path)
    if not datas:
        raise ValueError("没有加载到有效数据")

    # 获取已处理ID
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data.get('id'))
                except:
                    continue

    # 过滤未处理数据
    todo_datas = [d for d in datas if isinstance(d, dict) and d.get('id') not in processed_ids]
    print(f"总数据: {len(datas)}, 已处理: {len(datas) - len(todo_datas)}, 待处理: {len(todo_datas)}")

    if not todo_datas:
        print("没有需要处理的新数据")
        return

    # 处理批次
    batch_size = 40
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session, \
            ResultWriter(output_file) as writer:

        for i in tqdm(range(0, len(todo_datas), batch_size),
                      desc="Processing",
                      unit="batch"):
            batch = todo_datas[i:i + batch_size]
            await process_batch(session, writer, batch)


if __name__ == "__main__":
    asyncio.run(main())
