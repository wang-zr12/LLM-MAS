import asyncio
import json
import os
import aiohttp
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from IO import ResultWriter, load_file

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("API key not found. Set DEEPSEEK_API_KEY in your environment or .env file")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

# Input/output file configuration
input_data_path = 'CharacterEval/data/test_data.jsonl'
character_profiles_path = 'CharacterEval/data/character_profiles.json'
output_base_dir = 'my_results'
final_output_file = f'{output_base_dir}/generation_mas_0.jsonl'  # File for just final responses
os.makedirs(os.path.dirname(final_output_file), exist_ok=True)


# Load character information
try:
    role_informations = load_file(character_profiles_path)
except Exception as e:
    raise f"Failed to load character profiles: {str(e)}"


def make_inputs(context):
    """Prepare conversation inputs"""
    if not context:
        return []
    return [{"from": part.split("：")[0], "value": "：".join(part.split("：")[1:])}
            for part in context.split('\n') if "：" in part]


def prepare_messages_for_api(conversations, role, system_message):
    """Prepare API messages with conversations"""
    messages = [{"role": "system", "content": system_message}]
    for conv in conversations:
        role_type = "assistant" if conv.get('from') == role else "user"
        messages.append({"role": role_type, "content": f"{conv.get('from', '')}：{conv.get('value', '')}"})

    # Don't send the last message if it's from the assistant (we want to generate that)
    return messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages


def extract_final_response(cot_response):
    """Extract just the final response from a chain-of-thought output"""
    # Look for patterns like "最终回复:" or the last paragraph after reasoning
    if "最终回复:" in cot_response:
        return cot_response.split("最终回复:")[1].strip()

    # If there's a section marker for the final response
    if ":" in cot_response:
        parts = cot_response.split("4.")
        if len(parts) > 1:
            return parts[1].strip()

    # Default: return the last paragraph (assuming reasoning came before)
    paragraphs = [p for p in cot_response.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1].strip()
    return cot_response  # Fallback to the full response


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_response_async(session, data):
    """Asynchronously get API response with chain-of-thought reasoning"""
    try:
        context = data.get('context', '')
        role = data.get('role', '')
        role_info = role_informations.get(role, "")

        # Add Chain-of-Thought instructions to the system prompt
        cot_instructions = """
请按照以下步骤思考并回答：
1. 分析当前对话上下文和情境
2. 思考角色的性格特点、说话方式和可能的反应, 考虑角色在这种情况下的情绪和动机
3. 注意角色相关信息准确度
4. 给出最终回复（最后一段回复应该只包含所扮演角色的最终回复内容即 “角色：内容” 格式）
        """

        system_message = f"{role_info}\n\n{cot_instructions}\n\n现在你是一个角色扮演专家。请你根据上述信息扮演{role}进行对话。"

        messages = prepare_messages_for_api(
            make_inputs(context),
            role,
            system_message
        )

        if not messages:
            data["model_output"] = "Error: 无法生成有效消息"
            data["final_response"] = "Error: 无法生成有效消息"
            return data

        async with session.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 1.5,
                    "max_tokens": 1024,
                    "top_p": 0.8
                }
        ) as response:
            response_json = await response.json()
            if "choices" not in response_json or not response_json["choices"]:
                error_msg = f"Error: Invalid API response: {response_json}"
                data["model_output"] = error_msg
                data["final_response"] = error_msg
                return data

            full_cot_response = response_json["choices"][0]["message"]["content"]
            data["model_output"] = full_cot_response

            # Extract just the final response for the separate file
            data["final_response"] = extract_final_response(full_cot_response)
            return data

    except Exception as e:
        error_msg = f"Error processing data {data.get('id', 'unknown')}: {str(e)}"
        print(error_msg)
        data["model_output"] = error_msg
        data["final_response"] = error_msg
        raise


async def process_batch(session, writer, batch):
    """Process a batch of data and write results to both output files"""
    tasks = [get_response_async(session, data.copy()) for data in batch]
    for future in asyncio.as_completed(tasks):
        try:
            result = await future

            # Write just the final response
            final_result = {k: (v if k != "model_output" else result["final_response"])
                            for k, v in result.items() if k != "final_response"}
            await writer.write_result(final_result)

        except Exception as e:
            print(f"Error processing data: {str(e)}")


async def main():
    # Load test data
    try:
        raw_data = load_file(input_data_path)
        datas = []
        for line in raw_data:
            if isinstance(line, dict):
                datas.append(line)

        if not datas:
            print("Error: No valid data loaded")
            return
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return

    # Get processed IDs for both output files
    processed_ids = set()
    if os.path.exists(final_output_file):
        with open(final_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data.get('id'))
                except:
                    continue


    # Filter unprocessed data
    todo_datas = [d for d in datas if isinstance(d, dict) and d.get('id') not in processed_ids]
    print(f"Total data: {len(datas)}, Processed: {len(datas) - len(todo_datas)}, To process: {len(todo_datas)}")

    if not todo_datas:
        print("No new data to process")
        return

    # Process in batches
    batch_size = 40
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session, \
            ResultWriter(final_output_file) as final_writer:

        for i in tqdm(range(0, len(todo_datas), batch_size),
                      desc="Processing",
                      unit="batch"):
            batch = todo_datas[i:i + batch_size]
            await process_batch(session, final_writer, batch)


if __name__ == "__main__":
    asyncio.run(main())