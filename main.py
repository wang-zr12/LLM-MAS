import asyncio
from tqdm.asyncio import tqdm
from model.mas1 import MultiAgentFramework, process_batch
from IO import load_file, ResultWriter
import os, json

# 输入输出文件配置
input_data_path = 'CharacterEval/data/test_data.jsonl'
character_profiles_path = 'CharacterEval/data/character_profiles.json'
output_data_path = 'middle_results/generation_mas_1.jsonl'
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

async def main(input_data_path: str, char_path: str, output_file: str):
    # Load test data
    try:
        raw_data = load_file(input_data_path)
        datas = []
        for line in raw_data:
            if isinstance(line, dict):
                datas.append(line)

        if not datas:
            raise ValueError(f"Error: No valid data loaded")
    except Exception as e:
        raise ValueError(f"Error loading test data: {str(e)}")

    # Get processed IDs
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
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

    # Initialize framework
    llm_config = {"model_type": "deepseek", "api_key": os.getenv("DEEPSEEK_API_KEY_1", "")}
    agent_config1 = {"en_specifier": True, "en_critic": True} # for ablation test

    # Process in batches
    batch_size = 40
    framework = MultiAgentFramework(llm_config, agent_config1)

    async with ResultWriter(output_file) as writer:
            for i in tqdm(range(0, len(todo_datas), batch_size), desc="Processing", unit="batch"):
                batch = todo_datas[i:i + batch_size]
                await process_batch(writer, batch, char_path, framework)

if __name__ == "__main__":
    asyncio.run(main(input_data_path, character_profiles_path, output_data_path))