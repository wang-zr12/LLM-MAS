import asyncio
import json
import logging
import os
from LLMInterface import LLMInterface, DeepSeekLLM
from typing import Dict, List, Optional
from datetime import datetime
from IO import ResultWriter, load_file
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 输入输出文件配置
input_data_path = 'CharacterEval/data/test_data.jsonl'
character_profiles_path = 'CharacterEval/data/character_profiles.json'
output_data_path = 'middle_results/generation_mas_.jsonl'
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)


# Memory and Log Management
class MemoryLog:
    def __init__(self, max_history: int = 10):
        self.data = {"history": [], "logs": []}
        self.max_history = max_history

    def update(self, entry: str):
        self.data["history"].append(entry)
        self.data["logs"].append({"timestamp": str(datetime.now()), "entry": entry})
        if len(self.data["history"]) > self.max_history:
            self.data["history"].pop(0)

    def get_context(self) -> str:
        return "\n".join(self.data["history"])

# Task Specifier
class TaskSpecifier:
    def __init__(self, llm: LLMInterface, max_context_tokens: int = 4096, max_response_time: int = 15):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.max_response_time = max_response_time

    @staticmethod
    def prepare_messages(system_message: str, user_message: str) -> List[Dict]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def specify_task(self, role: str, context: str, role_info: str) -> str:
        system_message = (
            f"你是一名专注于角色扮演的分析与总结专家。请根据以下要素重构角色背景信息：\n"
            f"== 执行步骤 == \n"
            f"角色分析 - 解析{role}的设定文档; 上下文过滤 - 在背景信息提取与当前对话场景相关的要素并排除无用信息; 知识补充 - 填补角色设定的相关合理细节; 风格适配 - 确保符合角色语言特征\n"
            f"== 输出要求 ==\n对{role}重新组织背景说明"
        )
        user_message = f"== {role} Information == \n{role_info}\n== Conversation Context ==\n{context}#"
        messages = self.prepare_messages(system_message, user_message)

        try:
            response = await self.llm.generate_response(messages, self.max_context_tokens)
            return response
        except Exception as e:
            logger.error(f"Task specification failed: {e}")
            raise ValueError("error: {str(e)}")

# Critic Section
class CriticSection:
    def __init__(self, llm: LLMInterface, memory: Dict):
        self.llm = llm
        self.memory = memory

    @staticmethod
    def prepare_messages(system_message: str, user_message: str) -> List[Dict]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    async def evaluate(self, model_output: str, role: str, role_info: str, context: str) -> Dict:
        system_message = (
            f"You are an evaluator for role-playing task critic.\n"
            f"== Metrics for 'Next-Output Prediction' evaluation ==\n"
            f"[Conversational Ability]: fluency, coherence, consistency;\n"
            f"[Role-playing appeal]: personification, communication skills, expression diversity, empathy;\n"
            f"[Role consistency]: knowledge exposure, knowledge accuracy, knowledge hallucination, character behavior, personal utterance.\n"
            f"== Final Output ==\n If identified significant shortcomings then suggest improvements, Else,only output 'pass'/'通过'."
        )
        user_message = (
            f"=== {role} Information ===\n{role_info}\n"
            f"=== Conversation Context ===\n{context}\n"
            f"=== Next-Output Prediction as {role} ===\n{model_output}"
        )
        messages = self.prepare_messages(system_message, user_message)

        try:
            response = await self.llm.generate_response(messages)
            return {"pass": response == "pass" or "通过", "feedback": f"#Previous unsatisfied output:{model_output}.\n#Improvement suggestion:{response}"}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

# Action Agent
class ActionAgent:
    def __init__(self, llm: LLMInterface, role: str, memory: Dict):
        self.llm = llm
        self.role = role
        self.memory = memory

    @staticmethod
    def make_inputs(context: str) -> List[Dict]:
        if not context:
            return []
        return [{"from": part.split("：")[0], "value": "：".join(part.split("：")[1:])}
                for part in context.split('\n') if "：" in part]

    @staticmethod
    def prepare_messages(conversations: List[Dict], role: str, system_message: str) -> List[Dict]:
        messages = [{"role": "system", "content": system_message}]
        for conv in conversations:
            role_type = "assistant" if conv.get('from') == role else "user"
            messages.append({"role": role_type, "content": f"{conv.get('from', '')}：{conv.get('value', '')}"})
        return messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute(self, task: str, context: str, critic_feedback: Optional[str] = None) -> str:
        system_message = (
            f"你现在是 [{self.role}], 严格遵守以下设定:\n"
            f"== Role information ==\n {task}\n给出最终回复"
        )
        if critic_feedback:
            system_message += f"\nFeedback: {critic_feedback}"
        messages = self.prepare_messages(self.make_inputs(context), self.role, system_message)

        try:
            return await self.llm.generate_response(messages)
        except Exception as e:
            logger.error(f"Action agent {self.role} failed: {e}")
            raise RuntimeError(e)

# Multi-Agent Framework
class MultiAgentFramework:
    def __init__(self, llm_config: Dict, agent_config: Dict):
        model_type = llm_config.get("model_type", "deepseek")
        api_key = llm_config.get("api_key")
        if model_type == "deepseek":
            self.llm = DeepSeekLLM(api_key, llm_config.get("model", "deepseek-chat"))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.en_specifier = agent_config.get("en_specifier")
        self.en_critic = agent_config.get("en_critic")
        self.memory_log = MemoryLog()
        self.task_specifier = TaskSpecifier(self.llm)
        self.critic = CriticSection(self.llm, self.memory_log.data)
        self.agents = [ActionAgent(self.llm, f"Agent{i+1}", self.memory_log.data) for i in range(1)]

    async def run(self, data: Dict, char_path: str) -> dict:
        try:
            role = data.get("role", "")
            context = data.get("context", "")
            role_information = load_file(char_path)
            role_info = role_information.get(role, "")

            # Task specification
            if self.en_specifier:
                task = await self.task_specifier.specify_task(role, context, role_info)
            else:
                task = role_info

            # Agent execution
            outputs = []
            for agent in self.agents:
                output = await agent.execute(task, context)
                self.memory_log.update(f"{agent.role} output: {output}")
                data["model_output"] = output
                outputs.append(output)

            # Critic evaluation
            if self.en_critic:
                feedback = await self.critic.evaluate(outputs[0], role, task, context)
                if "error" in feedback:
                    logger.error(f"Evaluation failed: {feedback['error']}")
                    return {"Error": {feedback['error']}}

                # Adjust if feedback indicates issues
                if not feedback.get("pass"):
                    adjusted_output = await self.agents[0].execute(task, context, feedback["feedback"])
                    self.memory_log.update(f"{self.agents[0].role} adjusted output: {adjusted_output}")
                    outputs[0] = adjusted_output
                    data["model_output"] = adjusted_output

            return data

        except Exception as e:
            raise ConnectionRefusedError(f"Internal failure,{e}")

# Batch Processing
async def process_batch(writer: ResultWriter, batch: List[Dict], framework: MultiAgentFramework):
    tasks = [framework.run(data, char_path='CharacterEval/data/character_profiles.json') for data in batch]
    for future in asyncio.as_completed(tasks):
        try:
            if isinstance(future, Exception):
                raise future
            result = await future
            await writer.write_result(result)
        except Exception as e:
            print(f"Error processing data: {str(e)}")

# Main Function
async def main(input_data_path: str, output_file: str):
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

    output_file = output_file.split('.json')[0] + f'drop_spe.jsonl'

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
    logger.info(f"Total data: {len(datas)}, Processed: {len(datas) - len(todo_datas)}, To process: {len(todo_datas)}")

    if not todo_datas:
        logger.info("No new data to process")
        return

    # Initialize framework
    llm_config = {"model_type": "deepseek", "api_key": os.getenv("DEEPSEEK_API_KEY_1", "")}
    agent_config1 = {"en_specifier": False, "en_critic": True} # for ablation test

    # Process in batches
    batch_size = 40
    framework = MultiAgentFramework(llm_config, agent_config1)

    async with ResultWriter(output_file) as writer:
            for i in tqdm(range(0, len(todo_datas), batch_size), desc="Processing", unit="batch"):
                batch = todo_datas[i:i + batch_size]
                await process_batch(writer, batch, framework)

if __name__ == "__main__":
    asyncio.run(main(input_data_path, output_data_path))