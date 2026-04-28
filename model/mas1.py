"""
Multi-Agent Role-Playing Framework With batch processing support.
"""
import asyncio
import logging
import os
from LLMInterface import LLMInterface, DeepSeekLLM
from typing import Dict, List, Optional
from datetime import datetime
from IO import ResultWriter, load_file
from tenacity import retry, stop_after_attempt, wait_exponential
from model.knowledge_extractor import KnowledgeBase, KnowledgeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Memory and Log Management
class MemoryLog:
    def __init__(self, max_history: int = 100):
        self.data = {"history": [], "logs": []}
        self.max_history = max_history

    def update(self, entry: str):
        self.data["history"].append(entry)
        self.data["logs"].append({"timestamp": str(datetime.now()), "entry": entry})
        if len(self.data["history"]) > self.max_history:
            self.data["history"].pop(0)

    def get_context(self) -> str:
        return "\n".join(self.data["history"])


# Task Specifier (Analyzer)
class TaskSpecifier:
    def __init__(self, llm: LLMInterface, max_context_tokens: int = 4096, max_response_time: int = 15,
                 kb: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.max_response_time = max_response_time
        self.kb = kb or KnowledgeBase()
        self.extractor = KnowledgeExtractor(llm, self.kb)

    @staticmethod
    def prepare_messages(system_message: str, user_message: str) -> List[Dict]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def specify_task(self, role: str, context: str, role_info: str) -> str:
        # Stage 0: key-information extraction (NER + rules), async coref, conflict-aware KB update.
        kb_report = await self.extractor.process(role, role_info, context)
        kb_block = kb_report["kb_snapshot"] or "(暂无结构化信息)"
        conflict_block = (
            "\n".join(f"- {c['slot']}: {c['old']} -> {c['new']}" for c in kb_report["conflicts"])
            or "(无冲突)"
        )

        system_message = (
            f"你是一名专注于角色扮演的分析与总结专家。请根据以下要素重构角色背景信息：\n"
            f"== 执行步骤 == \n"
            f"角色分析 - 解析{role}的设定词典; "
            f"结构化事实参考 - 优先采用知识库中的事实，遇到冲突时以最新事实为准; "
            f"上下文过滤 - 在背景信息中提取与对话相关的信息，排除冗余信息; "
            f"知识补充 - 如果信息较少则根据经验知识补充角色设定以及与对话有关的合理细节。\n "
            f"== 输出要求 ==\n重新组织{role}人物说明。"
        )
        user_message = (
            f"== {role} 信息 == \n{role_info}\n"
            f"== 结构化知识库 ==\n{kb_block}\n"
            f"== 冲突更新 ==\n{conflict_block}\n"
            f"== 对话文本 ==\n{kb_report['resolved_context']}#"
        )
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
            f"你是一个角色扮演任务评价器，给出历史对话，角色信息和指标，评价模型输出。\n"
            f"== 'Model Output' 的评价指标 ==\n"
            f"[Conversational Ability]: coherence, consistency;\n"
            f"[Role-playing appeal]: personification, communication skills, expression diversity, empathy;\n"
            f"[Role consistency]: knowledge exposure, knowledge accuracy, knowledge hallucination.\n"
            f"== 输出 ==\n "
            f"如果在任何方面有明显不足，输出改进意见(注意不能偏离角色设定); 反之则输出'通过'."
        )
        user_message = (
            f"=== {role} Information ===\n{role_info}\n"
            f"=== Conversation Context ===\n{context}\n"
            f"=== Model Output as {role} ===\n{model_output}"
        )
        messages = self.prepare_messages(system_message, user_message)

        try:
            response = await self.llm.generate_response(messages)
            return {"pass": response == "通过",
                    "feedback": f"Previous unsatisfied output:{model_output}.\nImprovement suggestion:{response}"}
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
            f"== Role information ==\n {task}\n"
            f"给出最终回复。"
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
        self.knowledge_base = KnowledgeBase()
        self.task_specifier = TaskSpecifier(self.llm, kb=self.knowledge_base)
        self.critic = CriticSection(self.llm, self.memory_log.data)
        self.agents = [ActionAgent(self.llm, f"Agent{i + 1}", self.memory_log.data) for i in range(1)]

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
async def process_batch(writer: ResultWriter, batch: List[Dict], char_path: str, framework: MultiAgentFramework):
    tasks = [framework.run(data, char_path=char_path) for data in batch]
    for future in asyncio.as_completed(tasks):
        try:
            if isinstance(future, Exception):
                raise future
            result = await future
            await writer.write_result(result)
        except Exception as e:
            print(f"Error processing data: {str(e)}")

# Main Function
