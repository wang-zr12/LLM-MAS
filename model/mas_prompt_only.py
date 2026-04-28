"""
Multi-Agent Role-Playing Framework — prompt-only key information extraction variant.

Difference vs. mas1.py:
    The Analyzer (TaskSpecifier) runs three additional prompt-only LLM passes
    before reformulating the role description:
        (1) Structured fact extraction  — JSON slot dict, NER-style.
        (2) Async coreference resolution — pronoun -> role name.
        (3) Conflict-aware KB update    — compare with persisted slots and overwrite.
    No new model / regex rules are introduced; everything is driven by prompts
    against the same LLM already used by the framework.
    The KB is persisted as JSONL (one line per upsert) under ``middle_results/``.
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from LLMInterface import LLMInterface, DeepSeekLLM
from IO import ResultWriter, load_file
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_KB_JSONL = "middle_results/kb_updates.jsonl"


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


# Knowledge Base — in-memory + JSONL persistence
class JsonlKnowledgeBase:
    """role -> slot -> {value, source, timestamp}; appends every update to JSONL."""

    SLOTS = (
        "name", "alias", "age", "gender", "birthday", "location",
        "occupation", "relation", "skill", "personality", "goal", "item",
    )

    def __init__(self, jsonl_path: str = DEFAULT_KB_JSONL):
        self._kb: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        self._lock = asyncio.Lock()

    def render(self, role: str) -> str:
        slots = self._kb.get(role, {})
        return "\n".join(f"- {k}: {v['value']}" for k, v in slots.items()) if slots else ""

    def get_value(self, role: str, slot: str) -> Optional[str]:
        return self._kb.get(role, {}).get(slot, {}).get("value")

    async def upsert(self, role: str, slot: str, value: str, source: str) -> Dict:
        """Insert or overwrite a slot. Persists one JSONL record. Returns event dict."""
        if not value:
            return {}
        async with self._lock:
            prev = self._kb[role].get(slot)
            event = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "slot": slot,
                "value": value,
                "source": source,
                "previous": prev["value"] if prev else None,
                "is_conflict": bool(prev and prev["value"] != value),
            }
            self._kb[role][slot] = {
                "value": value,
                "source": source,
                "timestamp": event["timestamp"],
            }
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            return event


# Prompt-only Key Information Extractor
class PromptKIE:
    """All three stages (NER / coref / conflict-merge) realized by prompts only."""

    def __init__(self, llm: LLMInterface, kb: JsonlKnowledgeBase):
        self.llm = llm
        self.kb = kb

    @staticmethod
    def _safe_json(text: str) -> Dict:
        try:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {}
            return json.loads(text[start:end + 1])
        except Exception:
            return {}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    async def extract_facts(self, role: str, text: str) -> Dict[str, str]:
        if not text or not text.strip():
            return {}
        slots = ", ".join(JsonlKnowledgeBase.SLOTS)
        system_message = (
            "你是一个中文结构化事实抽取器。请从给定文本中抽取与目标角色相关的关键事实，"
            f"键限定为以下集合之一: {slots}。"
            "仅以严格 JSON 输出（无注释、无解释、无多余字段），无信息时输出 {}。"
        )
        user_message = f"目标角色: {role}\n文本:\n{text}"
        try:
            resp = await self.llm.generate_response([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ])
            obj = self._safe_json(resp)
            return {k: str(v).strip() for k, v in obj.items()
                    if k in JsonlKnowledgeBase.SLOTS and v}
        except Exception as e:
            logger.warning(f"[KIE] extract_facts failed: {e}")
            return {}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    async def resolve_coref(self, role: str, text: str) -> str:
        if not text or not text.strip():
            return text
        kb_view = self.kb.render(role) or "(暂无)"
        system_message = (
            "你是一个中文共指消解器。请将文本中明确指代『目标角色』的代词"
            "（如 他/她/那个人/此人/该角色）替换为目标角色的姓名；"
            "其它代词保留原样。仅输出替换后的文本，不要附加解释。"
        )
        user_message = f"目标角色: {role}\n已知信息:\n{kb_view}\n原文:\n{text}"
        try:
            out = await self.llm.generate_response([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ])
            return out.strip() or text
        except Exception as e:
            logger.warning(f"[KIE] resolve_coref failed: {e}")
            return text

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    async def reconcile_with_kb(
        self, role: str, new_facts: Dict[str, str]
    ) -> Dict[str, str]:
        """Ask the LLM to decide which new facts should overwrite the existing KB.

        Returns the accepted facts (subset of new_facts, possibly with refined values).
        """
        if not new_facts:
            return {}
        kb_view = self.kb.render(role) or "(暂无)"
        system_message = (
            "你是一个知识冲突裁决器。给定『已有知识』和『新抽取事实』，"
            "判断每个新事实是否应当采纳并写入知识库：\n"
            "- 若与已有不冲突：采纳；\n"
            "- 若与已有冲突：仅当新事实更明确、更具体或文中明确更新时采纳，否则丢弃；\n"
            "- 若新事实显然为错误或幻觉：丢弃。\n"
            "仅以 JSON 输出最终应写入的键值对，无需理由。"
        )
        user_message = (
            f"目标角色: {role}\n已有知识:\n{kb_view}\n"
            f"新抽取事实:\n{json.dumps(new_facts, ensure_ascii=False)}"
        )
        try:
            resp = await self.llm.generate_response([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ])
            obj = self._safe_json(resp)
            return {k: str(v).strip() for k, v in obj.items()
                    if k in JsonlKnowledgeBase.SLOTS and v}
        except Exception as e:
            logger.warning(f"[KIE] reconcile_with_kb failed: {e}")
            return new_facts  # fall back to taking everything

    async def process(self, role: str, role_info: str, context: str) -> Dict:
        # Seed once from the trusted role profile.
        if not self.kb._kb.get(role):
            seeds = await self.extract_facts(role, role_info)
            for slot, value in seeds.items():
                await self.kb.upsert(role, slot, value, source="role_profile")

        # Run coref + extraction concurrently.
        resolved_context = await self.resolve_coref(role, context)
        new_facts = await self.extract_facts(role, resolved_context)

        # Conflict-aware merge through a prompt.
        accepted = await self.reconcile_with_kb(role, new_facts)
        events = []
        for slot, value in accepted.items():
            ev = await self.kb.upsert(role, slot, value, source="dialogue")
            if ev:
                events.append(ev)

        return {
            "kb_snapshot": self.kb.render(role),
            "new_facts": new_facts,
            "accepted": accepted,
            "events": events,
            "resolved_context": resolved_context,
        }


# Task Specifier (Analyzer) — prompt-only KIE injected
class TaskSpecifier:
    def __init__(self, llm: LLMInterface, kb: JsonlKnowledgeBase,
                 max_context_tokens: int = 4096, max_response_time: int = 15):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.max_response_time = max_response_time
        self.kie = PromptKIE(llm, kb)

    @staticmethod
    def prepare_messages(system_message: str, user_message: str) -> List[Dict]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def specify_task(self, role: str, context: str, role_info: str) -> str:
        report = await self.kie.process(role, role_info, context)
        kb_block = report["kb_snapshot"] or "(暂无结构化信息)"
        conflict_lines = [
            f"- {e['slot']}: {e['previous']} -> {e['value']}"
            for e in report["events"] if e.get("is_conflict")
        ]
        conflict_block = "\n".join(conflict_lines) or "(无冲突)"

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
            f"== 对话文本 ==\n{report['resolved_context']}#"
        )
        messages = self.prepare_messages(system_message, user_message)

        try:
            return await self.llm.generate_response(messages, self.max_context_tokens)
        except Exception as e:
            logger.error(f"Task specification failed: {e}")
            raise ValueError(f"error: {e}")


# Critic Section
class CriticSection:
    def __init__(self, llm: LLMInterface, memory: Dict):
        self.llm = llm
        self.memory = memory

    @staticmethod
    def prepare_messages(system_message: str, user_message: str) -> List[Dict]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
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
        self.kb = JsonlKnowledgeBase(
            jsonl_path=agent_config.get("kb_jsonl", DEFAULT_KB_JSONL)
        )
        self.task_specifier = TaskSpecifier(self.llm, self.kb)
        self.critic = CriticSection(self.llm, self.memory_log.data)
        self.agents = [ActionAgent(self.llm, f"Agent{i + 1}", self.memory_log.data) for i in range(1)]

    async def run(self, data: Dict, char_path: str) -> dict:
        try:
            role = data.get("role", "")
            context = data.get("context", "")
            role_information = load_file(char_path)
            role_info = role_information.get(role, "")

            if self.en_specifier:
                task = await self.task_specifier.specify_task(role, context, role_info)
            else:
                task = role_info

            outputs = []
            for agent in self.agents:
                output = await agent.execute(task, context)
                self.memory_log.update(f"{agent.role} output: {output}")
                data["model_output"] = output
                outputs.append(output)

            if self.en_critic:
                feedback = await self.critic.evaluate(outputs[0], role, task, context)
                if "error" in feedback:
                    logger.error(f"Evaluation failed: {feedback['error']}")
                    return {"Error": {feedback['error']}}

                if not feedback.get("pass"):
                    adjusted_output = await self.agents[0].execute(task, context, feedback["feedback"])
                    self.memory_log.update(f"{self.agents[0].role} adjusted output: {adjusted_output}")
                    outputs[0] = adjusted_output
                    data["model_output"] = adjusted_output

            return data

        except Exception as e:
            raise ConnectionRefusedError(f"Internal failure,{e}")


# Batch Processing
async def process_batch(writer: ResultWriter, batch: List[Dict], char_path: str,
                        framework: MultiAgentFramework):
    tasks = [framework.run(data, char_path=char_path) for data in batch]
    for future in asyncio.as_completed(tasks):
        try:
            if isinstance(future, Exception):
                raise future
            result = await future
            await writer.write_result(result)
        except Exception as e:
            print(f"Error processing data: {str(e)}")
