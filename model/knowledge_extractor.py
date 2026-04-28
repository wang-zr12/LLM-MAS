"""
Key Information Extraction module for the Analyzer (Task Specifier).

Pipeline:
    1. NER + rule-based structured fact extraction
    2. Asynchronous coreference resolution
    3. New-information conflict detection & knowledge-base update
"""
import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from LLMInterface import LLMInterface
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# Rule patterns for fast structured extraction.
RULE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("age",        re.compile(r"(\d{1,3})\s*(?:岁|年纪|周岁)")),
    ("birthday",   re.compile(r"(?:生日|出生于|出生在)\s*[:：]?\s*([0-9]{1,4}[\-/年][0-9]{1,2}[\-/月][0-9]{0,2}日?)")),
    ("location",   re.compile(r"(?:住在|居住在|来自|身处|位于)\s*([\u4e00-\u9fa5A-Za-z]{2,15})")),
    ("occupation", re.compile(r"(?:职业|身份|工作)\s*[:：是为]\s*([\u4e00-\u9fa5A-Za-z]{2,12})")),
    ("relation",   re.compile(r"(?:我的|是)\s*(父亲|母亲|哥哥|姐姐|弟弟|妹妹|师傅|师父|徒弟|朋友|爱人|妻子|丈夫)")),
    ("alias",      re.compile(r"(?:别名|又名|外号|绰号)\s*[:：]?\s*([\u4e00-\u9fa5A-Za-z·]{1,20})")),
]

PRONOUN_RE = re.compile(r"(他|她|它|他们|她们|那个人|这个人|此人|该角色)")


class KnowledgeBase:
    """Per-role slot store with provenance tracking."""

    def __init__(self):
        self._kb: Dict[str, Dict[str, Dict]] = defaultdict(dict)

    def get(self, role: str) -> Dict[str, Dict]:
        return self._kb.get(role, {})

    def get_value(self, role: str, slot: str) -> Optional[str]:
        return self._kb.get(role, {}).get(slot, {}).get("value")

    def upsert(self, role: str, slot: str, value: str, source: str) -> Tuple[bool, Optional[str]]:
        prev = self._kb[role].get(slot)
        if prev and prev["value"] != value:
            old = prev["value"]
            self._kb[role][slot] = {
                "value": value,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "previous": old,
            }
            return True, old
        self._kb[role][slot] = {
            "value": value,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        return False, None

    def render(self, role: str) -> str:
        slots = self._kb.get(role, {})
        if not slots:
            return ""
        return "\n".join(f"- {k}: {v['value']}" for k, v in slots.items())

    def to_json(self, role: str) -> str:
        return json.dumps(self._kb.get(role, {}), ensure_ascii=False, indent=2)


class KnowledgeExtractor:
    """NER + rule-based structured fact extraction with async coref and conflict update."""

    def __init__(self, llm: LLMInterface, kb: Optional[KnowledgeBase] = None):
        self.llm = llm
        self.kb = kb or KnowledgeBase()
        self._lock = asyncio.Lock()

    @staticmethod
    def _rule_extract(text: str) -> Dict[str, str]:
        facts: Dict[str, str] = {}
        for slot, pattern in RULE_PATTERNS:
            m = pattern.search(text)
            if m:
                facts[slot] = m.group(1).strip()
        return facts

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    async def _llm_ner(self, role: str, text: str) -> Dict[str, str]:
        if not text.strip():
            return {}
        system_message = (
            "你是一个中文命名实体抽取器。从给定文本中抽取与目标角色相关的结构化事实，"
            "仅以严格 JSON 形式输出，键限定在: "
            "name, alias, age, gender, birthday, location, occupation, relation, "
            "skill, personality, goal, item。无信息则给空对象 {}。不要输出多余文字。"
        )
        user_message = f"目标角色: {role}\n文本: {text}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        try:
            resp = await self.llm.generate_response(messages)
            resp = resp.strip()
            start, end = resp.find("{"), resp.rfind("}")
            if start == -1 or end == -1:
                return {}
            obj = json.loads(resp[start:end + 1])
            return {k: str(v) for k, v in obj.items() if v}
        except Exception as e:
            logger.warning(f"LLM NER failed: {e}")
            return {}

    async def extract_facts(self, role: str, text: str) -> Dict[str, str]:
        llm_facts, rule_facts = await asyncio.gather(
            self._llm_ner(role, text),
            asyncio.to_thread(self._rule_extract, text),
        )
        merged = {**llm_facts, **rule_facts}
        return merged

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    async def resolve_coreference(self, role: str, text: str) -> str:
        if not PRONOUN_RE.search(text):
            return text
        kb_view = self.kb.render(role) or "(暂无)"
        system_message = (
            "你是一个中文共指消解器。给定目标角色和已有知识，请将文本中明确指代该角色的代词"
            "（如 他/她/那个人/此人/该角色）替换为角色姓名，其它代词保留原样。"
            "仅输出替换后的文本，不要解释。"
        )
        user_message = f"目标角色: {role}\n已知信息:\n{kb_view}\n原文:\n{text}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        try:
            return (await self.llm.generate_response(messages)).strip() or text
        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}")
            return text

    async def update_with_conflict_check(
        self, role: str, facts: Dict[str, str], source: str = "dialogue"
    ) -> List[Dict]:
        conflicts: List[Dict] = []
        async with self._lock:
            for slot, value in facts.items():
                is_conflict, old = self.kb.upsert(role, slot, value, source)
                if is_conflict:
                    conflicts.append({
                        "slot": slot, "old": old, "new": value, "source": source,
                    })
                    logger.info(
                        f"[KB conflict] role={role} slot={slot} old={old!r} -> new={value!r} (source={source})"
                    )
        return conflicts

    async def process(self, role: str, role_info: str, context: str) -> Dict:
        if not self.kb.get(role):
            seed_facts = await self.extract_facts(role, role_info)
            await self.update_with_conflict_check(role, seed_facts, source="role_profile")

        resolved_context = await self.resolve_coreference(role, context)
        new_facts = await self.extract_facts(role, resolved_context)
        conflicts = await self.update_with_conflict_check(role, new_facts, source="dialogue")

        return {
            "kb_snapshot": self.kb.render(role),
            "new_facts": new_facts,
            "conflicts": conflicts,
            "resolved_context": resolved_context,
        }
