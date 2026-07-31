from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

Label = Literal["POSITIVE", "NEGATIVE"]
Priority = Literal["HIGH", "MEDIUM", "PASS"]
SuggestedAction = Literal[
    "KEEP_CURRENT",
    "REVIEW",
    "MOVE_TO_POSITIVE",
    "MOVE_TO_NEGATIVE",
    "DELETE_OR_SPLIT",
]

HIDDEN_CONTEXT_TERMS = (
    "若",
    "如果",
    "可能",
    "取决于",
    "需确认",
    "未经核实",
    "未核实",
    "未指明",
    "尚未",
    "假设",
    "语境",
    "上下文",
    "有分歧",
    "法务确认",
    "不一定",
    "未必",
)

POSITIVE_HIGH_RISK_L2 = {
    "1.3 职业角色的默认代词",
    "1.6 收件与受众称谓默认",
    "2.5 “首位女性”式顺位标注",
    "2.6 商品、空间与内容的性别化归类",
    "6.1 二元穷举式称呼与并称",
    "6.2 家庭结构的窄化预设",
}

POSITIVE_MEDIUM_RISK_L2 = {
    "1.5 男性词统称混合群体",
    "2.2 性别专用与阴性化称谓",
    "2.3 标题与导语的冗余标性",
    "6.3 表单、字段与选项的强制二元",
    "6.6 空间、设施与分组的非必要划分",
}

MIXED_ITEM_RE = re.compile(r"\s[/>|]\s|/|＞|>")
SPECIFIC_FACT_RE = re.compile(
    r"[\u4e00-\u9fff]{1,4}(先生|女士|小姐|夫人|太太)"
    r"|首位女性|第一位女性|首位男性|第一位男性"
    r"|夫妻关系|父母双方|男女双方"
)

QUOTE_OR_DISTANCE_CUES = (
    "“",
    "”",
    "「",
    "」",
    "『",
    "』",
    '"',
    "说",
    "称",
    "表示",
    "指出",
    "写道",
    "提到",
    "记录",
    "记载",
    "原文",
    "报道",
    "引用",
    "转述",
    "声称",
    "问卷",
    "选项",
    "量表",
    "访谈",
    "供述",
    "判决",
    "条文",
    "法条",
    "规定",
    "据",
    "附录",
    "提纲",
    "字段",
    "编码",
    "标注要求",
    "笔记",
    "反对",
    "批评",
    "质疑",
    "否认",
    "未被采纳",
    "偏见",
    "刻板",
    "歧视",
    "违规",
    "不应",
    "不宜",
    "这也能问",
    "遭到",
)

DIRECT_INTERVENTION_RE = re.compile(
    r"(女人|女性|女生|男人|男性|男生).{0,10}"
    r"(天生|不适合|应该|就该|只能|必须|不许|不能|不如|负责|擅长|不擅长|适合)"
    r"|仅限(男性|女性|男|女)"
    r"|不要招(女生|女性)|不招(女生|女性)"
)

MEDICAL_OR_STATISTICAL_L1 = {
    "二、生理与医学必需",
    "三、制度性与统计性区分",
}


@dataclass(frozen=True)
class AuditRecord:
    id: str
    text: str
    current_label: Label
    l1: str
    l2: str
    register_group: str
    register: str
    original_split: str
    score: int
    priority: Priority
    flags: list[str]
    suggested_action: SuggestedAction
    current_reason: str
    controversial: str
    difficulty: str
    source_row: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _hidden_terms(text: str) -> list[str]:
    return [term for term in HIDDEN_CONTEXT_TERMS if term in text]


def _priority(score: int) -> Priority:
    if score >= 6:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return "PASS"


def audit_positive(row: dict[str, Any]) -> AuditRecord:
    text = _clean(row.get("输入句子"))
    l1 = _clean(row.get("L1类别"))
    l2 = _clean(row.get("L2子类"))
    reason = " ".join(
        _clean(row.get(key)) for key in ("问题说明", "备注", "处置理由")
    )

    score = 0
    flags: list[str] = []

    if _clean(row.get("是否争议")) == "是":
        score += 3
        flags.append("争议样本")

    if _clean(row.get("严重度")) in {"建议改", "谨慎"}:
        score += 2
        flags.append("处置强度较弱")

    hidden = _hidden_terms(reason)
    if hidden:
        score += min(4, 1 + len(hidden))
        flags.append("依赖假设或隐藏语境：" + "、".join(hidden[:4]))

    if l2 in POSITIVE_HIGH_RISK_L2:
        score += 3
        flags.append("已知高风险边界子类")
    elif l2 in POSITIVE_MEDIUM_RISK_L2:
        score += 1
        flags.append("边界子类")

    mixed = bool(MIXED_ITEM_RE.search(text))
    if mixed:
        score += 3
        flags.append("同条混合多个现象或结构化片段")

    specific_fact = bool(SPECIFIC_FACT_RE.search(text))
    if specific_fact:
        score += 2
        flags.append("可被理解为具体事实或具体对象")

    if "多触发点" in _clean(row.get("备注")):
        score += 1
        flags.append("多触发点")

    if mixed:
        suggestion: SuggestedAction = "DELETE_OR_SPLIT"
    elif specific_fact and (hidden or l2 in POSITIVE_HIGH_RISK_L2):
        suggestion = "MOVE_TO_NEGATIVE"
    elif score >= 3:
        suggestion = "REVIEW"
    else:
        suggestion = "KEEP_CURRENT"

    return AuditRecord(
        id=_clean(row.get("编号")),
        text=text,
        current_label="POSITIVE",
        l1=l1,
        l2=l2,
        register_group=_clean(row.get("语体大类")),
        register=_clean(row.get("语体")),
        original_split=_clean(row.get("切分")),
        score=score,
        priority=_priority(score),
        flags=flags,
        suggested_action=suggestion,
        current_reason=_clean(row.get("问题说明")),
        controversial=_clean(row.get("是否争议")),
        difficulty=_clean(row.get("难度")),
        source_row=dict(row),
    )


def audit_negative(row: dict[str, Any]) -> AuditRecord:
    text = _clean(row.get("输入句子"))
    l1 = _clean(row.get("L1类别"))
    l2 = _clean(row.get("L2子类"))
    reason = " ".join(
        _clean(row.get(key)) for key in ("保留理由", "备注", "处置理由")
    )

    # The seven rows moved in the first v2.1 clean-up have already been reviewed.
    if _clean(row.get("处置理由")) == "单句证据不足，优先保持原文":
        return AuditRecord(
            id=_clean(row.get("编号")),
            text=text,
            current_label="NEGATIVE",
            l1=l1,
            l2=l2,
            register_group=_clean(row.get("语体大类")),
            register=_clean(row.get("语体")),
            original_split=_clean(row.get("切分")),
            score=0,
            priority="PASS",
            flags=["v2.1 已人工确认"],
            suggested_action="KEEP_CURRENT",
            current_reason=_clean(row.get("保留理由")),
            controversial=_clean(row.get("是否争议")),
            difficulty=_clean(row.get("难度")),
            source_row=dict(row),
        )

    score = 0
    flags: list[str] = []

    if _clean(row.get("是否争议")) == "是":
        score += 3
        flags.append("争议样本")

    hidden = _hidden_terms(reason)
    if hidden:
        score += min(4, 1 + len(hidden))
        flags.append("保留理由依赖语境：" + "、".join(hidden[:4]))

    has_direct_problem = bool(DIRECT_INTERVENTION_RE.search(text))
    has_scope_cue = any(cue in text for cue in QUOTE_OR_DISTANCE_CUES)
    if (
        has_direct_problem
        and not has_scope_cue
        and l1 not in MEDICAL_OR_STATISTICAL_L1
    ):
        score += 5
        flags.append("文本直接表达干预问题，但缺少引用、反驳或事实范围线索")

    if has_direct_problem and not has_scope_cue and l1 not in MEDICAL_OR_STATISTICAL_L1:
        suggestion: SuggestedAction = "MOVE_TO_POSITIVE"
    elif score >= 3:
        suggestion = "REVIEW"
    else:
        suggestion = "KEEP_CURRENT"

    return AuditRecord(
        id=_clean(row.get("编号")),
        text=text,
        current_label="NEGATIVE",
        l1=l1,
        l2=l2,
        register_group=_clean(row.get("语体大类")),
        register=_clean(row.get("语体")),
        original_split=_clean(row.get("切分")),
        score=score,
        priority=_priority(score),
        flags=flags,
        suggested_action=suggestion,
        current_reason=_clean(row.get("保留理由")),
        controversial=_clean(row.get("是否争议")),
        difficulty=_clean(row.get("难度")),
        source_row=dict(row),
    )
