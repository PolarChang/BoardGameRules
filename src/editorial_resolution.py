"""Conservative, deterministic referent resolution; no model confidence shortcut."""
import re
import unicodedata
from dataclasses import dataclass

PRONOUN = re.compile(r"這個|那個|這枚|那枚|這張|那張|它|這樣|那樣|\b(?:it|this|that|these|those)\b", re.I)


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


@dataclass
class Resolution:
    question: str
    status: str
    cache: bool
    candidates: list


def resolve(question, blocks, previous_user_question="", selected_id=""):
    entities = {}
    for b in blocks:
        f = b["fields"]
        if b["kind"] not in {"規則", "圖片", "術語"}:
            continue
        if b["kind"] != "術語" and f.get("狀態") != "已確認":
            continue
        name = f.get("名稱") or f.get("標題") or f.get("中文")
        if name:
            entities[b["id"]] = (name, [name, *re.split(r"[,，、\n]", f.get("別名", ""))])
    if not PRONOUN.search(question):
        return Resolution(question, "explicit", True, [])
    # "這個罷工代幣" names its referent. More than one explicit entity is not unique.
    hits = [rid for rid, (_, names) in entities.items() if any(n and normalize(n) in normalize(question) for n in names)]
    if len(hits) == 1:
        return Resolution(question, "explicit", True, [])
    if selected_id in entities:
        name = entities[selected_id][0]
        return Resolution(PRONOUN.sub(name, question), "resolved_selection", True, [selected_id])
    previous = [rid for rid, (_, names) in entities.items() if any(n and normalize(n) in normalize(previous_user_question) for n in names)]
    # Only literal single-entity user context is used. AI history is never inspected.
    if len(previous) == 1 and not re.search(r"這樣|那樣|\b(?:this|that) way\b", question):
        return Resolution(PRONOUN.sub(entities[previous[0]][0], question), "resolved_user_context", True, previous)
    return Resolution(question, "ambiguous", False, hits or previous)


def resolve_safely(question, blocks, previous_user_question="", selected_id=""):
    try:
        return resolve(question, blocks, previous_user_question, selected_id)
    except Exception:
        # Technical failure only permits an already standalone question; never guess a referent.
        if not PRONOUN.search(question):
            return Resolution(question, "rewrite_unavailable", False, [])
        return Resolution(question, "ambiguous", False, [])
