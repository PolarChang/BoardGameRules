"""Server-authoritative draft and publication gates, including visual provenance."""
import re

KINDS = {"文件", "來源", "術語", "規則", "FAQ", "涵蓋", "圖片", "流程", "流程節點", "流程連線"}


def refs(value):
    return [x.strip() for x in re.split(r"[,，、\n]", value or "") if x.strip()]


def pages(value):
    result = set()
    for part in re.split(r"[,，、]", value or ""):
        if not re.fullmatch(r"\s*\d+(?:\s*[-–]\s*\d+)?\s*", part):
            return set()
        span = [int(x) for x in re.split(r"[-–]", part)]
        start, end = span[0], span[-1]
        if start < 1 or end < start or end > 1000:
            return set()
        result.update(range(start, end + 1))
    return result


def validate_draft(document):
    issues, seen = [], set()
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return ["缺少段落"]
    for b in blocks:
        if (not isinstance(b, dict) or b.get("kind") not in KINDS
                or not isinstance(b.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", b["id"])
                or not isinstance(b.get("fields"), dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in b["fields"].items())):
            issues.append("段落結構無效")
            continue
        if b["id"] in seen:
            issues.append(b["id"] + "：編號重複")
        seen.add(b["id"])
    return issues


def validate_publication(document):
    issues = validate_draft(document)
    if issues:
        return issues
    blocks = document["blocks"]
    by_kind = lambda kind: [b for b in blocks if b["kind"] == kind]
    if len(by_kind("文件")) != 1:
        issues.append("必須有且只有一個文件段落")
    for kind in ["來源", "術語", "規則", "涵蓋"]:
        if not by_kind(kind):
            issues.append("缺少" + kind)
    required = {
        "文件": ["遊戲", "版本"], "來源": ["名稱", "版本", "PDF頁數"], "術語": ["中文", "原文"],
        "規則": ["大項", "標題", "角色", "人數", "擴充", "內容", "條件", "例外", "來源", "PDF頁碼", "原文", "狀態"],
        "FAQ": ["問題", "答案", "引用", "角色", "人數", "擴充", "狀態"],
        "涵蓋": ["章節", "來源", "PDF頁碼", "狀態"],
        "圖片": ["名稱", "替代文字", "來源", "PDF頁碼", "資產", "相關規則", "狀態"],
    }
    sources = {b["id"]: b["fields"] for b in by_kind("來源")}
    rules = {b["id"]: b["fields"] for b in by_kind("規則")}
    for b in blocks:
        f, kind = b["fields"], b["kind"]
        add = lambda message: issues.append(b["id"] + "：" + message)
        for key in required.get(kind, []):
            if not f.get(key, "").strip():
                add("缺少" + key)
        if any(re.search("未確認|請填|請貼", v) for v in f.values()):
            add("仍有未確認內容")
        if kind in {"規則", "FAQ", "圖片"} and f.get("狀態") != "已確認":
            add("尚未人工確認")
        if f.get("網址") and not f["網址"].startswith("https://"):
            add("來源網址必須以 https:// 開頭")
        if kind == "規則":
            if f.get("大項") not in {"開始遊戲", "進行遊戲", "卡牌與配件", "計分與結束", "例外與 FAQ", "其他規則"}:
                add("大項無效")
            if not re.search(r"[\u3400-\u9fff]", f.get("標題", "")):
                add("標題須包含中文")
        if kind == "來源" and (not f.get("PDF頁數", "").isdigit() or not 1 <= int(f["PDF頁數"]) <= 1000):
            add("PDF頁數無效")
        if kind in {"規則", "涵蓋", "圖片"}:
            source = sources.get(f.get("來源"))
            span = pages(f.get("PDF頁碼"))
            total = source.get("PDF頁數", "") if source else ""
            if not span or not total.isdigit() or max(span) > int(total):
                add("來源或頁碼無效")
        linked = f.get("引用") if kind == "FAQ" else f.get("規則") if kind == "涵蓋" else f.get("相關規則")
        if any(r not in rules for r in refs(linked)):
            add("引用的規則不存在")
        if kind == "涵蓋":
            if f.get("狀態") not in {"已整理", "不適用"}:
                add("章節尚未整理")
            if f.get("狀態") == "已整理" and not refs(f.get("規則")):
                add("須列出規則")
            if f.get("狀態") == "不適用" and not f.get("說明"):
                add("須說明不適用原因")
            for rid in refs(f.get("規則")):
                rule = rules.get(rid)
                if rule and (rule.get("來源") != f.get("來源") or not pages(rule.get("PDF頁碼")).intersection(pages(f.get("PDF頁碼")))):
                    add("涵蓋規則不屬於此來源或頁面")
        if kind == "圖片":
            if not re.fullmatch(r"[a-f0-9]{64}", f.get("資產", "")) or len(pages(f.get("PDF頁碼"))) != 1:
                add("圖片資產或頁碼無效")
            if f.get("公開") not in {"是", "否"} or not f.get("順序", "").isdigit():
                add("圖片公開狀態或順序無效")
    for sid, source in sources.items():
        count = source.get("PDF頁數", "")
        covered = set().union(*(pages(b["fields"].get("PDF頁碼")) for b in by_kind("涵蓋") if b["fields"].get("來源") == sid))
        if count.isdigit() and 1 <= int(count) <= 1000 and not set(range(1, int(count) + 1)).issubset(covered):
            issues.append(sid + "：涵蓋表未交代所有頁面")
    from src.editorial_flow import validate_flows
    return issues + validate_flows(document)
