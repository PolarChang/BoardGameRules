"""Optional reviewed flow blocks are versioned with their rule document."""
import re

FLOW_KINDS = {"流程", "流程節點", "流程連線"}


def validate_flows(document):
    from src.editorial_validation import refs
    issues = []
    blocks = document["blocks"]
    flows = [b for b in blocks if b["kind"] == "流程"]
    rule_ids = {b["id"] for b in blocks if b["kind"] == "規則"}

    def add(identifier, message):
        issues.append(identifier + "：" + message)

    for b in [b for b in blocks if b["kind"] in FLOW_KINDS]:
        f, kind = b["fields"], b["kind"]
        required = {"流程": ["名稱", "範圍", "起點", "狀態"],
                    "流程節點": ["流程", "標題", "順序", "規則", "角色", "人數", "擴充", "結束", "狀態"],
                    "流程連線": ["流程", "起點", "終點", "條件", "順序", "循環", "狀態"]}[kind]
        for key in required:
            if not f.get(key, "").strip():
                add(b["id"], "缺少" + key)
        if f.get("狀態") != "已確認":
            add(b["id"], "流程尚未人工確認，不可發布")
        if kind == "流程":
            if f.get("範圍") not in {"總覽", "回合"}:
                add(b["id"], "流程範圍須為總覽或回合")
        else:
            if not any(flow["id"] == f.get("流程") for flow in flows):
                add(b["id"], "所屬流程不存在")
            if not re.fullmatch(r"[0-9]+", f.get("順序", "")) or len(f.get("順序", "")) > 5 or int(f["順序"]) > 10000:
                add(b["id"], "流程順序須為 0–10000 的整數")
            flag = "結束" if kind == "流程節點" else "循環"
            if f.get(flag) not in {"是", "否"}:
                add(b["id"], flag + "須為是或否")
            if kind == "流程節點" and any(r not in rule_ids for r in refs(f.get("規則"))):
                add(b["id"], "流程引用的規則不存在於此版本")

    for flow in flows:
        nodes = [b for b in blocks if b["kind"] == "流程節點" and b["fields"].get("流程") == flow["id"]]
        edges = [b for b in blocks if b["kind"] == "流程連線" and b["fields"].get("流程") == flow["id"]]
        node_ids = {b["id"] for b in nodes}
        if not nodes:
            add(flow["id"], "流程至少需要一個節點")
        if len(nodes) > 200 or len(edges) > 400:
            add(flow["id"], "單一流程最多 200 節點及 400 連線")
            continue
        if flow["fields"].get("起點") not in node_ids:
            add(flow["id"], "流程起點不存在於此流程")
        orders = [n["fields"].get("順序", "").lstrip("0") or "0" for n in nodes]
        if len(set(orders)) != len(nodes):
            add(flow["id"], "節點順序不可重複")
        valid_edges = []
        for edge in edges:
            if edge["fields"].get("起點") not in node_ids or edge["fields"].get("終點") not in node_ids:
                add(edge["id"], "連線端點不存在於此流程")
            else:
                valid_edges.append(edge)

        def walk(starts, reverse=False):
            seen, queue = set(), list(starts)
            while queue:
                identifier = queue.pop()
                if identifier in seen:
                    continue
                seen.add(identifier)
                for e in valid_edges:
                    if e["fields"]["終點" if reverse else "起點"] == identifier:
                        queue.append(e["fields"]["起點" if reverse else "終點"])
            return seen

        reachable = walk([flow["fields"].get("起點")])
        endings = [n["id"] for n in nodes if n["fields"].get("結束") == "是"]
        if not endings:
            add(flow["id"], "流程須有結束節點")
        can_end = walk(endings, True)
        for node in nodes:
            if node["id"] not in reachable:
                add(node["id"], "節點無法從流程起點到達")
            if node["id"] not in can_end:
                add(node["id"], "節點沒有通往結束的路徑")
            outgoing = [e for e in valid_edges if e["fields"]["起點"] == node["id"]]
            if node["fields"].get("結束") == "是" and outgoing:
                add(node["id"], "結束節點不可有離開連線")
            if len(outgoing) > 1 and len({e["fields"].get("條件", "").strip() for e in outgoing}) != len(outgoing):
                add(node["id"], "分支條件不可重複")
        remaining = [e for e in valid_edges if e["fields"].get("循環") != "是"]
        indegree = {n["id"]: sum(e["fields"]["終點"] == n["id"] for e in remaining) for n in nodes}
        queue = [identifier for identifier, count in indegree.items() if count == 0]
        visited = 0
        while queue:
            identifier = queue.pop()
            visited += 1
            for e in remaining:
                if e["fields"]["起點"] == identifier:
                    target = e["fields"]["終點"]
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
        if visited != len(nodes):
            add(flow["id"], "循環連線須明確標記")
    return issues
