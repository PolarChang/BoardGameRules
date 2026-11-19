"""Moderated contributions with durable, cross-instance abuse controls.

All writes serialize through a database row lock (SQLite write lock / PostgreSQL
row lock). This is intentionally conservative for the initial small community.
"""
import hashlib
import asyncio
import hmac
import ipaddress
import json
import os
import logging
import secrets
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field
from src.database import _connect

router = APIRouter()
_cleanup_tasks = set()
LIMITS = {"contribution": (600, 3, 10, 2), "comment": (60, 5, 50, 10), "vote": (60, 20, 200, 20), "report": (600, 3, 10, 2)}


def expire_records(cur, now):
    cur.execute("DELETE FROM community_events WHERE created_at<?", (now - 7 * 86400,))
    cur.execute("UPDATE community_votes SET ip='' WHERE created_at<? AND ip<>''", (now - 7 * 86400,))
    cur.execute("DELETE FROM community_ip_blocks WHERE until_at<?", (now,))
    cur.execute("DELETE FROM editorial_requests WHERE created_at<?", (now - 86400,))


def cleanup_expired():
    with locked() as (conn, cur):
        expire_records(cur, time.time())


@router.on_event("startup")
async def start_cleanup():
    async def loop():
        # Parent application initializes schema during startup before this task runs.
        await asyncio.sleep(1)
        while True:
            try:
                await asyncio.to_thread(cleanup_expired)
            except Exception:
                logging.getLogger(__name__).warning("Community retention cleanup unavailable; next attempt in one hour")
            await asyncio.sleep(3600)
    task = asyncio.create_task(loop())
    _cleanup_tasks.add(task)
    task.add_done_callback(_cleanup_tasks.discard)


@router.on_event("shutdown")
async def stop_cleanup():
    tasks = list(_cleanup_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def init_community_db():
    conn = _connect()
    try:
        cur = conn.cursor()
        for sql in [
            "CREATE TABLE IF NOT EXISTS community_lock (id INTEGER PRIMARY KEY, marker INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS community_members (id TEXT PRIMARY KEY, created_at DOUBLE PRECISION NOT NULL, restricted INTEGER NOT NULL DEFAULT 0)",
            "CREATE TABLE IF NOT EXISTS community_events (id TEXT PRIMARY KEY, actor TEXT NOT NULL, ip TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL, counted INTEGER NOT NULL)",
            "CREATE INDEX IF NOT EXISTS community_events_time ON community_events (created_at)",
            "CREATE TABLE IF NOT EXISTS community_ip_blocks (ip TEXT PRIMARY KEY, until_at DOUBLE PRECISION NOT NULL)",
            "CREATE TABLE IF NOT EXISTS community_items (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, actor TEXT NOT NULL, kind TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL)",
            "CREATE TABLE IF NOT EXISTS community_votes (actor TEXT NOT NULL, target TEXT NOT NULL, ip TEXT NOT NULL, status TEXT NOT NULL, active INTEGER NOT NULL, created_at DOUBLE PRECISION NOT NULL, PRIMARY KEY(actor,target))",
            "CREATE TABLE IF NOT EXISTS community_audit (id TEXT PRIMARY KEY, actor TEXT NOT NULL, target TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL)",
        ]:
            cur.execute(sql)
        cur.execute("INSERT INTO community_lock (id,marker) VALUES (1,0) ON CONFLICT (id) DO NOTHING")
        conn.commit()
    finally:
        conn.close()


@contextmanager
def locked():
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("UPDATE community_lock SET marker=marker WHERE id=1")
        if cur.rowcount != 1:
            raise RuntimeError("Missing rate limiter lock")
        yield conn, cur
        conn.commit()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, {"code": "community_unavailable", "message": "社群暫停寫入，請保留文字稍後重試"})
    finally:
        if conn:
            conn.close()


def identity(authorization):
    if os.getenv("ENABLE_COMMUNITY", "false") != "true":
        raise HTTPException(503, {"code": "community_disabled", "message": "社群尚未開放"})
    client = os.getenv("COMMUNITY_GOOGLE_CLIENT_ID", "")
    if not client:
        raise HTTPException(503, "登入尚未設定")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "請先登入")
    try:
        from google.oauth2.id_token import verify_oauth2_token
        from google.auth.transport.requests import Request as GoogleRequest
        claims = verify_oauth2_token(authorization[7:], GoogleRequest(), client)
    except Exception:
        raise HTTPException(401, "登入已失效，請重新登入")
    # Google is authoritative for Gmail / Workspace addresses, not arbitrary third-party mail.
    if claims.get("email_verified") is not True or not (claims.get("hd") or claims.get("email", "").lower().endswith("@gmail.com")):
        raise HTTPException(403, "請使用已驗證的 Gmail 或 Google Workspace 帳號")
    return "google:" + claims["sub"]


def trusted_ip(request):
    key = os.getenv("COMMUNITY_PROXY_SECRET", "")
    value = request.headers.get("x-community-ip", "")
    stamp = request.headers.get("x-community-timestamp", "")
    signature = request.headers.get("x-community-signature", "")
    try:
        ip = str(ipaddress.ip_address(value))
        fresh = abs(time.time() - int(stamp)) <= 60
    except ValueError:
        fresh, ip = False, ""
    expected = hmac.new(key.encode(), (stamp + "\n" + value).encode(), hashlib.sha256).hexdigest()
    if not key or not fresh or not hmac.compare_digest(expected, signature):
        raise HTTPException(403, "無法驗證連線來源")
    return hmac.new(key.encode(), ip.encode(), hashlib.sha256).hexdigest()


def challenge_valid(token):
    secret, hostname = os.getenv("TURNSTILE_SECRET_KEY", ""), os.getenv("COMMUNITY_HOSTNAME", "")
    if not secret or not hostname or not token:
        return False
    try:
        import requests
        result = requests.post("https://challenges.cloudflare.com/turnstile/v0/siteverify", data={"secret": secret, "response": token}, timeout=10).json()
        return result.get("success") is True and result.get("hostname") == hostname and result.get("action") == "community"
    except Exception:
        return False


def count(cur, sql, params):
    cur.execute(sql, params)
    return cur.fetchone()["n"]


def begin_write(conn, cur, actor, ip, action, target, now, challenge=False):
    # Expire IP-bearing records. UTC daily quotas remain exact within the retained window.
    expire_records(cur, now)
    cur.execute("INSERT INTO community_members (id,created_at) VALUES (?,?) ON CONFLICT (id) DO NOTHING", (actor, now))
    cur.execute("SELECT * FROM community_members WHERE id=?", (actor,))
    member = cur.fetchone()
    if member["restricted"]:
        raise HTTPException(403, "此帳號暫停社群寫入")
    cur.execute("SELECT until_at FROM community_ip_blocks WHERE ip=?", (ip,))
    blocked = cur.fetchone()
    if blocked:
        raise HTTPException(429, {"code": "ip_rate_limited", "message": "此連線暫停寫入"}, headers={"Retry-After": str(max(1, int(blocked["until_at"] - now)))})
    event_id = secrets.token_hex(16)
    cur.execute("INSERT INTO community_events (id,actor,ip,action,target,created_at,counted) VALUES (?,?,?,?,?,?,0)", (event_id, actor, ip, action, target, now))
    attempts = count(cur, "SELECT COUNT(*) AS n FROM community_events WHERE ip=? AND created_at>?", (ip, now - 600))
    if attempts > 300:
        cur.execute("INSERT INTO community_ip_blocks (ip,until_at) VALUES (?,?) ON CONFLICT (ip) DO UPDATE SET until_at=excluded.until_at", (ip, now + 600))
        conn.commit()
        raise HTTPException(429, {"code": "ip_rate_limited", "message": "此連線暫停寫入 10 分鐘"}, headers={"Retry-After": "600"})
    if attempts > 100 and not challenge:
        conn.commit()
        raise HTTPException(403, {"code": "challenge_required", "message": "請完成機器人驗證後重試"})
    return event_id, now - member["created_at"] < 86400


def charge(conn, cur, event_id, actor, action, now, new):
    span, short, daily, newcomer = LIMITS[action]
    short = int(os.getenv(f"COMMUNITY_{action.upper()}_BURST", str(short)))
    maximum = int(os.getenv(f"COMMUNITY_{action.upper()}_{'NEW_' if new else ''}DAILY", str(newcomer if new else daily)))
    day_start = int(now // 86400) * 86400
    recent = count(cur, "SELECT COUNT(*) AS n FROM community_events WHERE actor=? AND action=? AND counted=1 AND created_at>?", (actor, action, now - span))
    today = count(cur, "SELECT COUNT(*) AS n FROM community_events WHERE actor=? AND action=? AND counted=1 AND created_at>=?", (actor, action, day_start))
    if recent >= short or today >= maximum:
        conn.commit()
        retry = int(day_start + 86400 - now) if today >= maximum else span
        raise HTTPException(429, {"code": "account_rate_limited", "message": "已達此操作額度，文字仍可保留"}, headers={"Retry-After": str(max(1, retry))})
    cur.execute("UPDATE community_events SET counted=1 WHERE id=?", (event_id,))


class Submission(BaseModel):
    content: str = Field(min_length=2, max_length=4000)
    target: str = Field(default="", max_length=100)
    challenge_token: str = Field(default="", max_length=2048)


class Vote(BaseModel):
    active: bool = True
    challenge_token: str = Field(default="", max_length=2048)


def submit(game_id, actor, ip, kind, body, now=None, challenge=False):
    now = now or time.time()
    fingerprint = hashlib.sha256(" ".join(unicodedata.normalize("NFKC", body.content).lower().split()).encode()).hexdigest()
    with locked() as (conn, cur):
        event, new = begin_write(conn, cur, actor, ip, kind, body.target, now, challenge)
        if kind == "comment":
            cur.execute("SELECT id FROM community_items WHERE id=? AND game_id=? AND kind='contribution' AND status='visible'", (body.target, game_id))
            if not cur.fetchone():
                raise HTTPException(404, "找不到可留言的補充")
        if kind == "report":
            cur.execute("SELECT id FROM community_items WHERE id=? AND game_id=? AND kind IN ('comment','contribution')", (body.target, game_id))
            if not cur.fetchone():
                raise HTTPException(404, "找不到檢舉對象")
        cur.execute("SELECT id,status FROM community_items WHERE actor=? AND game_id=? AND kind=? AND target=? AND fingerprint=? AND created_at>? ORDER BY created_at DESC LIMIT 1", (actor, game_id, kind, body.target, fingerprint, now - 86400))
        duplicate = cur.fetchone()
        if duplicate:
            return {**dict(duplicate), "duplicate": True}
        charge(conn, cur, event, actor, kind, now, new)
        identifier = secrets.token_hex(16)
        status = "pending" if kind in {"contribution", "report"} else "visible"
        cur.execute("INSERT INTO community_items (id,game_id,actor,kind,target,content,fingerprint,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)", (identifier, game_id, actor, kind, body.target, body.content, fingerprint, status, now))
        return {"id": identifier, "status": status, "duplicate": False}


def vote(game_id, actor, ip, target, active, now=None, challenge=False):
    now = now or time.time()
    with locked() as (conn, cur):
        event, new = begin_write(conn, cur, actor, ip, "vote", target, now, challenge)
        cur.execute("SELECT actor FROM community_items WHERE id=? AND game_id=? AND kind='contribution' AND status='visible'", (target, game_id))
        item = cur.fetchone()
        if not item:
            raise HTTPException(404, "找不到補充")
        if item["actor"] == actor:
            raise HTTPException(403, "不能替自己的投稿投票")
        cur.execute("SELECT * FROM community_votes WHERE actor=? AND target=?", (actor, target))
        old = cur.fetchone()
        if old and bool(old["active"]) == active:
            return {"status": old["status"], "active": active}
        if not old and not active:
            return {"status": "counted", "active": False}
        charge(conn, cur, event, actor, "vote", now, new)
        status = old["status"] if old and old["status"] in {"void", "pending"} else "counted"
        cur.execute("INSERT INTO community_votes (actor,target,ip,status,active,created_at) VALUES (?,?,?,?,?,?) ON CONFLICT (actor,target) DO UPDATE SET ip=excluded.ip,status=excluded.status,active=excluded.active,created_at=excluded.created_at", (actor, target, ip, status, int(active), now))
        group = count(cur, "SELECT COUNT(DISTINCT actor) AS n FROM community_events WHERE ip=? AND target=? AND action='vote' AND counted=1 AND created_at>?", (ip, target, now - 600))
        if group >= 5:
            cur.execute("UPDATE community_votes SET status='pending' WHERE target=? AND ip=? AND created_at>? AND status<>'void'", (target, ip, now - 600))
            if status != "void":
                status = "pending"
        return {"status": status, "active": active}


@router.get("/community")
def listing(game_id: str, target: str = "", before: float = 1e20):
    conn = _connect()
    try:
        cur = conn.cursor()
        if target:
            cur.execute("SELECT id FROM community_items WHERE game_id=? AND id=? AND kind='contribution' AND status='visible'", (game_id, target))
            if not cur.fetchone():
                return {"items": []}
        cur.execute("SELECT id,kind,target,content,status,created_at FROM community_items WHERE game_id=? AND kind IN ('contribution','comment') AND status='visible' AND target=? AND created_at<? ORDER BY created_at DESC LIMIT 50", (game_id, target, before))
        result = [dict(row) for row in cur.fetchall()]
        for item in result:
            item["votes"] = count(cur, "SELECT COUNT(*) AS n FROM community_votes WHERE target=? AND status='counted' AND active=1", (item["id"],))
        return {"items": result}
    finally:
        conn.close()


@router.post("/community/contributions")
def contribution(game_id: str, body: Submission, request: Request, authorization: str = Header(default="")):
    from src.editorial_api import check_game
    check_game(game_id)
    actor = identity(authorization)
    return submit(game_id, actor, trusted_ip(request), "contribution", body, challenge=challenge_valid(body.challenge_token))


@router.post("/community/comments")
def comment(game_id: str, body: Submission, request: Request, authorization: str = Header(default="")):
    actor = identity(authorization)
    return submit(game_id, actor, trusted_ip(request), "comment", body, challenge=challenge_valid(body.challenge_token))


@router.put("/community/contributions/{target}/vote")
def cast_vote(game_id: str, target: str, body: Vote, request: Request, authorization: str = Header(default="")):
    actor = identity(authorization)
    return vote(game_id, actor, trusted_ip(request), target, body.active, challenge=challenge_valid(body.challenge_token))


@router.post("/community/reports")
def report(game_id: str, body: Submission, request: Request, authorization: str = Header(default="")):
    actor = identity(authorization)
    return submit(game_id, actor, trusted_ip(request), "report", body, challenge=challenge_valid(body.challenge_token))


class Review(BaseModel):
    target: str = Field(max_length=100)
    actor: str = Field(default="", max_length=200)
    decision: Literal["visible", "hidden", "rejected", "counted", "void", "restrict", "restore"]
    reason: str = Field(min_length=2, max_length=1000)


@router.get("/community/review")
def review_queue(game_id: str, x_knowledge_admin: str = Header(default="")):
    from src.editorial_api import admin
    admin(x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM community_items WHERE game_id=? ORDER BY created_at DESC LIMIT 100", (game_id,))
        items = [dict(row) for row in cur.fetchall()]
        cur.execute("SELECT v.actor,v.target,v.status,v.active FROM community_votes v JOIN community_items i ON i.id=v.target WHERE i.game_id=? AND v.status='pending' LIMIT 100", (game_id,))
        return {"items": items, "votes": [dict(row) for row in cur.fetchall()]}
    finally:
        conn.close()


@router.post("/community/review")
def review(game_id: str, body: Review, x_knowledge_admin: str = Header(default="")):
    from src.editorial_api import admin
    admin(x_knowledge_admin)
    with locked() as (conn, cur):
        cur.execute("SELECT id FROM community_items WHERE id=? AND game_id=?", (body.target, game_id))
        if not cur.fetchone():
            raise HTTPException(404, "找不到投稿")
        if body.decision in {"counted", "void"}:
            cur.execute("UPDATE community_votes SET status=? WHERE target=? AND actor=?", (body.decision, body.target, body.actor))
        elif body.decision in {"restrict", "restore"}:
            cur.execute("UPDATE community_members SET restricted=? WHERE id=?", (int(body.decision == "restrict"), body.actor))
        else:
            cur.execute("UPDATE community_items SET status=? WHERE id=?", (body.decision, body.target))
        if cur.rowcount != 1:
            raise HTTPException(404, "找不到處置對象")
        cur.execute("INSERT INTO community_audit (id,actor,target,decision,reason,created_at) VALUES (?,?,?,?,?,?)", (secrets.token_hex(16), "legacy-admin", body.target + ":" + body.actor, body.decision, body.reason, time.time()))
        return {"status": body.decision}
