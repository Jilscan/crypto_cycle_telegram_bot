from pathlib import Path
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import httpx
import yaml
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tiny HTTP health-check server for Koyeb Web Service (free plan needs this)
# Koyeb sets PORT for web services. We answer / and /health with "ok".
# ─────────────────────────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "0"))
if PORT:
    from aiohttp import web

    async def ping(request):
        return web.Response(text="ok")

    health_app = web.Application()
    health_app.add_routes([web.get("/", ping), web.get("/health", ping)])
    asyncio.get_event_loop().create_task(
        web._run_app(health_app, host="0.0.0.0", port=PORT)
    )

# ─────────────────────────────────────────────────────────────────────────────
# Config / Models
# ─────────────────────────────────────────────────────────────────────────────


class Config(BaseModel):
    telegram: Dict[str, Any]
    schedule: Dict[str, Any]
    symbols: Dict[str, List[str]]
    thresholds: Dict[str, Any]  # back-compat if risk_profiles not used
    glassnode: Dict[str, Any] = {}
    force_chat_id: str = ""
    logging: Dict[str, Any] = {"level": "INFO"}
    default_profile: str = "moderate"
    risk_profiles: Dict[str, Dict[str, Any]] | None = None


def load_config(path: str = "config.yml") -> "Config":
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


CFG = load_config()

# Per-chat risk profile storage
PROFILE_FILE = Path("profiles.yml")
CHAT_PROFILE: dict[int, str] = {}


def load_profiles():
    global CHAT_PROFILE
    if PROFILE_FILE.exists():
        try:
            CHAT_PROFILE = yaml.safe_load(PROFILE_FILE.read_text()) or {}
        except Exception:
            CHAT_PROFILE = {}
    else:
        CHAT_PROFILE = {}


def save_profiles():
    try:
        PROFILE_FILE.write_text(yaml.safe_dump(CHAT_PROFILE, sort_keys=False))
    except Exception as e:
        log.warning("Failed to save profiles.yml: %s", e)


def get_thresholds_for_chat(chat_id: Optional[int]) -> Dict[str, Any]:
    if CFG.risk_profiles:
        profile = CFG.default_profile
        if chat_id is not None and chat_id in CHAT_PROFILE:
            profile = CHAT_PROFILE[chat_id]
        pro
