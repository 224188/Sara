"""SARA Plugin Hub foundation.

This module defines a safe integration registry and approval-first control plane.
It does not perform OAuth, send messages, publish content, or call third-party APIs.
Real provider adapters must be added separately and must require explicit Boss approval.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
import os

router = APIRouter(prefix="/plugins", tags=["Plugin Hub"])


class PluginCategory(str, Enum):
    AI_RESEARCH = "ai_research"
    EMAIL_CALENDAR = "email_calendar"
    DOCUMENTS = "documents"
    DEVELOPER_TOOLS = "developer_tools"
    COMMUNICATION = "communication"
    FINANCE_TRADINGVIEW = "finance_tradingview"
    PRODUCTIVITY = "productivity"
    MEDIA = "media"
    TRAVEL = "travel"
    AUTOMATION = "automation"


class PluginStatus(str, Enum):
    AVAILABLE = "available"
    INSTALLED = "installed"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    NEEDS_CONFIGURATION = "needs_configuration"


class PluginDefinition(BaseModel):
    plugin_id: str
    name: str
    category: PluginCategory
    description: str
    free_tier: bool
    requires_api_key: bool = False
    may_require_paid_plan: bool = False
    status: PluginStatus = PluginStatus.AVAILABLE
    enabled: bool = False
    permissions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# Catalog entries are metadata only. "Connected" is never claimed until a real
# provider adapter completes and verifies its OAuth/API handshake.
CATALOG = [
    PluginDefinition(plugin_id="gemini", name="Google Gemini", category=PluginCategory.AI_RESEARCH,
        description="SARA's configured AI provider; separate from optional research connectors.",
        free_tier=True, requires_api_key=True, permissions=["generate_content"],
        limitations=["Free-tier quotas and model availability are controlled by Google."]),
    PluginDefinition(plugin_id="gmail", name="Gmail", category=PluginCategory.EMAIL_CALENDAR,
        description="Email integration (OAuth adapter not yet implemented).", free_tier=True,
        permissions=["read_email", "draft_email", "send_email"],
        limitations=["Google OAuth consent and API quota required; sending always needs Boss approval."]),
    PluginDefinition(plugin_id="google_calendar", name="Google Calendar", category=PluginCategory.EMAIL_CALENDAR,
        description="Calendar integration (OAuth adapter not yet implemented).", free_tier=True,
        permissions=["read_events", "create_events", "modify_events"],
        limitations=["Google OAuth consent required; create/modify actions require Boss approval."]),
    PluginDefinition(plugin_id="github", name="GitHub", category=PluginCategory.DEVELOPER_TOOLS,
        description="Repository and pull-request integration (adapter not yet implemented).", free_tier=True,
        permissions=["read_repository", "write_branch", "create_pull_request"],
        limitations=["GitHub permissions and API limits apply; merge, push to protected branches, and repo changes require Boss approval."]),
    PluginDefinition(plugin_id="tradingview", name="TradingView", category=PluginCategory.FINANCE_TRADINGVIEW,
        description="Alert ingestion concept; direct account integration not implemented.", free_tier=False,
        permissions=["receive_alerts"],
        limitations=["TradingView plan/webhook availability may depend on subscription; no trading orders are supported."]),
    PluginDefinition(plugin_id="whatsapp_business", name="WhatsApp Business", category=PluginCategory.COMMUNICATION,
        description="Messaging integration (provider adapter not implemented).", free_tier=False,
        requires_api_key=True, may_require_paid_plan=True, permissions=["read_messages", "send_messages"],
        limitations=["Meta Business verification, templates, and conversation charges may apply; sending requires Boss approval."]),
    PluginDefinition(plugin_id="google_drive", name="Google Drive", category=PluginCategory.DOCUMENTS,
        description="Cloud file listing and document access (OAuth adapter not implemented).", free_tier=True,
        permissions=["list_files", "read_files", "create_files"],
        limitations=["Google OAuth and storage quotas apply; file edits require Boss approval."]),
    PluginDefinition(plugin_id="google_docs", name="Google Docs", category=PluginCategory.DOCUMENTS,
        description="Document creation and editing (OAuth adapter not implemented).", free_tier=True,
        permissions=["read_documents", "create_documents", "edit_documents"],
        limitations=["Google OAuth required; document changes require Boss approval."]),
    PluginDefinition(plugin_id="slack", name="Slack", category=PluginCategory.COMMUNICATION,
        description="Workspace messaging integration (adapter not implemented).", free_tier=True,
        permissions=["read_channels", "draft_messages", "send_messages"],
        limitations=["Workspace admin approval and plan limits may apply; sending requires Boss approval."]),
    PluginDefinition(plugin_id="telegram", name="Telegram Bot", category=PluginCategory.COMMUNICATION,
        description="Bot-based notifications (adapter not implemented).", free_tier=True,
        requires_api_key=True, permissions=["send_notifications"],
        limitations=["A Telegram bot token is required; outbound messages require Boss approval."]),
    PluginDefinition(plugin_id="notion", name="Notion", category=PluginCategory.PRODUCTIVITY,
        description="Notes and workspace pages (adapter not implemented).", free_tier=True,
        requires_api_key=True, permissions=["read_pages", "create_pages", "edit_pages"],
        limitations=["Notion integration token and workspace sharing required; edits require Boss approval."]),
    PluginDefinition(plugin_id="todoist", name="Todoist", category=PluginCategory.PRODUCTIVITY,
        description="Task and reminder management (adapter not implemented).", free_tier=True,
        requires_api_key=True, permissions=["read_tasks", "create_tasks", "complete_tasks"],
        limitations=["Account plan and API limits apply; task changes require Boss approval."]),
    PluginDefinition(plugin_id="youtube", name="YouTube", category=PluginCategory.MEDIA,
        description="Video metadata and draft workflow (adapter not implemented).", free_tier=True,
        permissions=["read_channel", "prepare_upload"],
        limitations=["YouTube API quota applies; publishing is disabled until explicit Boss approval."]),
    PluginDefinition(plugin_id="spotify", name="Spotify", category=PluginCategory.MEDIA,
        description="Music library and playback controls (adapter not implemented).", free_tier=False,
        permissions=["read_library", "control_playback"],
        limitations=["Spotify plan and developer access requirements may apply."]),
    PluginDefinition(plugin_id="travel_search", name="Travel Search", category=PluginCategory.TRAVEL,
        description="Flight and hotel search connector placeholder; no booking adapter.", free_tier=True,
        permissions=["search_flights", "search_hotels"],
        limitations=["Search provider coverage and fares vary; bookings are not supported."]),
    PluginDefinition(plugin_id="automation_rules", name="SARA Automation Rules", category=PluginCategory.AUTOMATION,
        description="Local rule registry foundation (scheduler and action runner not implemented).", free_tier=True,
        permissions=["create_rules", "pause_rules"],
        limitations=["Rules remain inactive until a secure scheduler and approval checks are implemented."])
]


class InstallRequest(BaseModel):
    plugin_id: str


class EnableRequest(BaseModel):
    enabled: bool


class ApprovalRequest(BaseModel):
    action: str
    details: dict[str, Any] = Field(default_factory=dict)


# In-memory fallback keeps the scaffold usable without MongoDB. Production must
# inject persistent storage and a real authenticated Boss identity.
_state: dict[str, dict[str, Any]] = {}
_activity: list[dict[str, Any]] = []
_pending_approvals: dict[str, dict[str, Any]] = {}


def _boss_guard(x_sara_boss_key: str | None) -> None:
    expected = os.getenv("SARA_BOSS_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Plugin Hub is not configured: SARA_BOSS_API_KEY missing")
    if not x_sara_boss_key or x_sara_boss_key != expected:
        raise HTTPException(status_code=401, detail="Boss authentication required")


def _record(event: str, plugin_id: str | None = None, details: dict[str, Any] | None = None) -> None:
    _activity.append({
        "id": str(uuid4()), "event": event, "plugin_id": plugin_id,
        "details": details or {}, "timestamp": datetime.now(timezone.utc).isoformat()
    })
    del _activity[:-200]


@router.get("/catalog")
async def get_catalog():
    """Public metadata only; never exposes credentials or claims live connectivity."""
    result = []
    for item in CATALOG:
        data = item.model_dump()
        data.update(_state.get(item.plugin_id, {}))
        result.append(data)
    return {"plugins": result}


@router.post("/install")
async def install_plugin(req: InstallRequest, x_sara_boss_key: str | None = Header(default=None)):
    _boss_guard(x_sara_boss_key)
    plugin = next((p for p in CATALOG if p.plugin_id == req.plugin_id), None)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Unknown plugin")
    _state.setdefault(req.plugin_id, {}).update({
        "status": PluginStatus.NEEDS_CONFIGURATION.value, "enabled": False
    })
    _record("plugin_installed", req.plugin_id)
    return {"plugin_id": req.plugin_id, "status": "needs_configuration", "enabled": False,
            "note": "Metadata installed only; provider connection is not yet implemented."}


@router.patch("/{plugin_id}/enabled")
async def set_plugin_enabled(plugin_id: str, req: EnableRequest,
                             x_sara_boss_key: str | None = Header(default=None)):
    _boss_guard(x_sara_boss_key)
    if plugin_id not in {p.plugin_id for p in CATALOG}:
        raise HTTPException(status_code=404, detail="Unknown plugin")
    current = _state.get(plugin_id, {})
    if current.get("status") != PluginStatus.CONNECTED.value and req.enabled:
        raise HTTPException(status_code=409, detail="Connect and verify this provider before enabling it")
    current["enabled"] = req.enabled
    _state[plugin_id] = current
    _record("plugin_enabled" if req.enabled else "plugin_disabled", plugin_id)
    return {"plugin_id": plugin_id, "enabled": req.enabled}


@router.post("/{plugin_id}/approvals")
async def request_approval(plugin_id: str, req: ApprovalRequest,
                            x_sara_boss_key: str | None = Header(default=None)):
    _boss_guard(x_sara_boss_key)
    if plugin_id not in {p.plugin_id for p in CATALOG}:
        raise HTTPException(status_code=404, detail="Unknown plugin")
    approval_id = str(uuid4())
    _pending_approvals[approval_id] = {
        "id": approval_id, "plugin_id": plugin_id, "action": req.action,
        "details": req.details, "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    _record("approval_requested", plugin_id, {"approval_id": approval_id, "action": req.action})
    return {"approval": _pending_approvals[approval_id],
            "note": "Request recorded only. No external action was executed."}


@router.get("/approvals")
async def list_approvals(x_sara_boss_key: str | None = Header(default=None)):
    _boss_guard(x_sara_boss_key)
    return {"approvals": list(_pending_approvals.values())}


@router.get("/activity")
async def get_activity(limit: int = 50, x_sara_boss_key: str | None = Header(default=None)):
    _boss_guard(x_sara_boss_key)
    return {"activity": list(reversed(_activity[-max(1, min(limit, 200)):]))}
