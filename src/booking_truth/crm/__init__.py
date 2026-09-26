"""CRM adapters: a ``CrmAdapter`` protocol with validated payload models, and its implementations."""

from __future__ import annotations

import importlib
from typing import Any

from booking_truth.config import Settings
from booking_truth.crm.base import (
    OUTCOME_FOR_ACTION,
    ContactPayload,
    CrmAdapter,
    CrmError,
    CrmOk,
    CrmResult,
    CrmSyncPayload,
    MeetingOutcome,
    MeetingPayload,
    MeetingUpdatePayload,
    SyncAction,
)
from booking_truth.crm.null import NullCrm

HUBSPOT_MODULE = "booking_truth.crm.hubspot"


def build_crm(settings: Settings) -> tuple[CrmAdapter, str | None]:
    """The adapter for ``BT_CRM`` and a note for ``/healthz`` when it had to fall back.

    ``BT_CRM=hubspot`` needs the HubSpot adapter module; an installation without it falls back to
    :class:`NullCrm` and says so, instead of failing to start.
    """
    if settings.crm == "none":
        return NullCrm(), None
    try:
        module = importlib.import_module(HUBSPOT_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != HUBSPOT_MODULE:
            raise
        return (
            NullCrm(),
            "BT_CRM=hubspot is set but this installation has no HubSpot adapter; CRM writes are off",
        )
    factory: Any = getattr(module, "build_hubspot", None)
    if not callable(factory):
        return NullCrm(), f"{HUBSPOT_MODULE} has no build_hubspot(); CRM writes are off"
    adapter: CrmAdapter = factory(settings)
    return adapter, None


__all__ = [
    "OUTCOME_FOR_ACTION",
    "ContactPayload",
    "CrmAdapter",
    "CrmError",
    "CrmOk",
    "CrmResult",
    "CrmSyncPayload",
    "MeetingOutcome",
    "MeetingPayload",
    "MeetingUpdatePayload",
    "NullCrm",
    "SyncAction",
    "build_crm",
]
