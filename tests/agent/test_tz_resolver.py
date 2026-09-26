"""``tz_resolver`` in the running agent: the pre-scan resolves or flags a zone the prospect states in
their own message before the model runs, states a resolved zone back, offers ``confirm_timezone`` quick
replies for an ambiguous one, and states back a browser hint while it is the only zone known."""

from __future__ import annotations

from typing import Any

from agent_env import LEAD, AgentEnv


def events(data: dict[str, Any], guard: str = "tz_resolver") -> list[tuple[str, str]]:
    return [(e["guard"], e["event"]) for e in data["guard"]["events"] if e["guard"] == guard]


def tz_replies(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [q for q in data["quick_replies"] if q["action"]["type"] == "confirm_timezone"]


# A resolved statement is stated back --------------------------------------------------------------------


async def test_a_resolved_statement_is_stated_back_and_stored(guarded: AgentEnv) -> None:
    data = await guarded.say("Hi, I'm in Kathmandu.")
    assert "I'll use Asia/Kathmandu (UTC+05:45) for times" in data["reply"]
    assert "tell me if that's wrong" in data["reply"]
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is not None
    assert (lead.tz_zone, lead.tz_source, lead.tz_confirmed) == ("Asia/Kathmandu", "stated", False)


async def test_the_statement_is_not_repeated_once_the_reply_already_names_the_zone(
    guarded: AgentEnv,
) -> None:
    """``find_slots``'s own offer text already says "(shown in Europe/Berlin)", so the code-appended
    statement line would be pure repetition and is skipped."""
    data = await guarded.say("Hi, I'm in Berlin, could I book an intro call next week?")
    assert "(shown in Europe/Berlin)" in data["reply"]
    assert data["reply"].count("Europe/Berlin") == 1


# An ambiguous statement offers quick replies, not a silent guess -----------------------------------------


async def test_an_ambiguous_statement_offers_confirm_timezone_quick_replies(guarded: AgentEnv) -> None:
    data = await guarded.say("Hi, I'm on IST. Evenings work best for me.")
    replies = tz_replies(data)
    zones = {r["action"]["zone"] for r in replies}
    assert zones == {"Asia/Kolkata", "Asia/Jerusalem", "Europe/Dublin"}
    assert all(r["action"]["type"] == "confirm_timezone" for r in replies)
    assert events(data) == [("tz_resolver", "prescan_ambiguous")]
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is None or lead.tz_zone is None


async def test_picking_a_confirm_timezone_quick_reply_confirms_it(guarded: AgentEnv) -> None:
    offer = await guarded.say("Hi, I'm on IST. Evenings work best for me.")
    zone = tz_replies(offer)[0]["action"]["zone"]
    data = await guarded.act({"type": "confirm_timezone", "zone": zone})
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is not None
    assert (lead.tz_zone, lead.tz_source, lead.tz_confirmed) == (zone, "confirmed", True)
    assert zone in data["reply"]


# A restated zone after confirmation --------------------------------------------------------------------


async def test_restating_the_confirmed_zone_keeps_it_confirmed(guarded: AgentEnv) -> None:
    """The pre-scan still states the zone back (the only documented suppression is a reply that
    already names it), but the stored state stays ``confirmed`` rather than reverting to ``stated``."""
    await guarded.act({"type": "confirm_timezone", "zone": "Europe/Berlin"})
    data = await guarded.say("Sorry, just to repeat: I'm in Berlin.", session="s-2")
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is not None
    assert (lead.tz_zone, lead.tz_source, lead.tz_confirmed) == ("Europe/Berlin", "confirmed", True)
    assert "I'll use Europe/Berlin (UTC+02:00) for times" in data["reply"]


async def test_stating_a_different_zone_after_confirmation_overrides_it(guarded: AgentEnv) -> None:
    await guarded.act({"type": "confirm_timezone", "zone": "Europe/Berlin"})
    data = await guarded.say("Actually, ignore that, I'm in Sydney.", session="s-2")
    lead = guarded.deps.store.leads.get(LEAD)
    assert lead is not None
    assert (lead.tz_zone, lead.tz_source, lead.tz_confirmed) == ("Australia/Sydney", "stated", False)
    assert "I'll use Australia/Sydney" in data["reply"]


# The browser hint is stated back while it is the only zone known -----------------------------------------


async def test_the_browser_hint_is_stated_back_while_it_is_still_the_only_zone(guarded: AgentEnv) -> None:
    data = await guarded.say("Hi there!", hint="America/Denver")
    assert "I'll use America/Denver (UTC-06:00) for times based on your browser" in data["reply"]


async def test_a_message_with_no_statement_and_no_hint_states_nothing_back(guarded: AgentEnv) -> None:
    data = await guarded.say("Hi there!")
    assert "I'll use" not in data["reply"]


# Off: no pre-scan runs at all -----------------------------------------------------------------------------


async def test_without_the_guard_nothing_is_stated_back_or_stored_from_the_prescan(
    naive: AgentEnv,
) -> None:
    data = await naive.say("Hi, I'm in Kathmandu, evenings work best for me.")
    assert "I'll use" not in data["reply"]
    assert data["quick_replies"] == []
