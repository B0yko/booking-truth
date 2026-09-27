"""The scripted persona: script steps, conditions, fallbacks, picks and the hidden-window check."""

from __future__ import annotations

import copy
import random
from datetime import UTC, date, datetime
from typing import Any

import pytest

from booking_truth.harness.adapters import AgentReply, Lead, Turn
from booking_truth.harness.hfaults import (
    DUPLICATE_DELAY_S,
    concurrent_message,
    deliver_concurrent,
    deliver_duplicate,
    duplicate_delay,
)
from booking_truth.harness.personas import (
    AgentView,
    Offer,
    PersonaError,
    ScriptedPersona,
    agent_asks_confirmation,
    agent_asks_timezone,
    quick_reply_offers,
    reply_offers,
    slot_label,
    text_offers,
)
from booking_truth.harness.scenarios import ResolvedScenario, Scenario, load_suite

# Thursday 1 October 2026, 08:00 in New York. The host-zone persona's window is 13:00-17:00 New York on
# Fri 2, Mon 5, Tue 6, Wed 7 and Thu 8 October (EDT, UTC-4).
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
IN_WINDOW = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)  # Mon 5 Oct, 2:00 PM New York
OUT_OF_WINDOW = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)  # Mon 5 Oct, 9:00 AM New York
SUITE = {s.id: s for s in load_suite()}


def resolved(scenario_id: str = "happy-book-host-zone", **changes: Any) -> ResolvedScenario:
    scenario = SUITE[scenario_id]
    if changes:
        raw = copy.deepcopy(scenario.model_dump(mode="json", exclude_unset=True))
        raw["persona"].update(changes)
        scenario = Scenario.model_validate(raw)
    return ResolvedScenario(scenario, date(2026, 10, 1), now=NOW)


def reply(
    text: str, quick: tuple[dict[str, Any], ...] = (), booking: dict[str, Any] | None = None
) -> AgentReply:
    return AgentReply(status=200, reply=text, quick_replies=quick, booking=booking)


def view(last: AgentReply | None, *history: str) -> AgentView:
    messages = [*history, *([last.reply] if last is not None and last.reply else [])]
    return AgentView(last_reply=last, agent_messages=messages, now=NOW)


def slot(start: datetime, slot_id: str, text: str = "") -> dict[str, Any]:
    return {
        "label": text or f"slot {slot_id}",
        "action": {"type": "select_slot", "slot_id": slot_id},
        "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


OFFERS_TEXT = (
    "I can do Monday 5 October at 9:00 AM or Monday 5 October at 2:00 PM Eastern. Which one works for you?"
)


async def test_the_first_turn_is_the_first_script_line() -> None:
    persona = ScriptedPersona(resolved())
    first = await persona.next_turn(view(None))
    assert first is not None
    assert first.kind == "say"
    assert first.text.startswith("Hi! I'd like to book a 30-minute intro call")
    assert persona.turns == 1


async def test_a_timezone_step_runs_when_the_agent_asks_for_the_zone() -> None:
    persona = ScriptedPersona(resolved())
    await persona.next_turn(view(None))
    answer = await persona.next_turn(view(reply("Happy to help! Which time zone are you in?")))
    assert answer is not None
    assert answer.text == "Eastern time, New York."
    assert persona.zone_answered


async def test_a_timezone_step_is_skipped_when_the_agent_does_not_ask() -> None:
    persona = ScriptedPersona(resolved(), supports_actions=True)
    await persona.next_turn(view(None))
    pick = await persona.next_turn(view(reply("Here you go.", (slot(IN_WINDOW, "s_1"),))))
    assert pick is not None
    assert pick.kind == "pick"


async def test_the_clarification_answers_a_zone_question_the_script_did_not_expect() -> None:
    persona = ScriptedPersona(resolved("happy-cancel"))
    await persona.next_turn(view(None))
    clarify = await persona.next_turn(view(reply("Sure. Where are you located, so I use the right zone?")))
    assert clarify is not None
    assert (clarify.kind, clarify.text) == ("clarify", "New York.")
    done = await persona.next_turn(view(reply("Your call is cancelled.")))
    assert done is not None
    assert (done.text, done.end) == ("Thanks.", True)
    assert await persona.next_turn(view(reply("You're welcome."))) is None


async def test_a_pick_takes_the_first_offer_inside_the_window_as_a_structured_action() -> None:
    persona = ScriptedPersona(resolved(), supports_actions=True)
    await persona.next_turn(view(None))
    offers = (
        slot(OUT_OF_WINDOW, "s_early", "Mon 5 Oct, 9:00 AM"),
        slot(IN_WINDOW, "s_late", "Mon 5 Oct, 2:00 PM"),
    )
    pick = await persona.next_turn(view(reply("Which works?", offers)))
    assert pick is not None
    assert pick.action == {"type": "select_slot", "slot_id": "s_late"}
    assert pick.text == "Mon 5 Oct, 2:00 PM"
    assert pick.offer is not None
    assert pick.offer.start_utc == IN_WINDOW
    assert [o.slot_id for o in pick.choices] == ["s_early", "s_late"]
    assert persona.picks == [pick.offer]


async def test_a_text_agent_gets_the_label_as_text() -> None:
    persona = ScriptedPersona(resolved(), supports_actions=False)
    await persona.next_turn(view(None))
    pick = await persona.next_turn(view(reply(OFFERS_TEXT)))
    assert pick is not None
    assert pick.action is None
    assert pick.text == "Monday 5 October at 2:00 PM works for me."


async def test_offers_without_a_slot_id_are_picked_by_label_even_for_the_bundled_protocol() -> None:
    persona = ScriptedPersona(resolved(), supports_actions=True)
    await persona.next_turn(view(None))
    pick = await persona.next_turn(view(reply(OFFERS_TEXT)))
    assert pick is not None
    assert pick.action is None
    assert pick.text.endswith("works for me.")


async def test_offers_outside_the_window_get_the_correction_then_the_persona_gives_up() -> None:
    persona = ScriptedPersona(resolved())
    await persona.next_turn(view(None))
    early = reply("How about this?", (slot(OUT_OF_WINDOW, "s_1"),))
    texts = []
    for _ in range(3):
        turn = await persona.next_turn(view(early))
        assert turn is not None
        assert turn.kind == "correction"
        texts.append(turn.text)
    assert texts[0] == (
        "Could we do an afternoon between Friday 2 October and Thursday 8 October, between 1 and 5 pm "
        "New York time?"
    )
    assert await persona.next_turn(view(early)) is None
    assert persona.gave_up
    assert persona.turns == 4


async def test_no_offer_yet_asks_for_times_in_the_window() -> None:
    persona = ScriptedPersona(resolved(correction=None))
    await persona.next_turn(view(None))
    ask = await persona.next_turn(view(reply("Let me check the calendar.")))
    assert ask is not None
    assert ask.kind == "ask"
    assert ask.text == (
        "What times do you have between Friday 2 October and Thursday 8 October? Something from 1 pm to 5 pm "
        "my time would be ideal."
    )


async def test_default_correction_when_the_scenario_has_none() -> None:
    persona = ScriptedPersona(resolved(correction=None))
    await persona.next_turn(view(None))
    turn = await persona.next_turn(view(reply("Try this.", (slot(OUT_OF_WINDOW, "s_1"),))))
    assert turn is not None
    assert turn.text.startswith("None of those times work for me. What times do you have")


async def test_accepting_an_offer_outside_the_window_is_a_persona_error() -> None:
    script = [{"say": "Hi, book me a call."}, {"pick": "offered[0]"}, {"say": "Thanks.", "end": True}]
    persona = ScriptedPersona(resolved(script=script))
    await persona.next_turn(view(None))
    with pytest.raises(PersonaError, match="outside its hidden window"):
        await persona.next_turn(view(reply("Only this one.", (slot(OUT_OF_WINDOW, "s_1"),))))


async def test_saying_an_offered_label_is_an_acceptance_too() -> None:
    script = [
        {"say": "Hi, book me a call."},
        {"say": "Yes, {{offered[0].label}} works.", "when": "agent_offered_slots"},
        {"say": "Thanks.", "end": True},
    ]
    persona = ScriptedPersona(resolved(script=script))
    await persona.next_turn(view(None))
    ok = await persona.next_turn(view(reply("This one?", (slot(IN_WINDOW, "s_1", "Mon 5 Oct, 2:00 PM"),))))
    assert ok is not None
    assert ok.text == "Yes, Mon 5 Oct, 2:00 PM works."

    persona = ScriptedPersona(resolved(script=script))
    await persona.next_turn(view(None))
    with pytest.raises(PersonaError):
        await persona.next_turn(view(reply("This one?", (slot(OUT_OF_WINDOW, "s_1"),))))


async def test_a_pick_is_skipped_when_the_agent_already_booked() -> None:
    persona = ScriptedPersona(resolved())
    await persona.next_turn(view(None))
    booked = reply(
        "You're all set for Monday 5 October at 2:00 PM.",
        booking={"ref": "abc", "status": "accepted", "action": "booked"},
    )
    turn = await persona.next_turn(view(booked))
    assert turn is not None
    assert (turn.kind, turn.text, turn.end) == ("say", "Great, thanks!", True)


async def test_conditional_steps_read_the_last_agent_message() -> None:
    persona = ScriptedPersona(resolved("fault-slot-taken-after-offer"))
    await persona.next_turn(view(None))
    first = await persona.next_turn(view(reply("Pick one.", (slot(IN_WINDOW, "s_1"),))))
    assert first is not None
    assert first.kind == "pick"
    second_offer = datetime(2026, 10, 6, 18, 0, tzinfo=UTC)
    taken = reply("Sorry, that one was just taken. How about another?", (slot(second_offer, "s_2"),))
    again = await persona.next_turn(view(taken))
    assert again is not None
    assert again.kind == "pick"
    assert again.offer is not None
    assert again.offer.start_utc == second_offer


@pytest.mark.parametrize(
    ("text", "asks"),
    [
        ("What time zone are you in?", True),
        ("Could you tell me your time zone so I can show the right times.", True),
        ("I'll use America/Denver based on your browser. Is that right?", True),
        ("Just to check, are you on UTC+2? Is that correct?", True),
        ("Where are you located?", True),
        ("Here are times in your zone: Monday 5 October at 2:00 PM. Which one works for you?", False),
        ("Would 2:00 PM America/New_York work for you?", False),
        ("I'll use Asia/Kolkata (UTC+05:30) for times, tell me if that's wrong.", False),
        (None, False),
    ],
)
def test_agent_asks_timezone(text: str | None, asks: bool) -> None:
    assert agent_asks_timezone(text) is asks


@pytest.mark.parametrize(
    ("text", "asks"),
    [
        ("Shall I book Monday 5 October at 2:00 PM for you?", True),
        ("Would you like me to go ahead?", True),
        ("Does that work for you?", True),
        ("Please confirm and I'll book it.", True),
        ("Just to confirm, you want to cancel your call on Tuesday?", True),
        ("Which of these works for you?", False),
        ("Can you confirm your time zone?", False),
        ("You're all set.", False),
    ],
)
def test_agent_asks_confirmation(text: str, asks: bool) -> None:
    assert agent_asks_confirmation(text) is asks


def test_a_booked_time_is_not_an_offer() -> None:
    offers = text_offers(
        "You're all set! Your call is booked for Monday 5 October at 2:00 PM.",
        prospect_zone="America/New_York",
        host_zone="America/New_York",
        reference=NOW,
    )
    assert offers == []


def test_text_offers_are_labelled_in_the_zone_the_agent_used() -> None:
    offers = text_offers(
        "I have Monday 5 October at 2:00 PM ET or 3:00 PM ET.",
        prospect_zone="Europe/Berlin",
        host_zone="America/New_York",
        reference=NOW,
    )
    assert [o.start_utc for o in offers] == [IN_WINDOW, datetime(2026, 10, 5, 19, 0, tzinfo=UTC)]
    assert offers[0].label == "Monday 5 October at 2:00 PM America/New_York"
    assert offers[0].slot_id is None


def test_slot_label_names_foreign_zones_only() -> None:
    assert slot_label(IN_WINDOW, "America/New_York", "America/New_York") == "Monday 5 October at 2:00 PM"
    assert slot_label(IN_WINDOW, "UTC", "America/New_York") == "Monday 5 October at 6:00 PM UTC"
    assert slot_label(IN_WINDOW, "UTC+05:30", "Asia/Kolkata") == "Monday 5 October at 11:30 PM UTC+05:30"


def test_quick_reply_offers_need_a_start_and_render_a_missing_label() -> None:
    items = [
        {"label": "Keep my current time", "action": {"type": "cancel"}},
        {"action": {"type": "select_slot", "slot_id": "s_1"}, "start_utc": "2026-10-05T18:00:00Z"},
        {"label": "Tue", "action": {"type": "select_slot", "slot_id": "s_2"}, "start_utc": "not a time"},
    ]
    offers = quick_reply_offers(items, "America/New_York")
    assert offers == [Offer(IN_WINDOW, "Monday 5 October at 2:00 PM", "s_1")]


def test_reply_offers_prefer_quick_replies() -> None:
    both = reply(OFFERS_TEXT, (slot(IN_WINDOW, "s_1", "chip"),))
    assert [
        o.label
        for o in reply_offers(
            both, prospect_zone="America/New_York", host_zone="America/New_York", reference=NOW
        )
    ] == ["chip"]
    assert reply_offers(None, prospect_zone="UTC", host_zone="UTC", reference=NOW) == []


# Harness faults --------------------------------------------------------------------------------------------


def test_duplicate_delay_is_between_50_and_200_ms_and_reproducible() -> None:
    delays = [duplicate_delay(random.Random(f"trace-{i}")) for i in range(200)]
    low, high = DUPLICATE_DELAY_S
    assert all(low <= d <= high for d in delays)
    assert duplicate_delay(random.Random("same")) == duplicate_delay(random.Random("same"))


def test_concurrent_message_asks_for_the_requested_offer_or_the_last_one() -> None:
    offers = [Offer(IN_WINDOW, "first"), Offer(OUT_OF_WINDOW, "second")]
    assert concurrent_message(offers, 1) == ("Please book second for me instead.", 1)
    assert concurrent_message(offers[:1], 1) == ("Please book first for me instead.", 0)
    with pytest.raises(ValueError, match="at least one"):
        concurrent_message([], 1)


class _RecordingClient:
    supports_actions = True

    def __init__(self) -> None:
        self.sent: list[tuple[float, Turn]] = []
        self.clock = 0.0

    async def send(self, turn: Turn) -> AgentReply:
        import asyncio
        import time

        start = time.perf_counter()
        self.sent.append((start, turn))
        await asyncio.sleep(0.01 if turn.channel == "api" else 0.02)
        return AgentReply(
            status=200, reply=f"ok {turn.session_id}", sent_at=start, received_at=time.perf_counter()
        )

    async def aclose(self) -> None:
        return None


LEAD = Lead(email="lena-00000000@example.com", name="Lena M.")


async def test_duplicate_delivery_resends_the_same_message_id_after_the_delay() -> None:
    client = _RecordingClient()
    turn = Turn(session_id="s", message_id="s-m3", lead=LEAD, message="Monday works")
    deliveries, last, report = await deliver_duplicate(client, turn, delay_s=0.06)
    (t1, first), (t2, second) = client.sent
    assert first == second
    assert 0.05 <= t2 - t1 < 0.2
    assert [d.duplicate for d in deliveries] == [False, True]
    assert last is deliveries[1].reply
    assert report.to_json()["message_id"] == "s-m3"
    assert report.to_json()["continued_from"] == "duplicate"
    assert report.to_json()["injected"] is True


async def test_concurrent_channel_sends_both_sessions_at_once() -> None:
    client = _RecordingClient()
    turn_a = Turn(
        session_id="a", message_id="a-m3", lead=LEAD, action={"type": "select_slot", "slot_id": "s_1"}
    )
    turn_b = Turn(
        session_id="a-b", message_id="a-b-m1", lead=LEAD, message="Please book x", channel="webhook"
    )
    deliveries, reply_a, report = await deliver_concurrent(
        client, turn_a, turn_b, requested_index=1, used_index=1
    )
    (t1, _), (t2, _) = client.sent
    assert abs(t2 - t1) < 0.01
    assert reply_a.reply == "ok a"
    assert [d.session for d in deliveries] == ["A", "B"]
    assert report.to_json()["arrival_order"] == ["A", "B"]
    assert report.to_json()["channel_b"] == "webhook"
    assert report.to_json()["injected"] is True
