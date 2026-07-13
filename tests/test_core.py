"""Core domain tests for agreements, challenges, and multi-instance isolation.

Run from the repo root with:  python3 -m unittest discover -s tests -t .
(or simply:  python3 tests/test_core.py)
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import (  # noqa: E402
    InstanceStore,
    UserConfig,
    challenge_expired,
    challenge_streak,
    checkin_stats,
    checkin_streak,
    is_scheduled_day,
    next_item_id,
    parse_agreement_text,
    parse_challenge_text,
)

ALICE = UserConfig(id=1, name="Alice")
BOB = UserConfig(id=2, name="Bob")
CARO = UserConfig(id=3, name="Caro")

TODAY = datetime.now(timezone.utc).date()


class LedgerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = InstanceStore(Path(self._tmp.name) / "data.json", [ALICE, BOB])
        self.ledger = self.store.ledger_for("default")


class AgreementLifecycleTests(LedgerTestCase):
    def test_create_is_pending_with_creator_accepted(self):
        agreement = self.ledger.create_agreement(ALICE, "no phones at dinner")
        self.assertEqual(agreement["status"], "pending")
        self.assertEqual(agreement["accepted_by"], [ALICE.id])
        self.assertEqual(agreement["declined_by"], [])
        self.assertEqual(agreement["participants"], {str(ALICE.id): "Alice"})

    def test_second_accept_activates(self):
        agreement = self.ledger.create_agreement(ALICE, "gym twice a week")
        agreement = self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        self.assertEqual(agreement["status"], "active")
        self.assertIn("activated_at", agreement)
        self.assertCountEqual(agreement["accepted_by"], [ALICE.id, BOB.id])

    def test_decline_recorded_and_reversible(self):
        agreement = self.ledger.create_agreement(ALICE, "split rent 60/40")
        agreement = self.ledger.respond_agreement(agreement["id"], BOB, accept=False)
        self.assertEqual(agreement["status"], "pending")
        self.assertEqual(agreement["declined_by"], [BOB.id])
        agreement = self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        self.assertEqual(agreement["status"], "active")
        self.assertEqual(agreement["declined_by"], [])

    def test_decline_after_active_deactivates_below_two(self):
        agreement = self.ledger.create_agreement(ALICE, "quiet hours after 22h")
        self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        agreement = self.ledger.respond_agreement(agreement["id"], BOB, accept=False)
        self.assertEqual(agreement["status"], "pending")

    def test_third_party_can_join_active_agreement(self):
        agreement = self.ledger.create_agreement(ALICE, "shared groceries")
        self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        agreement = self.ledger.respond_agreement(agreement["id"], CARO, accept=True)
        self.assertEqual(agreement["status"], "active")
        self.assertIn(CARO.id, agreement["accepted_by"])
        self.assertEqual(agreement["participants"][str(CARO.id)], "Caro")

    def test_revoke_only_by_creator(self):
        agreement = self.ledger.create_agreement(ALICE, "one dessert per day")
        with self.assertRaises(ValueError):
            self.ledger.revoke_agreement(agreement["id"], BOB)
        agreement = self.ledger.revoke_agreement(agreement["id"], ALICE)
        self.assertEqual(agreement["status"], "revoked")
        with self.assertRaises(ValueError):
            self.ledger.respond_agreement(agreement["id"], BOB, accept=True)

    def test_breach_only_on_active(self):
        agreement = self.ledger.create_agreement(ALICE, "no snoozing alarms")
        with self.assertRaises(ValueError):
            self.ledger.record_breach(agreement["id"], BOB, ALICE, "snoozed 4x")
        self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        agreement = self.ledger.record_breach(agreement["id"], BOB, ALICE, "snoozed 4x")
        self.assertEqual(len(agreement["breaches"]), 1)
        breach = agreement["breaches"][0]
        self.assertEqual(breach["user_id"], BOB.id)
        self.assertEqual(breach["reported_by"], ALICE.id)
        self.assertEqual(breach["note"], "snoozed 4x")

    def test_latest_open_prefers_pending(self):
        first = self.ledger.create_agreement(ALICE, "first")
        self.ledger.respond_agreement(first["id"], BOB, accept=True)
        second = self.ledger.create_agreement(ALICE, "second")
        self.assertEqual(self.ledger.latest_open_agreement()["id"], second["id"])
        self.ledger.respond_agreement(second["id"], BOB, accept=True)
        self.assertEqual(self.ledger.latest_open_agreement()["id"], second["id"])


class ChallengeLifecycleTests(LedgerTestCase):
    def test_create_defaults(self):
        challenge = self.ledger.create_challenge(ALICE, "read books", None, None)
        self.assertEqual(challenge["status"], "active")
        self.assertIsNone(challenge["target"])
        self.assertIsNone(challenge["deadline"])
        self.assertEqual(challenge["participants"], {str(ALICE.id): "Alice"})

    def test_score_accumulates_and_tracks_daily(self):
        challenge = self.ledger.create_challenge(ALICE, "push-ups", 100, None)
        challenge, completed = self.ledger.add_challenge_score(challenge["id"], ALICE, 30)
        self.assertFalse(completed)
        self.assertEqual(challenge["scores"][str(ALICE.id)], 30)
        self.assertEqual(challenge["daily"][str(ALICE.id)][TODAY.isoformat()], 30)

    def test_target_completion_declares_winner(self):
        challenge = self.ledger.create_challenge(ALICE, "push-ups", 50, None)
        self.ledger.add_challenge_score(challenge["id"], BOB, 20)
        challenge, completed = self.ledger.add_challenge_score(challenge["id"], BOB, 30)
        self.assertTrue(completed)
        self.assertEqual(challenge["status"], "completed")
        self.assertEqual(challenge["winner_ids"], [BOB.id])
        with self.assertRaises(ValueError):
            self.ledger.add_challenge_score(challenge["id"], ALICE, 10)

    def test_negative_correction_clamps_at_zero(self):
        challenge = self.ledger.create_challenge(ALICE, "km run", None, None)
        self.ledger.add_challenge_score(challenge["id"], ALICE, 5)
        challenge, _ = self.ledger.add_challenge_score(challenge["id"], ALICE, -8)
        self.assertEqual(challenge["scores"][str(ALICE.id)], 0)
        with self.assertRaises(ValueError):
            self.ledger.add_challenge_score(challenge["id"], ALICE, 0)

    def test_deadline_expiry_ends_challenge_with_top_scorer(self):
        yesterday = (TODAY - timedelta(days=1)).isoformat()
        challenge = self.ledger.create_challenge(ALICE, "steps", None, yesterday)
        challenge["scores"] = {str(ALICE.id): 10, str(BOB.id): 25}
        ended = self.ledger.expire_due_challenges(TODAY)
        self.assertEqual(len(ended), 1)
        self.assertEqual(ended[0]["status"], "ended")
        self.assertEqual(ended[0]["winner_ids"], [BOB.id])

    def test_scoring_expired_challenge_finalizes_and_raises(self):
        yesterday = (TODAY - timedelta(days=1)).isoformat()
        challenge = self.ledger.create_challenge(ALICE, "steps", None, yesterday)
        with self.assertRaises(ValueError):
            self.ledger.add_challenge_score(challenge["id"], ALICE, 5)
        self.assertEqual(challenge["status"], "ended")

    def test_finish_challenge_creator_only(self):
        challenge = self.ledger.create_challenge(ALICE, "plank minutes", None, None)
        with self.assertRaises(ValueError):
            self.ledger.finish_challenge(challenge["id"], BOB)
        challenge = self.ledger.finish_challenge(challenge["id"], ALICE)
        self.assertEqual(challenge["status"], "ended")

    def test_streak_counts_consecutive_days(self):
        challenge = self.ledger.create_challenge(ALICE, "push-ups", None, None)
        daily = {}
        for offset in (1, 2, 3, 5):
            daily[(TODAY - timedelta(days=offset)).isoformat()] = 10
        challenge["daily"] = {str(ALICE.id): daily}
        # Nothing today yet, so the streak counts back from yesterday.
        self.assertEqual(challenge_streak(challenge, ALICE.id, TODAY), 3)
        challenge["daily"][str(ALICE.id)][TODAY.isoformat()] = 10
        self.assertEqual(challenge_streak(challenge, ALICE.id, TODAY), 4)
        self.assertEqual(challenge_streak(challenge, BOB.id, TODAY), 0)


class AgreementCheckinTests(LedgerTestCase):
    def _tracked_agreement(self, days="daily", until=None):
        agreement = self.ledger.create_agreement(
            ALICE, "meet 8:30 weekdays, 10:00 weekends",
            {"days": days, "until": until, "log": {}},
        )
        self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        return agreement

    def test_checkin_records_per_day(self):
        agreement = self._tracked_agreement()
        agreement = self.ledger.record_checkin(agreement["id"], ALICE, TODAY.isoformat(), True)
        agreement = self.ledger.record_checkin(agreement["id"], BOB, TODAY.isoformat(), False)
        log = agreement["checkin"]["log"]
        self.assertTrue(log[str(ALICE.id)][TODAY.isoformat()])
        self.assertFalse(log[str(BOB.id)][TODAY.isoformat()])
        self.assertEqual(checkin_stats(agreement, ALICE.id), (1, 0))
        self.assertEqual(checkin_stats(agreement, BOB.id), (0, 1))

    def test_checkin_answer_can_be_changed(self):
        agreement = self._tracked_agreement()
        self.ledger.record_checkin(agreement["id"], ALICE, TODAY.isoformat(), False)
        agreement = self.ledger.record_checkin(agreement["id"], ALICE, TODAY.isoformat(), True)
        self.assertEqual(checkin_stats(agreement, ALICE.id), (1, 0))

    def test_checkin_rejects_future_and_unscheduled_days(self):
        kind_not_today = "weekends" if TODAY.weekday() < 5 else "weekdays"
        agreement = self._tracked_agreement(days=kind_not_today)
        with self.assertRaises(ValueError):
            self.ledger.record_checkin(agreement["id"], ALICE, TODAY.isoformat(), True)
        daily = self._tracked_agreement()
        tomorrow = (TODAY + timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            self.ledger.record_checkin(daily["id"], ALICE, tomorrow, True)

    def test_checkin_requires_tracked_agreement(self):
        agreement = self.ledger.create_agreement(ALICE, "plain pact")
        self.ledger.respond_agreement(agreement["id"], BOB, accept=True)
        with self.assertRaises(ValueError):
            self.ledger.record_checkin(agreement["id"], ALICE, TODAY.isoformat(), True)

    def test_checkin_auto_accepts_participant(self):
        agreement = self.ledger.create_agreement(
            ALICE, "morning runs", {"days": "daily", "until": None, "log": {}}
        )
        self.assertEqual(agreement["status"], "pending")
        agreement = self.ledger.record_checkin(agreement["id"], CARO, TODAY.isoformat(), True)
        self.assertEqual(agreement["status"], "active")
        self.assertIn(CARO.id, agreement["accepted_by"])

    def test_solo_checkin_needs_second_acceptor(self):
        agreement = self.ledger.create_agreement(
            ALICE, "morning runs", {"days": "daily", "until": None, "log": {}}
        )
        with self.assertRaises(ValueError):
            self.ledger.record_checkin(agreement["id"], ALICE, TODAY.isoformat(), True)
        self.assertEqual(agreement["status"], "pending")

    def test_period_end_completes_agreement(self):
        yesterday = (TODAY - timedelta(days=1)).isoformat()
        agreement = self._tracked_agreement(until=yesterday)
        done = self.ledger.finish_due_agreements(TODAY)
        self.assertEqual([a["id"] for a in done], [agreement["id"]])
        self.assertEqual(agreement["status"], "completed")
        with self.assertRaises(ValueError):
            self.ledger.record_checkin(agreement["id"], ALICE, yesterday, True)
        with self.assertRaises(ValueError):
            self.ledger.respond_agreement(agreement["id"], CARO, accept=True)

    def test_checkin_streak_skips_unscheduled_days(self):
        agreement = self._tracked_agreement(days="weekdays")
        log = {}
        day = TODAY - timedelta(days=1)
        weekdays_marked = 0
        while weekdays_marked < 6:  # six most recent weekdays, skipping weekends
            if day.weekday() < 5:
                log[day.isoformat()] = True
                weekdays_marked += 1
            day -= timedelta(days=1)
        agreement["checkin"]["log"] = {str(ALICE.id): log}
        self.assertEqual(checkin_streak(agreement, ALICE.id, TODAY), 6)
        # A miss on the oldest marked day doesn't matter; one in the middle breaks it.
        middle = sorted(log.keys())[2]
        log[middle] = False
        self.assertEqual(checkin_streak(agreement, ALICE.id, TODAY), 3)

    def test_latest_checkin_agreement_found(self):
        self.ledger.create_agreement(ALICE, "plain pact")
        tracked = self._tracked_agreement()
        self.assertEqual(self.ledger.latest_checkin_agreement()["id"], tracked["id"])


class ParseAgreementTests(unittest.TestCase):
    def test_plain_text_has_no_checkin(self):
        self.assertEqual(parse_agreement_text("no phones at dinner", TODAY),
                         ("no phones at dinner", None))

    def test_schedule_and_duration(self):
        text, checkin = parse_agreement_text(
            "meet 8.30 weekdays and 10 weekends everyday for 1y", TODAY
        )
        self.assertEqual(text, "meet 8.30 weekdays and 10 weekends")
        self.assertEqual(checkin["days"], "daily")
        self.assertEqual(checkin["until"], (TODAY + timedelta(days=365)).isoformat())

    def test_weekday_schedule_without_duration(self):
        text, checkin = parse_agreement_text("gym before work weekdays", TODAY)
        self.assertEqual(text, "gym before work")
        self.assertEqual(checkin, {"days": "weekdays", "until": None, "log": {}})

    def test_duration_alone_implies_daily(self):
        text, checkin = parse_agreement_text("no sugar for 30d", TODAY)
        self.assertEqual(text, "no sugar")
        self.assertEqual(checkin["days"], "daily")
        self.assertEqual(checkin["until"], (TODAY + timedelta(days=30)).isoformat())

    def test_until_date(self):
        future = (TODAY + timedelta(days=90)).isoformat()
        _, checkin = parse_agreement_text(f"meditate daily until {future}", TODAY)
        self.assertEqual(checkin["until"], future)

    def test_past_until_rejected(self):
        past = (TODAY - timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            parse_agreement_text(f"meditate daily until {past}", TODAY)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            parse_agreement_text("", TODAY)
        with self.assertRaises(ValueError):
            parse_agreement_text("daily for 7d", TODAY)  # schedule but no text

    def test_is_scheduled_day(self):
        monday = TODAY - timedelta(days=TODAY.weekday())
        saturday = monday + timedelta(days=5)
        self.assertTrue(is_scheduled_day("daily", monday))
        self.assertTrue(is_scheduled_day("daily", saturday))
        self.assertTrue(is_scheduled_day("weekdays", monday))
        self.assertFalse(is_scheduled_day("weekdays", saturday))
        self.assertFalse(is_scheduled_day("weekends", monday))
        self.assertTrue(is_scheduled_day("weekends", saturday))


class ParseChallengeTests(unittest.TestCase):
    def test_plain_title(self):
        self.assertEqual(parse_challenge_text("read books", TODAY), (None, "read books", None))

    def test_target_and_duration_days(self):
        target, title, deadline = parse_challenge_text("100 push-ups for 7d", TODAY)
        self.assertEqual((target, title), (100, "push-ups"))
        self.assertEqual(deadline, (TODAY + timedelta(days=7)).isoformat())

    def test_duration_weeks(self):
        _, _, deadline = parse_challenge_text("climb for 2 weeks", TODAY)
        self.assertEqual(deadline, (TODAY + timedelta(days=14)).isoformat())

    def test_duration_months_and_years(self):
        _, _, deadline = parse_challenge_text("run for 3m", TODAY)
        self.assertEqual(deadline, (TODAY + timedelta(days=90)).isoformat())
        _, _, deadline = parse_challenge_text("meet at 8:30 for 1y", TODAY)
        self.assertEqual(deadline, (TODAY + timedelta(days=365)).isoformat())

    def test_until_date(self):
        future = (TODAY + timedelta(days=10)).isoformat()
        target, title, deadline = parse_challenge_text(f"50 swims until {future}", TODAY)
        self.assertEqual((target, title, deadline), (50, "swims", future))

    def test_until_past_rejected(self):
        past = (TODAY - timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            parse_challenge_text(f"swims until {past}", TODAY)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            parse_challenge_text("   ", TODAY)
        with self.assertRaises(ValueError):
            parse_challenge_text("100 for 7d", TODAY)  # target+deadline but no title

    def test_expired_helper(self):
        self.assertFalse(challenge_expired({"deadline": None}, TODAY))
        self.assertFalse(challenge_expired({"deadline": TODAY.isoformat()}, TODAY))
        self.assertTrue(
            challenge_expired({"deadline": (TODAY - timedelta(days=1)).isoformat()}, TODAY)
        )
        self.assertFalse(challenge_expired({"deadline": "garbage"}, TODAY))


class MultiInstanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "data.json"
        self.store = InstanceStore(self.path, [ALICE, BOB])

    def test_instances_are_isolated(self):
        self.store.ensure_instance("chat:-100", "Flat", -100)
        default_ledger = self.store.ledger_for("default")
        group_ledger = self.store.ledger_for("chat:-100")
        default_ledger.create_agreement(ALICE, "default-only pact")
        group_ledger.create_challenge(CARO, "group-only race", None, None)
        self.assertEqual(len(default_ledger.state.get("challenges", [])), 0)
        self.assertEqual(len(group_ledger.state.get("agreements", [])), 0)

    def test_persists_across_reload(self):
        self.store.ensure_instance("chat:-100", "Flat", -100)
        group_ledger = self.store.ledger_for("chat:-100")
        agreement = group_ledger.create_agreement(CARO, "dishes rotate daily")
        group_ledger.respond_agreement(agreement["id"], BOB, accept=True)

        reloaded = InstanceStore(self.path, [ALICE, BOB])
        ledger = reloaded.ledger_for("chat:-100")
        stored = ledger.find_agreement(agreement["id"])
        self.assertEqual(stored["status"], "active")
        self.assertEqual(stored["participants"][str(CARO.id)], "Caro")

    def test_legacy_records_upgrade_in_place(self):
        # Simulate data written by the previous bot version: no declined_by,
        # participants, breaches, daily, or winner_ids fields.
        instance = self.store.ensure_instance("chat:-200", "Legacy", -200)
        instance["agreements"] = [
            {
                "id": "a1",
                "text": "old pact",
                "creator_id": ALICE.id,
                "creator_name": "Alice",
                "accepted_by": [ALICE.id],
                "status": "pending",
                "created_at": "2026-07-01T00:00:00+00:00",
            }
        ]
        instance["challenges"] = [
            {
                "id": "c1",
                "title": "old race",
                "target": 10,
                "creator_id": ALICE.id,
                "creator_name": "Alice",
                "scores": {str(ALICE.id): 4},
                "status": "active",
                "created_at": "2026-07-01T00:00:00+00:00",
            }
        ]
        ledger = self.store.ledger_for("chat:-200")

        agreement = ledger.respond_agreement("a1", BOB, accept=True)
        self.assertEqual(agreement["status"], "active")
        self.assertEqual(agreement["declined_by"], [])

        challenge, completed = ledger.add_challenge_score("c1", BOB, 3)
        self.assertFalse(completed)
        self.assertEqual(challenge["scores"][str(BOB.id)], 3)
        challenge, completed = ledger.add_challenge_score("c1", ALICE, 6)
        self.assertTrue(completed)
        self.assertEqual(challenge["winner_ids"], [ALICE.id])

    def test_next_item_id_continues_sequence(self):
        items = [{"id": "c1"}, {"id": "c7"}, {"id": "a3"}]
        self.assertEqual(next_item_id(items, "c"), "c8")
        self.assertEqual(next_item_id(items, "a"), "a4")
        self.assertEqual(next_item_id([], "a"), "a1")


if __name__ == "__main__":
    unittest.main()
