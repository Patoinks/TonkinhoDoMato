"""Resignation, decided by TonkAI's own network estimate and nothing else.

The owner's rule: when TonkAI is hopeless it resigns, and the point is that it
learns never to get there. "Hopeless" is read ONLY from the network:

  * survival objective: even the best legal move predicts death within
    ``survival_plies`` half-moves, i.e. value <= 1 - gamma ** survival_plies
  * victory objective: even the best legal move has expected score <= 0.05
  * shaped objective: the survival part as above, with the victory measure
    near its floor (0.25): (1 - w) * (1 - gamma ** plies) + w * 0.25

Never an engine score, never material, never TinyOpponent's evaluation, and in
V3 never the heuristic-adjusted move scores (king/piece safety, search, memory,
information bonus). A resignation driven by material would turn "down five
pawns" into a lost game and teach classical piece values through the back door.

Safeguards
  * Playout check (as in AlphaZero): some games that would resign are played
    to the end instead, to measure how often resigning would have been wrong.
  * Shadow mode: resignation starts off and only records "would resign here".
    It turns on only after enough verified triggers, a false-resignation rate
    of at most 5%, AND at least 10% non-losses in recent games. The last
    condition is the important one: a value head that predicts loss everywhere
    while TonkAI loses almost every game is "right" every time, so the playout
    check alone would approve resigning at move one and destroy the data.

A resignation is a real loss, as in chess and in AlphaZero: the game ends at
the resignation ply and is labelled lost. That is the owner's intent - TonkAI
must learn never to reach a hopeless position - and the playout check bounds
how often the label is wrong. Setting ``min_non_loss`` to zero ("resign for
real") removes the variance condition; verification and tightening remain.

This module must not import any engine, opponent or heuristic module; a test
enforces that.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass

import numpy as np

OFF, SHADOW, ACTIVE, VERIFY = "off", "shadow", "active", "verify"


@dataclass
class ResignConfig:
    enabled: bool = False
    threshold: float | None = None   # overrides the objective's default
    survival_plies: int = 10         # survival: death predicted within this
    victory_score: float = 0.05      # victory: expected score at or below this
    patience: int = 3                # consecutive hopeless decisions
    min_ply: int = 40                # never resign before this half-move
    playout: float = 0.10            # share of would-resign games played out
    min_verified: int = 200          # played-out triggers before trusting it
    max_false_rate: float = 0.05     # allowed wrong resignations
    min_non_loss: float = 0.10       # results must not be (almost) constant
    window: int = 1000               # recent games for the non-loss rate
    tighten: float = 0.8             # threshold multiplier when too eager
    seed: int = 0x5E516              # own stream; never the games' RNG
    force: bool = False              # owner's choice: always ACTIVE, no shadow, no played-out checks

    def validate(self):
        if self.patience < 1 or self.min_ply < 0 or self.survival_plies < 1:
            raise ValueError("resign patience/survival_plies must be positive, min_ply >= 0")
        if self.min_verified < 1 or self.window < 1:
            raise ValueError("resign min_verified and window must be positive")
        for name in ("playout", "max_false_rate", "min_non_loss", "victory_score"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"resign {name} must be between zero and one")
        if not 0 < self.tighten < 1:
            raise ValueError("resign tighten must be between zero and one")
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ValueError("resign threshold must be between zero and one")


def hope_threshold(rc: ResignConfig, objective: str, gamma: float, weight: float = 0.5) -> float:
    """TonkAI resigns when its best option's value is at or below this."""
    if rc.threshold is not None:
        return rc.threshold
    if objective == "survival":
        return 1.0 - gamma ** rc.survival_plies
    if objective == "victory":
        return rc.victory_score
    if objective == "shaped":
        return (1.0 - weight) * (1.0 - gamma ** rc.survival_plies) + weight * 0.25
    raise ValueError(f"unknown objective {objective!r}")


def best_hope(network_values) -> float:
    """The most hopeful legal move, by the raw network estimate only."""
    return float(np.max(np.asarray(network_values, dtype=np.float64)))


def choose_mode(state: str, playout: float, rng) -> str | None:
    """Mode for one game. ACTIVE games are sometimes played out to verify."""
    if state == OFF:
        return None
    if state == SHADOW:
        return SHADOW
    return VERIFY if rng.random() < playout else ACTIVE


class ResignTracker:
    """One game. Consumes TonkAI's hope per decision; stops only when ACTIVE."""

    def __init__(self, mode: str, threshold: float, patience: int, min_ply: int):
        if mode not in (SHADOW, ACTIVE, VERIFY):
            raise ValueError(f"invalid resign mode {mode!r}")
        self.mode, self.threshold = mode, float(threshold)
        self.patience, self.min_ply = int(patience), int(min_ply)
        self.streak = 0
        self.decisions = 0
        self.trigger_ply: int | None = None
        self.resigned = False
        self.lowest_hope = math.inf

    @classmethod
    def from_plan(cls, plan):
        return None if plan is None else cls(*plan)

    def observe(self, hope: float, ply: int) -> bool:
        """Record one decision. True means: resign now, do not move."""
        self.decisions += 1
        self.lowest_hope = min(self.lowest_hope, float(hope))
        if ply >= self.min_ply and hope <= self.threshold:
            self.streak += 1
        else:
            self.streak = 0
        if self.trigger_ply is None and self.streak >= self.patience:
            self.trigger_ply = int(ply)
        if self.mode == ACTIVE and self.trigger_ply is not None:
            self.resigned = True
        return self.resigned

    def report(self, plies: int) -> dict:
        return {"mode": self.mode, "threshold": self.threshold,
                "triggered": self.trigger_ply is not None, "trigger_ply": self.trigger_ply,
                "resigned": self.resigned, "plies": int(plies),
                "lowest_hope": None if math.isinf(self.lowest_hope) else self.lowest_hope}


class ResignController:
    """Learner side: picks each game's mode, verifies, and gates itself."""

    def __init__(self, rc: ResignConfig, threshold: float):
        rc.validate()
        self.rc = rc
        self.threshold = float(threshold)
        self.initial_threshold = self.threshold
        self.state = SHADOW if rc.enabled else OFF
        self.reason = "shadow: waiting for verified triggers" if rc.enabled else "disabled"
        if rc.enabled and rc.force:
            self.state, self.reason = ACTIVE, "active (forced by the owner)"
        self.rng = np.random.default_rng(rc.seed)
        self.recent = deque(maxlen=rc.window)             # 1.0 = game not lost
        self.verified = deque(maxlen=rc.min_verified)     # True = would have been wrong
        self.after_trigger = deque(maxlen=rc.window)      # plies played after a trigger
        self.resign_plies = deque(maxlen=rc.window)
        self.counts = Counter()

    def plan(self):
        """(mode, threshold, patience, min_ply) for the next game, or None."""
        if self.rc.enabled and self.rc.force:
            return (ACTIVE, self.threshold, self.rc.patience, self.rc.min_ply)
        mode = choose_mode(self.state, self.rc.playout, self.rng)
        if mode is None:
            return None
        return (mode, self.threshold, self.rc.patience, self.rc.min_ply)

    def update(self, report: dict | None, lost: bool) -> list[str]:
        """Feed one finished game. Returns human-readable state changes."""
        self.counts["games"] += 1
        resigned = bool(report and report["resigned"])
        # A resignation is a loss for the purpose of "are results constant?".
        self.recent.append(0.0 if (lost or resigned) else 1.0)
        if report:
            if report["triggered"]:
                self.counts["triggered"] += 1
            if resigned:
                self.counts["resigned"] += 1
                self.resign_plies.append(report["trigger_ply"])
            elif report["triggered"] and math.isclose(report["threshold"], self.threshold):
                wrong = not lost
                self.verified.append(wrong)
                self.counts["verified"] += 1
                self.counts["false"] += int(wrong)
                self.after_trigger.append(report["plies"] - report["trigger_ply"])
        return self._gate()

    @property
    def false_rate(self) -> float | None:
        return float(np.mean(self.verified)) if self.verified else None

    @property
    def non_loss(self) -> float:
        return float(np.mean(self.recent)) if self.recent else 0.0

    def _tighten(self, messages):
        old = self.threshold
        self.threshold = max(1e-6, self.threshold * self.rc.tighten)
        self.verified.clear()
        self.counts["tightened"] += 1
        messages.append(f"[resign] false resignations above {self.rc.max_false_rate:.0%}: "
                        f"threshold {old:.4f} -> {self.threshold:.4f}")

    def _gate(self) -> list[str]:
        if self.state == OFF:
            return []
        if self.rc.force:        # never demoted, never tightened
            self.state, self.reason = ACTIVE, "active (forced by the owner)"
            return []
        rc, messages = self.rc, []
        n, rate, non_loss = len(self.verified), self.false_rate, self.non_loss
        full = n >= rc.min_verified
        if full and rate > rc.max_false_rate:
            self._tighten(messages)
            n, rate, full = 0, None, False
        if self.state == SHADOW:
            if not full:
                self.reason = f"shadow: {n}/{rc.min_verified} verified triggers"
            elif non_loss < rc.min_non_loss:
                self.reason = (f"shadow: only {non_loss:.1%} non-losses in the last "
                               f"{len(self.recent)} games - results are constant")
            else:
                self.state, self.reason = ACTIVE, "active"
                self.counts["activations"] += 1
                messages.append(f"[resign] ACTIVE: {n} verified, false {rate:.1%}, "
                                f"non-losses {non_loss:.1%}, threshold {self.threshold:.4f}")
        elif non_loss < rc.min_non_loss:
            self.state = SHADOW
            self.reason = f"shadow: non-losses fell to {non_loss:.1%} - results are constant"
            self.counts["deactivations"] += 1
            messages.append(f"[resign] back to SHADOW: {self.reason}")
        return messages

    def summary(self) -> dict:
        after = float(np.mean(self.after_trigger)) if self.after_trigger else None
        return {"state": self.state, "reason": self.reason, "threshold": self.threshold,
                "initial_threshold": self.initial_threshold, "games": self.counts["games"],
                "triggered": self.counts["triggered"], "resigned": self.counts["resigned"],
                "verified": self.counts["verified"], "false_resignations": self.counts["false"],
                "false_rate_window": self.false_rate, "non_loss_rate": self.non_loss,
                "mean_resign_ply": float(np.mean(self.resign_plies)) if self.resign_plies else None,
                "mean_plies_after_trigger": after,
                "estimated_plies_saved": (after or 0.0) * self.counts["resigned"],
                "tightened": self.counts["tightened"], "config": asdict(self.rc)}

    def line(self) -> str:
        rate = "-" if self.false_rate is None else f"{self.false_rate:.1%}"
        return (f"Resign: {self.state.upper()} ({self.reason})  thr {self.threshold:.4f}  "
                f"triggered {self.counts['triggered']}/{self.counts['games']}  "
                f"resigned {self.counts['resigned']}  false {rate}  "
                f"non-loss {self.non_loss:.1%}")

    def state_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason, "threshold": self.threshold,
                "initial_threshold": self.initial_threshold, "rng": self.rng.bit_generator.state,
                "recent": list(self.recent), "verified": list(self.verified),
                "after_trigger": list(self.after_trigger), "resign_plies": list(self.resign_plies),
                "counts": dict(self.counts)}

    def load_state_dict(self, state: dict) -> None:
        if not self.rc.enabled:
            return
        self.state, self.reason = state["state"], state["reason"]
        self.threshold, self.initial_threshold = state["threshold"], state["initial_threshold"]
        self.rng.bit_generator.state = state["rng"]
        self.recent.extend(state["recent"])
        self.verified.extend(state["verified"])
        self.after_trigger.extend(state["after_trigger"])
        self.resign_plies.extend(state["resign_plies"])
        self.counts.update(state["counts"])
        if self.rc.force:
            self.state, self.reason = ACTIVE, "active (forced by the owner)"


def add_resign_args(parser) -> None:
    group = parser.add_argument_group("resignation (TonkAI's own estimate only)")
    group.add_argument("--resign", action="store_true",
                       help="let TonkAI resign hopeless games; starts in shadow mode")
    group.add_argument("--resign-threshold", type=float, default=None)
    group.add_argument("--resign-survival-plies", type=int, default=10)
    group.add_argument("--resign-victory-score", type=float, default=0.05)
    group.add_argument("--resign-patience", type=int, default=3)
    group.add_argument("--resign-min-ply", type=int, default=40)
    group.add_argument("--resign-playout", type=float, default=0.10)
    group.add_argument("--resign-min-verified", type=int, default=200)
    group.add_argument("--resign-min-non-loss", type=float, default=0.10)
    group.add_argument("--resign-force", action="store_true",
                       help="always resign when triggered: no shadow mode and no played-out verification")


def resign_config_from_args(args) -> ResignConfig:
    rc = ResignConfig(enabled=bool(args.resign), threshold=args.resign_threshold,
                      survival_plies=args.resign_survival_plies,
                      victory_score=args.resign_victory_score,
                      patience=args.resign_patience, min_ply=args.resign_min_ply,
                      playout=args.resign_playout, min_verified=args.resign_min_verified,
                      min_non_loss=args.resign_min_non_loss, force=bool(getattr(args, "resign_force", False)))
    rc.validate()
    return rc
