"""
pelagic_brain.py — load, inspect and run a shark brain trained in the page.

A brain saved from the page is a single JSON file. This module reads it,
rebuilds the network exactly as the browser runs it, and gives you three
things: a forward pass you can drive yourself, a readable report on what
the thing actually learned, and a converter to numpy / plain arrays.

No dependencies. numpy is used if present, and only for the .npz export.

    python pelagic_brain.py brain.json                 # report
    python pelagic_brain.py brain.json --demo          # watch it steer
    python pelagic_brain.py brain.json --npz out.npz   # export arrays

    from pelagic_brain import Brain
    b = Brain.load("brain.json")
    turn = b.step({"how close": 0.8, "fish ahead": 0.6, "fish beside": -0.3})

The network is 14 -> 10 -> 1 with tanh units. Ten of the weights are a
per-unit leak: each hidden unit can carry a fraction of its own previous
state, which is the brain's only memory. The leak is clamped to [0, 0.9]
exactly as the simulation clamps it — a negative value would just make the
unit oscillate against itself every tick, and a value past 1.0 latches it.

Weight layout inside the flat `weights` array:

    w_in    hidden * inputs values, hidden-major
            unit j reads weights[j*inputs : (j+1)*inputs]
    w_rec   hidden values, one leak per unit
    w_out   hidden values
    b_out   1 value
"""

from __future__ import annotations

import json
import math
import sys
from typing import Dict, Iterable, List, Optional, Sequence

FORMAT = "pelagic-brain"


class BrainError(Exception):
    """Raised when a file is not a usable saved brain."""


class Brain:
    """A trained shark brain, restored from a saved JSON file."""

    def __init__(self, data: dict):
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise BrainError("not a saved brain file")

        net = data.get("network") or {}
        self.n_in: int = int(net.get("inputs", 0))
        self.n_hid: int = int(net.get("hidden", 0))
        self.senses: List[str] = list(net.get("senses") or [])
        self.memory_on: bool = bool(net.get("memoryOn", True))

        # which senses were unlocked when this brain was saved; the rest were
        # held at zero during training and must be held at zero here too
        active = net.get("active")
        self.active: List[int] = (
            [int(a) for a in active] if active else [1] * self.n_in
        )

        w = data.get("weights")
        if not isinstance(w, list):
            raise BrainError("no weights in file")
        expected = self.n_in * self.n_hid + self.n_hid * 2 + 1
        if len(w) != expected:
            raise BrainError(
                f"expected {expected} weights for a {self.n_in}-{self.n_hid}-1 "
                f"network, found {len(w)}"
            )
        if len(self.senses) != self.n_in:
            raise BrainError("sense labels do not match the input count")

        cut_rec = self.n_in * self.n_hid
        cut_out = cut_rec + self.n_hid
        self.w_in: List[List[float]] = [
            [float(x) for x in w[j * self.n_in:(j + 1) * self.n_in]]
            for j in range(self.n_hid)
        ]
        self.w_rec: List[float] = [float(x) for x in w[cut_rec:cut_out]]
        self.w_out: List[float] = [float(x) for x in w[cut_out:cut_out + self.n_hid]]
        self.b_out: float = float(w[cut_out + self.n_hid])

        self.world: dict = data.get("world") or {}
        self.training: dict = data.get("training") or {}
        self.prey: dict = data.get("prey") or {}
        self._index = {name: i for i, name in enumerate(self.senses)}
        self.reset()

    # ---------------------------------------------------------------- io

    @classmethod
    def load(cls, path: str) -> "Brain":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def loads(cls, text: str) -> "Brain":
        return cls(json.loads(text))

    # ----------------------------------------------------------- running

    def reset(self) -> None:
        """Forget everything the memory units are holding."""
        self.hidden: List[float] = [0.0] * self.n_hid

    def leak(self, j: int) -> float:
        """The effective time constant of hidden unit j, after clamping."""
        if not self.memory_on:
            return 0.0
        r = self.w_rec[j]
        if r <= 0.0:
            return 0.0
        return 0.9 if r > 0.9 else r

    def vector(self, senses) -> List[float]:
        """Accept a dict keyed by sense name, or a plain sequence."""
        if isinstance(senses, dict):
            unknown = set(senses) - set(self._index)
            if unknown:
                raise BrainError("unknown sense(s): " + ", ".join(sorted(unknown)))
            x = [0.0] * self.n_in
            for name, value in senses.items():
                x[self._index[name]] = float(value)
            if "bias" in self._index:
                x[self._index["bias"]] = 1.0
            return x
        x = [float(v) for v in senses]
        if len(x) != self.n_in:
            raise BrainError(f"expected {self.n_in} sense values, got {len(x)}")
        return x

    def step(self, senses) -> float:
        """One tick. Returns the turn command, in [-1, 1].

        Positive turns one way and negative the other; the simulation
        multiplies this by its turn rate. Call reset() between runs, or the
        memory units carry state across what should be separate hunts.
        """
        x = self.vector(senses)
        prev = self.hidden
        nxt = [0.0] * self.n_hid
        for j in range(self.n_hid):
            row = self.w_in[j]
            s = 0.0
            for i in range(self.n_in):
                if self.active[i]:
                    s += row[i] * x[i]
            s += self.leak(j) * prev[j]
            nxt[j] = math.tanh(s)
        out = self.b_out
        for j in range(self.n_hid):
            out += self.w_out[j] * nxt[j]
        self.hidden = nxt
        return math.tanh(out)

    # --------------------------------------------------------- inspection

    def influence(self) -> List[tuple]:
        """How much each sense actually reaches the output.

        For sense i this sums |w_in[j][i]| * |w_out[j]| over the hidden
        units, which weights a connection by how much its unit matters
        downstream. Crude, but it ranks senses far better than raw |w_in|.
        """
        scores = []
        for i, name in enumerate(self.senses):
            total = sum(
                abs(self.w_in[j][i]) * abs(self.w_out[j]) for j in range(self.n_hid)
            )
            scores.append((name, total, bool(self.active[i])))
        return sorted(scores, key=lambda r: -r[1])

    def prey_traits(self) -> List[dict]:
        """Mean and spread of each inherited trait in the saved school."""
        traits = self.prey.get("traits") or []
        pool = self.prey.get("pool") or []
        if not traits or not pool:
            return []
        out = []
        for i, t in enumerate(traits):
            col = [float(g[i]) for g in pool if i < len(g)]
            if not col:
                continue
            mean = sum(col) / len(col)
            var = sum((c - mean) ** 2 for c in col) / len(col)
            lo, hi = float(t.get("min", 0.0)), float(t.get("max", 1.0))
            out.append({
                "name": t.get("name", f"trait {i}"),
                "mean": lo + mean * (hi - lo),
                "sd": math.sqrt(var) * (hi - lo),
                "min": lo,
                "max": hi,
            })
        return out

    def report(self) -> str:
        t, w = self.training, self.world
        L: List[str] = []
        add = L.append
        add(f"Shark brain — {self.n_in}\u2013{self.n_hid}\u20131, tanh")
        add(f"  generation {t.get('generation', '?')}, "
            f"stage {t.get('stage', '?')} of {t.get('stagesTotal', '?')}"
            f"   best single hunt: {t.get('bestSingleHunt', '?')} fish")
        add(f"  sight {w.get('vision', '?')} units in a "
            f"{w.get('fieldOfViewDegrees', '?')}\u00b0 cone"
            f"   memory {'on' if self.memory_on else 'off'}")

        if t.get("noseTrials"):
            add(f"  measured in training over {t['noseTrials']} paired trials:")
            add(f"    nose working {t.get('noseWorking')}  "
                f"vs nostrils blocked {t.get('nostrilsBlocked')}")
        if t.get("memoryTrials"):
            add(f"    memory working {t.get('memoryWorking')}  "
                f"vs memory blocked {t.get('memoryBlocked')}")

        add("")
        add("What reaches the output, strongest first")
        ranked = self.influence()
        top = max((r[1] for r in ranked), default=1.0) or 1.0
        for name, score, on in ranked:
            bar = "\u2588" * int(round(22 * score / top))
            tail = "" if on else "   (not yet given when saved)"
            add(f"  {name:<14} {score:6.2f} {bar}{tail}")

        add("")
        add("Memory, per hidden unit")
        any_mem = False
        for j in range(self.n_hid):
            r = self.leak(j)
            if r <= 0.001:
                continue
            any_mem = True
            # a leak r averages roughly 1/(1-r) ticks of history
            add(f"  unit {j}: leak {r:.3f}  \u2248 {1.0 / (1.0 - r):5.1f} ticks of history"
                f"   (weight to output {self.w_out[j]:+.2f})")
        if not any_mem:
            add("  none — every unit reacts to the present tick only")

        traits = self.prey_traits()
        if traits:
            add("")
            add("The school this brain was hunting")
            for tr in traits:
                add(f"  {tr['name']:<26} {tr['mean']:7.2f}  \u00b1{tr['sd']:.2f}"
                    f"   (range {tr['min']:g} to {tr['max']:g})")
        return "\n".join(L)

    # ---------------------------------------------------------- exporting

    def arrays(self) -> dict:
        """Plain nested lists, or numpy arrays when numpy is installed."""
        out = {
            "w_in": self.w_in,
            "w_rec": self.w_rec,
            "w_out": self.w_out,
            "b_out": self.b_out,
            "active": self.active,
        }
        try:
            import numpy as np
        except ImportError:
            return out
        return {
            "w_in": np.array(self.w_in, dtype=np.float32),
            "w_rec": np.array(self.w_rec, dtype=np.float32),
            "w_out": np.array(self.w_out, dtype=np.float32),
            "b_out": np.float32(self.b_out),
            "active": np.array(self.active, dtype=np.int8),
        }

    def to_npz(self, path: str) -> None:
        try:
            import numpy as np
        except ImportError:
            raise BrainError("numpy is needed for .npz export")
        np.savez(path, **self.arrays())


def _demo(brain: Brain) -> str:
    """Drive the brain through a scripted encounter and show what it does."""
    lines = ["Turn command through a scripted encounter",
             "(positive and negative are opposite directions)", ""]

    def run(label, frames):
        brain.reset()
        row = []
        for senses in frames:
            usable = {k: v for k, v in senses.items() if k in brain._index}
            row.append(f"{brain.step(usable):+.3f}")
        lines.append(f"  {label:<36} {'  '.join(row)}")

    # Crossing open water with nothing in view. The walls still register, and
    # their readings shift as it travels, so the turn command keeps moving.
    run("blind, crossing open water", [
        {"wall ahead": 0.15 + 0.13 * k, "wall astern": 0.80 - 0.12 * k,
         "wall left": 0.30, "wall right": 0.10 + 0.05 * k}
        for k in range(6)
    ])
    # The same thing with the wall senses held at zero: the input vector is a
    # constant, so the output is a constant, so the path is a closed circle.
    # This is the dead state the simulation had before the wall range was
    # un-truncated, kept here because it is worth being able to see.
    run("blind, with the walls not felt", [{} for _ in range(6)])
    run("a fish appears off to one side", [
        {"how close": 0.4, "fish ahead": 0.5, "fish beside": 0.8,
         "wall astern": 0.4} for _ in range(6)
    ])
    run("closing on it, dead ahead", [
        {"how close": 0.9, "fish ahead": 1.0, "fish beside": 0.05,
         "closing": 0.7, "wall astern": 0.4} for _ in range(6)
    ])
    run("lost it, scent trails to one side", [
        {"scent here": 0.5, "scent beside": 0.7, "scent ahead": 0.3,
         "wall ahead": 0.3, "wall astern": 0.5} for _ in range(6)
    ])
    lines.append("")
    lines.append("  Compare the first two rows. Identical numbers mean a fixed turn")
    lines.append("  rate, which is a circle; a drifting number means it is quartering")
    lines.append("  the water. Everything after the first two rows holds its senses")
    lines.append("  fixed, so those rows settle as the memory units fill up.")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    args = list(argv[1:])
    if not args:
        print(__doc__.strip())
        return 1
    path = args[0]
    try:
        brain = Brain.load(path)
    except (OSError, json.JSONDecodeError, BrainError) as exc:
        print(f"could not load {path}: {exc}", file=sys.stderr)
        return 2

    if "--npz" in args:
        dest = args[args.index("--npz") + 1]
        brain.to_npz(dest)
        print(f"wrote {dest}")
        return 0

    print(brain.report())
    if "--demo" in args:
        print()
        print(_demo(brain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
