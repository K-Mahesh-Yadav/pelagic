# A shark that learns to hunt

A school of fish, a shark that learns to hunt them, and a live view of every
neuron in its head while it does.

Both sides evolve. The shark's brain is a small neural network bred by
selection, with no ML library and no gradients. The fish carry inherited
traits that are under selection at the same time, so the school gets harder
to catch as the shark gets better at catching it.

Everything runs in one self-contained HTML file. No build step, no server,
no dependencies. Open it and it starts training.

**Live demo:** https://k-mahesh-yadav.github.io/shark-that-learns/
**Demo video:** _(add your link)_

---

## Run it

Open `index.html` in any modern browser. That's it.

Training begins immediately from random weights, which means the first
couple of minutes are a shark blundering around empty water. To see a
competent hunter straight away:

1. Open `demo-brain.json` and copy the whole file.
2. In the page, go to **Keep this brain** and press **Load weights**.
3. Paste, then press **Load these weights**.

That brain is 150 generations old and catches about 8 fish per 660-tick run.
Training continues from there rather than restarting.

---

## What you are looking at

**The water (left).** Bone-coloured fish, a shark, and the scent plume the
school leaves behind it. Fish turn rust-coloured when they bolt.

**The shark's brain (right, top).** A live 14–10–1 network:

- Input circles are the actual sensor values that tick. Colour is sign,
  opacity is magnitude.
- Hidden circles are the real post-tanh activation of each unit.
- The output circle is the turn command steering the shark you are watching.
- Line thickness is the real weight; brass is positive, blue negative.
- Grey nodes are senses the curriculum has not unlocked yet. Their weights
  exist but are multiplied by zero.
- The spine beside the hidden column is memory, lit once stage 8 arrives.

It redraws every second frame, so it samples the state rather than showing
every tick. Connections below |w| = 0.13 are not drawn, for legibility.

**What each faculty is worth.** Every generation the champion hunts
identical water up to three times: intact, with its nose blocked, and with
its memory blocked. Same brain, same fish, same start.

**The school.** The seven inherited traits, each with a tick mark showing
where it started and a band showing the spread across the population.

**Training.** Generation, curriculum stage, and the catch curve.

### Controls

| Control | What it does |
|---|---|
| Pause the water | Freezes the display and gives the whole frame budget to training |
| Run at ×3 | Three simulation ticks per frame |
| Hide the scent | Toggles the plume overlay |
| Show what it sees | Draws the 160-unit vision cone and both nostrils |
| Hand it everything | Skips the curriculum straight to full sense plus memory |
| Start evolution over | Fresh random population on a random seed |
| Print on paper | Light theme — the same study as an ink plate |

---

## How it works

### The school

Reynolds' boids: separation, alignment, cohesion, plus a flight term away
from the predator. Each fish only sees neighbours inside its own perception
radius, which is what lets a school split around the shark and re-form
behind it.

### The shark

A 14–10–1 network with tanh units and 161 weights. Fourteen senses:

    bias, how close, fish ahead, fish beside, closing,
    wall ahead, wall astern, wall left, wall right,
    shoal ahead, shoal beside,
    scent here, scent beside, scent ahead

One output: how hard to turn. Speed falls off with turn rate, so hard
cornering costs it ground.

Sight reaches 160 units in a 150° forward cone and stops. The wall senses
deliberately carry at any distance — without them, a shark with nothing in
view reads a constant input vector, and a constant turn is a circle.

Ten of the weights are a leaky self-connection, one per hidden unit, giving
each a time constant. That is the brain's only memory. The leak is clamped
to [0, 0.9]: a negative value makes a unit oscillate against itself every
tick, and anything past 1.0 latches it solid. Neither is a time constant.

### The scent

Every fish deposits scent each tick into a grid that diffuses and decays
with a half-life of about 1.5 seconds, so the plume behind a school is a
record of where it has just been. The shark reads it with two nostrils and
a reference point astern, comparing left against right rather than being
handed a direction.

Measured falloff: about 20 units of concentration within 60 units of a
fish, 5.7 at 120–180, 0.25 at 420–480, nothing beyond. The nose reaches
roughly 2.7× further than the eye, but vaguely.

### Training

Neuroevolution. Population of 24, each evaluated for 660 ticks in its own
arena. The top 4 are carried forward unchanged; the rest are uniform
crossover plus Gaussian mutation. All 24 face identical starting conditions
within a generation, and that environment is held for three generations so
an adaptation survives long enough for selection to notice it.

Senses are handed over in eight stages rather than all at once. The reason
matters: a new sense whose weights have been drifting arrives as *noise in
a working brain*, and the fastest route back to fitness is to silence it.
So newly unlocked weights are **zeroed** — they arrive silent, and can only
be recruited if they pay for themselves — then mutated hard for seven
generations to search the new subspace while leaving the rest alone.

### The prey

Each fish inherits seven traits: perception radius, separation, alignment,
cohesion, flee strength, flee distance, and escape bearing. When one is
eaten its place goes to a mutated copy of a survivor, chosen in proportion
to how well that survivor has been feeding.

That feeding term is the whole balance. A fish that is bolting is not
eating, and watchfulness costs something even when nothing happens, so
paranoia is paid for in descendants rather than corpses. Without it the
school evolves into maximally jumpy sprinters and the arms race runs away.

The escape bearing is the one trait that needs no invented cost. Breaking
sideways keeps you nearer the thing chasing you, and breaking too far
points you back into it, so the price is geometry. Measured against a fixed
shark: 0° loses 10.1 fish, 36° loses 6.5, 90° loses 15.5, 120° loses 26.3.
An evolved school independently converged on 31–37°.

Which side a fish breaks to is a coin flip at birth and is not inherited.
If it were, the population would converge on one turning direction and the
school would wheel as one body instead of fanning apart.

---

## Files

| File | What it is |
|---|---|
| `index.html` | The whole thing. Simulation, training, and UI in one file. |
| `demo-brain.json` | A 150-generation champion. Paste it into **Load weights**. |
| `pelagic_brain.py` | Load, inspect and run a saved brain offline. No dependencies. |

```
python pelagic_brain.py demo-brain.json            # readable report
python pelagic_brain.py demo-brain.json --demo     # watch it steer
python pelagic_brain.py demo-brain.json --npz out.npz
```

```python
from pelagic_brain import Brain
b = Brain.load("demo-brain.json")
turn = b.step({"how close": 0.8, "fish ahead": 0.6, "fish beside": -0.3})
```

The saved file records which senses were unlocked, so a brain saved
mid-curriculum holds the not-yet-given inputs at zero in Python exactly as
the browser did.

---

## Honest notes

**The numbers move.** The nose is worth roughly 1.1–1.3× and memory about
1.1–1.2× across the runs measured, but a single generation's trial can
easily show either one losing. Read the running mean, not one trial. If you
quote a figure, re-run it and quote your own.

**Once both sides evolve, the catch curve stops being comparable.** A shark
at generation 120 may be better than one at generation 40 and still eat
less, because it is hunting a better school. The trait chart is the other
half of that story.

**This is a simulated world, not a real one.** The constants — speeds,
vision, vigilance costs — were tuned by sweeps until the behaviour was
balanced, not derived from measurements of animals. Reynolds wrote his
paper for computer graphics, after biologically inspired motion that looks
right rather than statistical accuracy, and that caveat applies here too.

**Initial weights are `N(0, 0.7²)` with no fan-in scaling.** With 14 inputs
that saturates a fair share of the starting population, so early
generations steer harder than they need to. Glorot scaling would suggest
about 0.27. Untested, and probably worth fixing.

**The shark's olfaction uses concentration differences between nostrils.**
Gardiner & Atema's actual finding is that *arrival-time* differences do the
work in real sharks, and that the concentration-comparison model is the
assumption their paper pushes against. This implementation uses the simpler,
older model.

---

## Credits

- Reynolds, C. W. (1987). *Flocks, Herds, and Schools: A Distributed
  Behavioral Model.* SIGGRAPH '87.
  [PDF](https://www.red3d.com/cwr/papers/1987/SIGGRAPH87.pdf)
- Hamilton, W. D. (1971). *Geometry for the Selfish Herd.* J. Theor. Biol.
  31(2), 295–311.
  [Link](https://www.sciencedirect.com/science/article/pii/0022519371901895)
- Gardiner, J. M. & Atema, J. (2010). *The Function of Bilateral Odor
  Arrival Time Differences in Olfactory Orientation of Sharks.* Current
  Biology 20, 1187–1191.
  [Link](https://www.cell.com/current-biology/fulltext/S0960-9822(10)00591-9)
