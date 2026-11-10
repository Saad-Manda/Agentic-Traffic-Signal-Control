# How it works: data, simulator and training

A plain-language explanation of what our dataset contains, what the CityFlow simulator does
with it, and how an RL agent is trained from it. Written for the project team, mentors and
viva preparation. Formal details: `docs/dataset_card.md` (data), `.claude/reference/architecture.md`
(model), `.claude/reference/database.md` (split and leakage rules).

> The decision interval (15 s), reward (−pressure), state, metrics and seeds are **frozen**
> in `docs/evaluation_protocol.md` (10 Nov 2026). Values still marked *(provisional)* (number
> of training episodes) are hyperparameters chosen on the validation flow.

---

## 1. The key idea: the dataset is a scenario, not a labelled table

In supervised learning a dataset is a table of inputs and correct answers, for example
`image → "cat"`, and the model learns to reproduce the answers.

**Our dataset has no answers.** No file says "at 08:05 the correct signal was green for
north–south". The data describes only the **traffic demand**: which vehicles enter the road
network, when, and which roads they want to take.

The simulator plays this demand out second by second. The agent's job is to **operate the
traffic lights** while that happens. It learns by acting, observing the queues its actions
cause, and improving. This is reinforcement learning (RL).

> **Analogy.** A driving-school simulator: the "dataset" is the level (the roads and the other
> cars). Nobody supplies the correct steering angle for every second. The learner drives the
> level many times and improves from the score.

| | Supervised learning | Our project (RL) |
|---|---|---|
| Dataset contains | inputs + correct labels | road map + vehicle trips (no labels) |
| Training examples come from | the dataset directly | **the agent's own interaction** with the simulator |
| Learning signal | error vs. label | **reward** (how bad the congestion is) |
| "Test set" | unseen rows | **unseen traffic scenarios** (other flow files, another city) |

---

## 2. What the data looks like

Each city has two kinds of JSON file (CityFlow format, from the CoLight release).

### 2.1 The road network (`roadnet_3_4.json`): the map

Jinan is a grid of **4 columns × 3 rows = 12 signalised intersections**. Intersections are
400 m apart east–west and 800 m apart north–south. Every road has 3 lanes (left-turn,
straight, right-turn) and a 40 km/h (11.1 m/s) speed limit. The ring of "virtual"
intersections around the edge is where vehicles enter and leave. They have no signals and
are not agents.

```
   y=4   ○      ○      ○      ○              ● signalised intersection = one RL agent
         │      │      │      │              ○ virtual boundary node (entry/exit)
   y=3 ○─●──────●──────●──────●─○
         │      │      │      │
   y=2 ○─●──────●──────●──────●─○            ↑ north = increasing y
         │      │      │      │              → east  = increasing x
   y=1 ○─●──────●──────●──────●─○
         │      │      │      │
   y=0   ○      ○      ○      ○
        x=1    x=2    x=3    x=4
```

Each signalised intersection has **8 green phases**. A phase is a set of movements that may
go at the same time, for example "north and south straight" or "east and west left turns". Right
turns are always allowed. There is also **phase 0, a 5-second transition** (the "yellow")
inserted whenever the signal changes.

Road IDs follow `road_<x>_<y>_<direction>`: the road *leaving* intersection (x, y) heading
0 = east, 1 = north, 2 = west, 3 = south.

### 2.2 The traffic flow (`anon_3_4_jinan_real.json`): the vehicles

A list of **6,295 vehicle trips over one hour** (Jinan training file). These are the first three,
unedited:

```json
{"startTime": 0,  "route": ["road_0_2_0", "road_1_2_0", "road_2_2_0", "road_3_2_1", "road_3_3_1"]}
{"startTime": 5,  "route": ["road_0_2_0", "road_1_2_0", "road_2_2_1", "road_2_3_2", "road_1_3_2"]}
{"startTime": 10, "route": ["road_0_2_0", "road_1_2_0", "road_2_2_0", "road_3_2_0", "road_4_2_0"]}
```

Each record also has fixed vehicle properties (length 5 m, max speed 11.1 m/s, acceleration
2 m/s², and so on), identical for all vehicles.

**Reading the first record:** at second 0 a car enters at the west edge of row 2. It drives
east straight through intersections (1,2) and (2,2), **turns left (north)** at (3,2), goes
straight through (3,3), and leaves at the north edge. It crosses **four signalised
intersections**, and at each one it must wait if its movement has a red light.

That is the entire dataset: *when* each car appears and *which roads* it takes. It does
**not** contain speeds, waiting times or travel times. Those are *outcomes* that depend on
how the lights are operated, which is exactly what we control.

The trips were reconstructed by the CoLight authors from real roadside-camera observations,
so the demand pattern (volume, turning proportions, busy directions) is real.

---

## 3. What the simulator does

**CityFlow** is a microscopic traffic simulator: it tracks every individual vehicle. Each
simulated second it:

1. **inserts** the vehicles whose `startTime` has arrived, at the first road of their route;
2. **moves** every vehicle with a car-following model: accelerate up to the speed limit,
   keep a safe gap, brake for the vehicle ahead;
3. **stops** vehicles at red signals, so queues form behind them;
4. **routes** vehicles onto the next road of their route when their movement is green;
5. **removes** vehicles that have reached the end of their route.

The one thing CityFlow does **not** decide is **which phase each signal shows**. That input
comes from our controller.

```
                     controller decides the signal phase
                                   │
                                   ▼
   flow file  ──►  ┌──────────────────────────────┐  ──►  measurements:
   (who enters,    │  CityFlow: move all vehicles │       queue per lane,
    when, where)   │  for one second              │       vehicles per lane,
                   └──────────────┬───────────────┘       waiting vehicles,
                                  └── repeat 3600× ──┘     travel times
```

### Evidence that the controller is what matters

Same map, same 6,295 vehicles, only the signal policy changed (our CityFlow benchmark,
worklog/007):

| Controller | Average travel time | Mean queue |
|---|---|---|
| Fixed-Time: cycle through phases, 30 s each | 515 s | 431 vehicles |
| Max-Pressure: every 10 s, serve the movement with the largest queue imbalance | 375 s | 247 vehicles |

Changing only the rule for the lights made trips **27 % faster**. The aim of the project is
a learned controller that does better still, and stays safe when it is unsure.

---

## 4. How the model is trained

### 4.1 The decision loop

Every **Δt = 15 simulated seconds** each of the 12 intersection agents runs
the loop below:

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. OBSERVE   state s : 44 numbers — queue, vehicles and pressure on    │
 │              the 12 incoming lanes / movements, each as a share of the │
 │              lane's capacity, + current phase (+ neighbours' states)   │
 │ 2. ACT       action a : choose one of the 8 green phases               │
 │ 3. SIMULATE  CityFlow runs 15 s (5 s transition first if phase changed)│
 │ 4. REWARD    r = −pressure  ≈ −(cars waiting to enter − cars waiting   │
 │              downstream)   (frozen)      — less congestion = higher r  │
 │ 5. STORE     the experience (s, a, r, s′) in a replay buffer           │
 │ 6. LEARN     sample past experiences, update the neural network        │
 └────────────────────────────────────────────────────────────────────────┘
```

This is a **Markov decision process (MDP)**: state, action, reward, next state.

**Each intersection keeps its own clock.** An intersection that keeps its phase decides again
15 s later; one that changes phase first shows 5 s of yellow, so its next decision comes 20 s
later. After the first change the 12 intersections are no longer in step, so the simulator
stops whenever *any* of them is due (always on a 5-second grid) and only those intersections
act. One experience (s, a, r, s′) is recorded per intersection per finished interval
(`trafficrl.envs.traffic_env`, ROADMAP 4.1).

**How the state is scaled.** A lane 400 m long holds at most 53 vehicles (400 m / 7.5 m per
vehicle including the gap); an 800 m lane holds 106. Every lane reading is divided by that
capacity, so each input is between 0 (empty) and 1 (full). Nothing is learned from data for
this step: it uses only the road lengths, so no information from validation or test traffic
can leak into it, and the same rule works for Hangzhou's 600 m roads. In the training hour no
lane ever exceeded 96 % of its capacity (`notebooks/01_eda_train.ipynb`).

### 4.2 Where the training examples come from

The neural network learns from the **experience tuples (s, a, r, s′) the agent itself
generates**, not from the flow file directly. One episode (one simulated hour) gives
3600 s / 15 s = at most 240 decisions × 12 intersections ≈ **2,900 experiences** (fewer
when 5 s transitions are inserted). The flow file only
fixes *which traffic the agent faces*. The agent's actions decide what happens.

### 4.3 What the network learns

We use **Deep Q-Networks (DQN)**. For a state *s*, the network outputs one number per phase:
**Q(s, a)**, the predicted total future reward if phase *a* is chosen now. The agent picks the
phase with the highest Q. Training nudges each prediction toward

```
target = r + γ · max_a′ Q(s′, a′)      (γ ≈ 0.9–0.99: how much the future counts)
```

that is, "the reward I just got, plus the best I expect from where I ended up". No human
label is needed. The reward and the network's own next estimate supply the target.

### 4.4 Episodes: replaying the same hour many times

Training repeats the **same training hour** for hundreds of episodes *(number provisional)*:

- **Early episodes:** the agent explores, choosing many phases at random (ε-greedy), and big
  queues form. The reward is low.
- **Later episodes:** it exploits what it has learned, for example "when the northbound queue
  is long and the eastbound one is empty, give north–south green". Queues shrink and
  the reward rises.

Replaying one hour is **not cheating**, because the agent is being taught a *policy*
(how to react to queues), not memorising the hour. Whether that policy generalises is
measured on *different* hours (section 5).

### 4.5 Many agents

There are 12 intersections, so 12 agents act at the same time, and one agent's decision
changes the traffic arriving at its neighbours. The baselines show the spectrum:

| Method | Learning | Coordination |
|---|---|---|
| Fixed-Time, Max-Pressure | none (hand-designed rules) | none |
| Single-agent DQN | one network shared by all 12 intersections | none (local view only) |
| IDQN (multi-agent) | an independent network per intersection | none (each sees only itself) |
| **UG-MARL (ours)** | one network per intersection, **ensemble of K heads** | **attention over neighbours' states** |

---

## 5. Train, validation, test and held-out

All splits are **different hours of real traffic**. Jinan files share the same map.

| Split | File | Vehicles | Role |
|---|---|---|---|
| **Train** | Jinan `real` | 6,295 | the traffic the agent practises on, many episodes |
| **Validation** | Jinan `real_2000` | 4,365 | unseen traffic; choose hyperparameters, gate threshold τ and the best checkpoint |
| **Test** | Jinan `real_2500` | 5,494 | run **once** at the end, learning switched off; the reported numbers |
| **Held-out** | Hangzhou (3 files) | 2,983 / 6,538 / 6,984 | a **different city and map**; does the policy transfer without retraining? |

**Why the separation matters.** An agent evaluated on its own training hour might simply be
tuned to that hour's quirks. That is the RL version of data leakage. Measuring on hours
it never saw (and a city it never saw) is the only honest evidence that it learned
something general. The split is enforced in code (`data/processed/splits.json`,
`trafficrl.data.splits.flow_files`, `tests/data/test_splits.py`).

**Evaluation metrics** *(final set frozen in 3.5)*: average travel time (main metric), queue
length, waiting time, throughput (vehicles completing their trip), each reported as
mean ± std and 95 % CI over ≥ 3 seeds.

---

## 6. Our contribution: the uncertainty gate

A learned controller can make a confident-looking but bad decision in a situation it never
practised, such as an unusual burst of traffic. That is a real risk for a traffic light.

UG-MARL gives each agent **K Q-heads** (an ensemble) instead of one, each trained slightly
differently:

```
                    ┌─► head 1: Q-values ─┐
  state + ─► shared ├─► head 2: Q-values ─┼─► mean Q  ──►  σ(s) ≤ τ ?  ── yes ─► phase = argmax mean Q
  neighbours encoder├─► …                 │   spread σ(s)              │
                    └─► head K: Q-values ─┘                            └─ no ──► phase = Max-Pressure
```

- **Heads agree** (low σ): the situation is familiar, so use the learned decision.
- **Heads disagree** (high σ): the agent hasn't really learned this situation (*epistemic
  uncertainty*), so **fall back to Max-Pressure**, a rule with a proof of queue stability
  (Varaiya 2013). This bounds how bad the controller can get.
- **Explainability:** how often the gate fires, where and when, plus attention weights
  (which neighbours an agent paid attention to). We report these as associations, not causes.

The ablations remove one piece at a time (no gate, no attention, no ensemble) to show what
each contributes.

---

## 7. Glossary

| Term | Meaning here |
|---|---|
| **Episode** | one full simulated hour of one flow file |
| **Phase** | a set of movements given green together (8 per intersection + transition) |
| **Δt** | green time between agent decisions (15 s, frozen) |
| **Queue** | vehicles stopped (speed < 0.1 m/s) on a lane |
| **Pressure** | queue on incoming lanes minus queue on the outgoing lanes they feed; high = this movement needs green |
| **Q-value** | predicted total future reward of choosing a phase now |
| **Replay buffer** | memory of past (s, a, r, s′) experiences sampled for training |
| **ε-greedy** | explore with probability ε (random phase), otherwise pick the best known phase |
| **Ensemble / σ(s)** | K Q-heads; σ is their disagreement, used as uncertainty |
| **Max-Pressure** | rule-based controller: serve the phase with the highest pressure |
| **Held-out city** | dataset never used for training or tuning, only for final generalisation tests |
