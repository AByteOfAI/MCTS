# Monte Carlo Tree Search with Deep Reinforcement Learning for Robotic Navigation

<p align="center">
  <strong>EEE598 Final Project — Fall 2024 | Arizona State University</strong><br>
  Team 14: Sakshi Lathi · Abhijit Sinha · Anusha Chatterjee
</p>

---

## Overview

This repository contains our research paper and presentation for EEE598 (Fall 2024) at Arizona State University. We investigate **Monte Carlo Tree Search (MCTS)** and its integration with **Deep Reinforcement Learning (DRL)** for solving the **robotic follow-ahead navigation** problem — where a robot must lead a human target while dynamically avoiding obstacles and maintaining line-of-sight.

> **📄 Full Paper:** [`Team14_Lathi_Sinha_Chatterjee_MonteCarlo.pdf`](Team14_Lathi_Sinha_Chatterjee_MonteCarlo.pdf)

---

## Problem Statement

In robotic follow-ahead scenarios, a robot must navigate *in front of* a human while simultaneously:

- **Predicting human intentions** — anticipating the target's future movement in real time
- **Avoiding obstacles** — dynamically replanning around physical barriers
- **Preventing occlusion** — maintaining an unobstructed line-of-sight to the human
- **Ensuring safety** — preventing collisions in shared human-robot spaces

Existing methods typically address follow-behind or side-by-side configurations and struggle with the vast state spaces and unpredictability inherent in leading a human through dynamic environments.

---

## How MCTS Works

MCTS builds a decision tree incrementally through four iterative stages:

```
┌─────────────┐
│  Selection   │  Navigate tree using UCB to balance exploration vs. exploitation
└──────┬──────┘
       ▼
┌─────────────┐
│  Expansion   │  Add child nodes representing unexplored actions
└──────┬──────┘
       ▼
┌─────────────┐
│  Simulation  │  Run rollouts to estimate future rewards
└──────┬──────┘
       ▼
┌──────────────────┐
│ Backpropagation  │  Update node statistics along the selected path
└──────────────────┘
```

Node selection is guided by the **Upper Confidence Bound (UCB)** formula:

```
UCB = w / nᶜ  +  c · √(ln nᵖ / nᶜ)
```

| Symbol | Meaning |
|--------|---------|
| `w` | Value of the node (expected reward) |
| `nᶜ` | Visit count of the child node |
| `nᵖ` | Visit count of the parent node |
| `c` | Exploration constant (typically **1.4**) |

---

## MCTS-DRL Framework

The key contribution of our study is the integration of MCTS with a **trained DRL policy** that replaces random rollouts during the simulation phase:

```
  Human Trajectory        Occupancy Map        Robot Pose
  Prediction                   │                   │
       │                      │                   │
       └──────────┬───────────┘───────────────────┘
                  ▼
       ┌─────────────────────┐
       │   MCTS-DRL Engine   │
       │                     │
       │  1. Expand tree     │
       │  2. Check collision │──── collision? → prune node
       │  3. Check occlusion │──── occluded? → penalize (−1)
       │  4. DRL evaluation  │──── Q(o, aᵢ) reward estimate
       │  5. Backpropagate   │
       │  6. Select best UCB │
       └─────────┬───────────┘
                 ▼
          Navigational Goal
           (c = 0, exploit)
```

---

## Key Results

### MCTS-DRL vs. Standalone Methods

| Metric | DRL Only | MCTS Only | **MCTS-DRL** |
|---|---|---|---|
| Trajectory Accuracy | Moderate | Inconsistent | **Excellent** |
| Obstacle Avoidance | Limited | Moderate | **High** |
| Occlusion Handling | Poor | Moderate | **High** |
| Mean Reward | −18.4 | 3.2 ± 5.9 | **5.4** |

### Cumulative Rewards by Trajectory (20 trials)

| Human Trajectory | DRL | MCTS | **MCTS-DRL** |
|---|---|---|---|
| Circular | −17.95 | 2.87 ± 5.96 | **4.53** |
| S-shaped | −21.84 | −3.83 ± 4.33 | **−1.61** |

### SL-MCTS vs. Traditional MCTS

| Metric | Traditional MCTS | **SL-MCTS** |
|---|---|---|
| Success Rate | 78% | **92%** |
| Avg. Path Length | 15 steps | **12 steps** |
| Computation Time | 2.4s | **1.3s** |

The MCTS-DRL hybrid consistently outperforms both standalone approaches across straight, U-shaped, S-shaped, and L-shaped test trajectories.

---

## Applications Discussed

| Domain | Description |
|---|---|
| **Robotic Follow-Ahead** | Robot leads a human through dynamic environments with obstacle and occlusion avoidance |
| **Multi-Agent Pathfinding** | Autonomous warehouse robots navigating around shelves and each other |
| **Wearable Exoskeletons** | Real-time gait assistance that adapts to patient feedback |
| **Humanoid Robotics** | Task planning and safe human interaction (e.g., Tesla Optimus) |

---

## Repository Contents

```
MCTS/
├── README.md                                        # This file
└── Team14_Lathi_Sinha_Chatterjee_MonteCarlo.pdf     # Full paper + presentation slides
```

---

## References

1. Leisiazar, S., Park, E. J., Lim, A., & Chen, M. (2023). *An MCTS-DRL Based Obstacle and Occlusion Avoidance Methodology in Robotic Follow-Ahead Applications.*

2. Li, W., Liu, Y., Ma, Y., Xu, K., Qiu, J., & Gan, Z. (2023). *A Self-Learning Monte Carlo Tree Search Algorithm for Robot Path Planning.* Frontiers in Neurorobotics.

3. *Robust walking control of a lower limb rehabilitation exoskeleton coupled with a musculoskeletal model via deep reinforcement learning.*

---

## Team

| Name | Email |
|---|---|
| Sakshi Lathi | [slathi@asu.edu](mailto:slathi@asu.edu) |
| Abhijit Sinha | [asinh117@asu.edu](mailto:asinh117@asu.edu) |
| Anusha Chatterjee | [achatt53@asu.edu](mailto:achatt53@asu.edu) |

**Course:** EEE598 — Fall 2024, School of Electrical, Computer, and Energy Engineering, Arizona State University

---

<p align="center">
  <em>Built with ❤️ at Arizona State University</em>
</p>
