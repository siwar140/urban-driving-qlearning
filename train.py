# -*- coding: utf-8 -*-
"""
Entraînement Q-Learning — Agent de conduite urbaine
=====================================================
Exécuter : python train.py
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from car_env import UrbanCarEnv
from q_agent  import QLearningAgent
from test_scenarios import run_all


def train():
    env   = UrbanCarEnv()
    agent = QLearningAgent(
        num_states  = env.NUM_STATES,
        num_actions = env.NUM_ACTIONS,
        alpha       = 0.20,
        gamma       = 0.95,
        epsilon     = 1.0,
        q_init      = 1.0,
    )

    num_episodes   = 80_000
    eps_end        = 0.01
    eps_decay      = (1.0 - eps_end) / (num_episodes * 0.80)
    alpha_start    = 0.20
    alpha_end      = 0.04

    rewards_hist = []
    crash_hist   = []

    print(f"Environnement : {env.NUM_STATES} etats x {env.NUM_ACTIONS} actions")
    print(f"Entrainement  : {num_episodes:,} episodes\n")
    print("=" * 65)

    for ep in range(num_episodes):
        ratio         = ep / num_episodes
        agent.alpha   = alpha_start - ratio * (alpha_start - alpha_end)
        agent.epsilon = max(eps_end, agent.epsilon - eps_decay)

        state = env.reset()
        idx   = env.get_state_index(state)
        done  = False
        ep_r  = 0.0

        while not done:
            action              = agent.choose_action(idx, is_training=True)
            next_state, r, done = env.step(action)
            next_idx            = env.get_state_index(next_state)
            agent.learn(idx, action, r, next_idx, env.done)
            idx  = next_idx
            ep_r += r

        rewards_hist.append(ep_r)
        crash_hist.append(1 if env.crashed else 0)

        if (ep + 1) % 4_000 == 0:
            w     = 4_000
            avg_r = np.mean(rewards_hist[-w:])
            cr    = np.mean(crash_hist[-w:]) * 100
            print(f"Ep {ep+1:>6}/{num_episodes}  eps={agent.epsilon:.3f}"
                  f"  lr={agent.alpha:.3f}  Recomp.moy={avg_r:+7.2f}"
                  f"  Crash={cr:5.1f}%")

    agent.save("q_table.npy")

    # ── Évaluation ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION PHASE (eps=0)")
    run_all(agent.Q, n=500)

    # ── Courbes ───────────────────────────────────────────────────────────
    W   = 500
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q-Learning Training — Urban Driving Agent")

    sm_r = np.convolve(rewards_hist, np.ones(W) / W, mode="valid")
    axes[0].plot(rewards_hist, alpha=0.12, color="steelblue", lw=0.5)
    axes[0].plot(sm_r, color="crimson", label=f"Moy. {W} ep.")
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_title("Reward per episode")
    axes[0].set_xlabel("Episodes")
    axes[0].legend()

    sm_c = np.convolve(crash_hist, np.ones(W) / W, mode="valid")
    axes[1].plot(sm_c * 100, color="darkorange")
    axes[1].set_title("Collision Rate (%)")
    axes[1].set_xlabel("Episodes")
    axes[1].set_ylim(0, 110)
    axes[1].axhline(5, color="green", lw=0.8, ls="--", label="5%")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("Courbes sauvegardees -> training_curves.png")


if __name__ == "__main__":
    train()