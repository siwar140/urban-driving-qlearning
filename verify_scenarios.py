# -*- coding: utf-8 -*-
"""
Vérifie, épisode par épisode, que l'AGENT FAIT ce que le SCÉNARIO décrit.
Usage : python "Verify scenarios" [q_table.npy]

Règles contrôlées (à partir des conditions réelles de l'épisode) :
  Feu ROUGE   → vitesse nulle à t=T_LIGHT, démarre au feu jaune sans courir, puis repart après le feu
  Feu VERT    → en marche à t=T_LIGHT
  Feu JAUNE   → démarre mais ne roule pas vite
  Obstacle dans MA voie → a quitté sa voie avant T_OBSTACLE, sans collision
  Obstacle autre voie   → reste dans sa voie, sans collision
  Pas d'obstacle        → reste dans sa voie
  Piéton      → arrêté pendant toute sa traversée (T-5..T+5), puis repart
  Sans piéton → en marche à t=T_PED
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from collections import defaultdict
from car_env import UrbanCarEnv
from q_agent  import QLearningAgent
from test_scenarios import SCENARIOS

ACT = ["Dem", "Acc", "Frein", "Maint", "Voie"]


def play(env, agent, ov):
    s = env.reset(ov); i = env.get_state_index(s); d = False
    spd, lane, act = [], [], []
    while not d:
        a = agent.choose_action(i, is_training=False)
        s, _, d = env.step(a); i = env.get_state_index(s)
        spd.append(env.speed); lane.append(env.lane); act.append(a)
    return spd, lane, act


def at(arr, t):
    return arr[t] if t < len(arr) else None


def check(env, spd, lane, act=None):
    """Retourne {règle: True/False} selon les conditions réelles de l'épisode."""
    E, res = env, {}
    TL, TO, TP, H = E.T_LIGHT, E.T_OBSTACLE, E.T_PED, E.PED_HALF
    # ── Feu ──
    v = at(spd, TL)
    if v is not None:
        if E.light_color in (0, 2):
            res["feu rouge: arrete a T"]  = v <= 0.01
            # Au feu jaune (TL + 1): démarre mais ne marche pas (vitesse modérée)
            act_j = at(act, TL + 1) if act is not None else None
            spd_j = at(spd, TL + 1)
            if act_j is not None and spd_j is not None:
                res["feu jaune: demarre sans courir"] = (act_j == 0) and (spd_j <= E.START_MAX)
            w = at(spd, TL + E.ZONE_HALF)
            res["feu rouge: repart"]      = (w is None) or w > 0.5
        else:
            res["feu vert: roule"] = v >= 0.05
    # ── Obstacle ──
    l = at(lane, TO)
    if l is not None:
        if E.car_ahead and E.car_lane == 1:
            res["obstacle meme voie: evite"]  = (l != 1) and not E.crashed
        elif E.car_ahead:
            res["obstacle autre voie: reste"] = (l == 1) and not E.crashed
        else:
            res["route libre: reste en voie"] = (l == 1)
    # ── Piéton ──
    if E.pedestrian:
        win = [at(spd, t) for t in range(TP - H, TP + H + 1)]
        if all(x is not None for x in win):
            res["pieton: arrete pendant traversee"] = max(win) <= 0.01
            w = at(spd, TP + E.ZONE_HALF)
            res["pieton: repart"] = (w is None) or w > 0.5
    else:
        v = at(spd, TP)
        if v is not None:
            res["sans pieton: roule"] = v >= 0.5
    return res


def timeline(act):
    runs = []
    for j, a in enumerate(act):
        if runs and runs[-1][0] == a: runs[-1][2] = j
        else: runs.append([a, j, j])
    return " ".join(f"{ACT[a]}[{b}-{c}]" if b != c else f"{ACT[a]}[{b}]" for a, b, c in runs)


def main(path="q_table.npy", n=200):
    Q = np.load(path)
    env = UrbanCarEnv(); agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)
    if Q.shape != (env.NUM_STATES, env.NUM_ACTIONS):
        sys.exit(f"Q-table {Q.shape} incompatible avec l'environnement {(env.NUM_STATES, env.NUM_ACTIONS)} : relancez train.py")
    agent.Q = Q; agent.epsilon = 0.0

    worst = 1.0
    print("=" * 100)
    for grp, name, ov in SCENARIOS:
        ok = defaultdict(int); tot = defaultdict(int); full = 0
        sample = None
        for k in range(n):
            spd, lane, act = play(env, agent, ov)
            r = check(env, spd, lane, act)
            for key, v in r.items():
                tot[key] += 1; ok[key] += int(v)
            conform = all(r.values())
            full += conform
            if sample is None or (not conform and sample[0]):
                sample = (conform, timeline(act), dict(light=env.LIGHT_NAMES[env.light_color],
                          car=env.car_ahead, car_lane=env.car_lane, ped=env.pedestrian))
        pct = full / n; worst = min(worst, pct)
        flag = "OK " if pct >= 0.99 else ("~~ " if pct >= 0.9 else "XX ")
        print(f"{flag}{pct*100:5.1f}%  {name}")
        for key in tot:
            print(f"          {ok[key]/tot[key]*100:5.1f}%  {key}")
        c, tl, cond = sample
        print(f"          ex. {'conforme' if c else 'NON conforme'} {cond}\n          {tl[:200]}")
    print("=" * 100)
    print(f"Pire scenario : {worst*100:.1f}% d'episodes conformes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "q_table.npy")