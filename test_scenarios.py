# -*- coding: utf-8 -*-
"""
Scénarios de test couvrant toutes les règles comportementales.

Règles testées :
  Feu ROUGE   → Freiner jusqu'au stop
  Feu VERT    → Maintenir si en marche, Démarrer si arrêté
  Feu ORANGE  → Maintenir (pas de pénalité)
  Obstacle dans notre voie → Changer de voie
  Obstacle hors voie       → Ne pas s'en approcher (rester en voie)
  Piéton       → Freiner à distance précise
  Sans piéton  → Avancer normalement
  Combinés     → Toutes situations

Deux modes :
  * run_all()      : chaque scénario est testé N fois, indépendamment (stats)
  * run_timeline() : TIMELINE = suite d'épisodes dans un ORDRE CHRONOLOGIQUE
                     (ex. feu rouge, puis feu vert, puis orange, ...).
                     Les épisodes restent indépendants (reset à chaque fois),
                     mais ils sont joués les uns après les autres, comme un trajet.

Usage :
  python test_scenarios.py [q_table.npy] [--mode all|timeline|both]
                           [--passes N] [--seed S]
"""
import sys, random, argparse
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from collections import Counter, defaultdict
from car_env import UrbanCarEnv
from q_agent  import QLearningAgent


# ─── Définition des scénarios ─────────────────────────────────────────────────
# (groupe, nom, overrides_dict)
SCENARIOS = [
    # ── Feux ────────────────────────────────────────────────────────────────
    ("Feu", "Feu ROUGE  → arret, demarre au jaune, repart",
     dict(light_color=0, car_ahead=0, pedestrian=0)),

    ("Feu", "Feu VERT   → doit avancer sans s'arreter",
     dict(light_color=1, car_ahead=0, pedestrian=0)),

    ("Feu", "Feu JAUNE  → demarre mais ne roule pas vite",
     dict(light_color=2, car_ahead=0, pedestrian=0)),

    # ── Obstacle ────────────────────────────────────────────────────────────
    ("Obstacle", "Voiture MEME voie → changer de voie",
     dict(light_color=1, car_ahead=1, car_lane=1, pedestrian=0)),
    ("Obstacle", "Voiture voie GAUCHE → rester en voie",
     dict(light_color=1, car_ahead=1, car_lane=0, pedestrian=0)),
    ("Obstacle", "Voiture voie DROITE → rester en voie",
     dict(light_color=1, car_ahead=1, car_lane=2, pedestrian=0)),
    ("Obstacle", "Clear Road → avancer normalement",
     dict(light_color=1, car_ahead=0, pedestrian=0)),

    # ── Piéton ──────────────────────────────────────────────────────────────
    ("Pieton", "Pieton sur la route → freiner a distance",
     dict(light_color=1, car_ahead=0, pedestrian=1)),

    ("Pieton", "Pas de pieton → avancer",
     dict(light_color=1, car_ahead=0, pedestrian=0)),

    # ── Combinés ────────────────────────────────────────────────────────────
    ("Combine", "Rouge + Obstacle + Pieton (pire cas)",
     dict(light_color=0, car_ahead=1, car_lane=1, pedestrian=1)),

    ("Combine", "Vert + Obstacle + Pieton",
     dict(light_color=1, car_ahead=1, car_lane=1, pedestrian=1)),

    ("Combine", "Orange + Route libre + Pieton",
     dict(light_color=2, car_ahead=0, pedestrian=1)),

    ("Combine", "Vert + Obstacle + pas de pieton",
     dict(light_color=1, car_ahead=1, car_lane=1, pedestrian=0)),

    # ── Aléatoire ────────────────────────────────────────────────────────────
    ("Aleatoire", "Conditions reelles (tout aleatoire)", None),
]


# ─── Trajet chronologique ────────────────────────────────────────────────────
# Liste ORDONNÉE d'épisodes : (nom court, overrides).  L'ordre = l'ordre de jeu.
# Modifiez / réordonnez librement cette liste : le test ET la visualisation
# pygame la suivent automatiquement.
TIMELINE = [
    ("RED LIGHT",             dict(light_color=0, car_ahead=0, pedestrian=0)),
    ("GREEN LIGHT",              dict(light_color=1, car_ahead=0, pedestrian=0)),
    ("YELLOW LIGHT",             dict(light_color=2, car_ahead=0, pedestrian=0)),
    ("OPEN ROAD",           dict(light_color=1, car_ahead=0, pedestrian=0)),
    ("Obst. LEFT",  dict(light_color=1, car_ahead=1, car_lane=0, pedestrian=0)),
    ("Obst. SAME LANE",    dict(light_color=1, car_ahead=1, car_lane=1, pedestrian=0)),
    ("Obst. RIGHT",  dict(light_color=1, car_ahead=1, car_lane=2, pedestrian=0)),
    ("Pedestrian",                dict(light_color=1, car_ahead=0, pedestrian=1)),
    ("Green+Obst+Ped",  dict(light_color=1, car_ahead=1, car_lane=1, pedestrian=1)),
    ("Yellow+Free+Ped",   dict(light_color=2, car_ahead=0, pedestrian=1)),
    ("Red+Obst+Ped", dict(light_color=0, car_ahead=1, car_lane=1, pedestrian=1)),
    ("Random 1",           None),
    ("Random 2",           None),
    ("Random 3",           None),
]


# ─── Exécution d'un épisode ───────────────────────────────────────────────────
def _play(env, agent, overrides):
    state = env.reset(overrides)
    idx   = env.get_state_index(state)
    done  = False
    while not done:
        a = agent.choose_action(idx, is_training=False)
        state, _, done = env.step(a)
        idx = env.get_state_index(state)
    return env.total_reward


def _run(env, agent, overrides, n):
    rewards, crashes = [], 0
    ok  = defaultdict(int)
    fails = Counter()

    for _ in range(n):
        rewards.append(_play(env, agent, overrides))
        if env.crashed:
            crashes += 1
            fails[env.crash_reason] += 1
        for k in ("feu", "obstacle", "pieton"):
            if env.outcomes.get(k) is True:
                ok[k] += 1

    return dict(
        survie  = 1 - crashes / n,
        reward  = float(np.mean(rewards)),
        ok      = {k: v / n for k, v in ok.items()},
        fails   = fails,
    )


def _icon(r):
    return "OK " if r >= 0.95 else ("~~" if r >= 0.80 else "XX")


def run_all(Q, n=300):
    env   = UrbanCarEnv()
    agent = QLearningAgent(num_states=env.NUM_STATES,
                           num_actions=env.NUM_ACTIONS)
    agent.Q       = Q
    agent.epsilon = 0.0

    W = 110
    print("\n" + "=" * W)
    print(f"{'SCENARIO':<46}{'Survie':>8}{'Recomp.':>10}"
          f"{'Feu':>8}{'Obstacle':>10}{'Pieton':>8}   Echecs")
    print("=" * W)

    group_survie  = defaultdict(list)
    current_group = None
    random_res    = None

    for group, name, overrides in SCENARIOS:
        if group != current_group:
            print(f"\n── {group} ──")
            current_group = group
        n_runs = max(n, 1000) if overrides is None else n
        res    = _run(env, agent, overrides, n_runs)
        if overrides is None:
            random_res = res
        group_survie[group].append(res["survie"])

        top     = ", ".join(f"{k}x{v}" for k, v in res["fails"].most_common(2)) or "--"
        feu_s   = f"{res['ok'].get('feu',0)*100:5.1f}%"
        obs_s   = f"{res['ok'].get('obstacle',0)*100:6.1f}%"
        ped_s   = f"{res['ok'].get('pieton',0)*100:5.1f}%"
        sur_s   = f"{res['survie']*100:6.1f}%"
        rew_s   = f"{res['reward']:8.1f}"

        print(f"  {name:<44}{_icon(res['survie'])} {sur_s}{rew_s}"
              f"  {feu_s}  {obs_s}  {ped_s}   {top}")

    print("\n" + "-" * W)
    print("RESUME PAR CATEGORIE")
    for g, vals in group_survie.items():
        m = np.mean(vals)
        print(f"  {g:<14}{_icon(m)} {m*100:5.1f}%")

    if random_res:
        print("\nSCENARIO ALEATOIRE (conditions reelles) :")
        for k, v in random_res["ok"].items():
            print(f"  {k:<10}{_icon(v)} {v*100:5.1f}%")
        print(f"\n  Survie   : {random_res['survie']*100:.1f}%")
        print(f"  Recomp.  : {random_res['reward']:.1f}")
    print("=" * W)


def _flag(v):
    return "OK" if v is True else ("XX" if v is False else "--")


def run_timeline(Q, passes=1, seed=None):
    """Joue TIMELINE dans l'ordre, `passes` fois de suite.

    Pass 1  : détail épisode par épisode (ordre chronologique).
    Passes>1: taux de réussite agrégé par position dans le trajet.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env   = UrbanCarEnv()
    agent = QLearningAgent(num_states=env.NUM_STATES,
                           num_actions=env.NUM_ACTIONS)
    agent.Q       = Q
    agent.epsilon = 0.0

    n     = len(TIMELINE)
    stats = [dict(surv=0, rewards=[], fails=Counter()) for _ in range(n)]
    trajets_parfaits = 0
    W = 104

    print("\n" + "=" * W)
    print("TRAJET CHRONOLOGIQUE - episodes joues dans l'ordre"
          f"  ({n} episodes x {passes} passage(s))")
    print("=" * W)
    print(f"{'#':>3}  {'EPISODE':<26}{'Resultat':>10}{'Recomp.':>10}"
          f"{'Feu':>6}{'Obst.':>7}{'Ped.':>6}   Cause")
    print("-" * W)

    for p in range(passes):
        parfait = True
        for i, (name, ov) in enumerate(TIMELINE):
            reward  = _play(env, agent, ov)
            crashed = env.crashed
            st = stats[i]
            st["rewards"].append(reward)
            if crashed:
                parfait = False
                st["fails"][env.crash_reason] += 1
            else:
                st["surv"] += 1

            if p == 0:
                o = env.outcomes
                print(f"{i+1:>3}  {name:<26}"
                      f"{('CRASH' if crashed else 'OK'):>10}{reward:10.1f}"
                      f"{_flag(o.get('feu')):>6}{_flag(o.get('obstacle')):>7}"
                      f"{_flag(o.get('pieton')):>6}"
                      f"   {env.crash_reason if crashed else '--'}")
        trajets_parfaits += parfait

    if passes > 1:
        print("\n" + "-" * W)
        print(f"STATISTIQUES SUR {passes} PASSAGES (par position dans le trajet)")
        print(f"{'#':>3}  {'EPISODE':<26}{'Survie':>10}{'Recomp.':>10}   Echecs frequents")
        for i, (name, _) in enumerate(TIMELINE):
            st  = stats[i]
            sv  = st["surv"] / passes
            top = ", ".join(f"{k}x{v}" for k, v in st["fails"].most_common(2)) or "--"
            print(f"{i+1:>3}  {name:<26}{_icon(sv)} {sv*100:5.1f}%"
                  f"{np.mean(st['rewards']):10.1f}   {top}")

    print("\n" + "-" * W)
    print(f"Trajets sans aucun crash : {trajets_parfaits}/{passes} "
          f"({trajets_parfaits / passes * 100:.1f}%)")
    print("=" * W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("q_table", nargs="?", default="q_table.npy")
    ap.add_argument("--mode", choices=("all", "timeline", "both"), default="both")
    ap.add_argument("--passes", type=int, default=1,
                    help="nombre de passages du trajet chronologique")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    try:
        Q = np.load(args.q_table)
    except FileNotFoundError:
        sys.exit(f"Erreur : '{args.q_table}' introuvable. Lancez 'train.py' d'abord.")
    if Q.shape != (UrbanCarEnv.NUM_STATES, UrbanCarEnv.NUM_ACTIONS):
        sys.exit(f"Forme incorrecte {Q.shape}, attendu "
                 f"{(UrbanCarEnv.NUM_STATES, UrbanCarEnv.NUM_ACTIONS)}.")

    if args.mode in ("timeline", "both"):
        run_timeline(Q, passes=max(1, args.passes), seed=args.seed)
    if args.mode in ("all", "both"):
        run_all(Q)