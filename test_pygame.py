# -*- coding: utf-8 -*-
"""
Visualisation pygame — Agent de conduite urbaine Q-Learning
===========================================================
La voiture AVANCE sur la route (défilement de la route en fonction de la vitesse).
Les événements (feu, obstacle, piéton) arrivent de l'avant.

Les épisodes suivent TIMELINE (test_scenarios.py) dans l'ordre chronologique et
sont enchaînés SANS pause : même si chaque épisode est indépendant côté env, la
route continue de défiler, la vitesse et la voie affichées sont lissées, et le
résultat de l'épisode s'affiche dans un bandeau non bloquant.

Commandes :
  ← / →   Épisode précédent / suivant du trajet
  ESPACE   Pause / Reprise
  R        Rejouer l'épisode courant
  L        Verrouiller l'épisode (rejoue le même en boucle)
  ↑ / ↓    Vitesse de simulation (fps)
  Q/Echap  Quitter
"""
import sys, math
import pygame
import numpy as np
from car_env import UrbanCarEnv
from q_agent  import QLearningAgent
from test_scenarios import TIMELINE

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE MISE EN PAGE
# ═══════════════════════════════════════════════════════════════════════════════
W, H = 800, 660
FPS  = 14    # pas/seconde (suffisant pour voir chaque décision)

# Géométrie de la route (zone centrale de l'écran)
RX   = 185          # x gauche de la route
RW   = 390          # largeur totale (3 voies × 130 px)
LW   = 130          # largeur d'une voie
# Centres des voies (x)
LCX  = [RX + LW // 2,
         RX + LW + LW // 2,
         RX + 2 * LW + LW // 2]

AGENT_Y = H - 140   # y de la voiture agent (fixe, proche du bas)
CAR_W, CAR_H = 48, 78

# Vitesse de défilement de la route (pixels/pas à v=1.0)
SCROLL_SPEED = 10

# Nombre de pas avant l'événement à partir duquel il est affiché (y=0 → top écran)
LOOK_AHEAD = UrbanCarEnv.ZONE_HALF   # l'événement entre à l'écran au début de sa zone

# Palette
C_BG        = (14,  16,  22)
C_ROAD      = (38,  40,  48)
C_GRASS     = (25,  70,  30)
C_CURB      = (160, 130,  55)
C_DASH      = (210, 210, 190)
C_AGENT     = ( 30, 140, 255)
C_OBSTACLE  = (210,  45,  45)
C_PED       = (255, 200,  50)
C_TEXT      = (235, 240, 245)
C_DIM       = (110, 115, 130)
C_OK        = ( 50, 210,  75)
C_WARN      = (255, 165,   0)
C_ERR       = (240,  55,  55)
C_PANEL     = (  0,   0,   0, 175)
LIGHT_RGB   = {0: (255, 40, 40), 1: (45, 220, 65), 2: (255, 165, 0)}


# ═══════════════════════════════════════════════════════════════════════════════
#  DESSIN — UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def in_zone(t_event, step):
    """True si on est dans la zone de l'événement (t_event ± ZONE_HALF)."""
    return abs(step - t_event) <= UrbanCarEnv.ZONE_HALF


def _event_y(t_event, step):
    """Y à l'écran d'un événement (approche de haut vers bas vers AGENT_Y)."""
    remaining = t_event - step
    if remaining > LOOK_AHEAD:
        return -999   # pas encore visible
    px_per_step = AGENT_Y / LOOK_AHEAD
    if remaining < 0:            # une fois dépassé, l'événement s'éloigne vite (vue de l'arrière)
        px_per_step *= 3
    return int(AGENT_Y - remaining * px_per_step)


def draw_road(surf, scroll):
    """Route avec herbe latérale, trottoirs et lignes pointillées."""
    # Herbe latérale
    pygame.draw.rect(surf, C_GRASS,  (0,       0, RX,          H))
    pygame.draw.rect(surf, C_GRASS,  (RX + RW, 0, W - RX - RW, H))
    # Trottoirs
    pygame.draw.rect(surf, C_CURB,   (RX - 9,  0, 9, H))
    pygame.draw.rect(surf, C_CURB,   (RX + RW, 0, 9, H))
    # Chaussée
    pygame.draw.rect(surf, C_ROAD,   (RX, 0, RW, H))
    # Lignes de voie (pointillées, défilent avec scroll)
    for xi in (1, 2):
        x = RX + xi * LW
        for y0 in range(-80, H + 80, 70):
            yy = (y0 + int(scroll)) % (H + 80) - 80
            pygame.draw.rect(surf, C_DASH, (x - 2, yy, 4, 36))


def draw_car(surf, cx, cy, color, label="", highlight=False):
    """Voiture rectangulaire stylisée."""
    rect = pygame.Rect(cx - CAR_W // 2, cy - CAR_H // 2, CAR_W, CAR_H)
    if highlight:
        pygame.draw.rect(surf, (255, 255, 120), rect.inflate(8, 8), border_radius=10)
    pygame.draw.rect(surf, color, rect, border_radius=10)
    # Pare-brise
    ww = pygame.Rect(cx - CAR_W // 2 + 6, cy - CAR_H // 2 + 10,
                     CAR_W - 12, CAR_H // 4)
    pygame.draw.rect(surf, (175, 215, 255), ww, border_radius=5)
    # Roues (4 coins)
    for dx in (-(CAR_W // 2), CAR_W // 2 - 7):
        for dy in (-(CAR_H // 2) + 6, CAR_H // 2 - 18):
            pygame.draw.rect(surf, (18, 18, 18),
                             (cx + dx, cy + dy, 7, 16), border_radius=3)
    if label:
        lbl_s = pygame.font.SysFont("Arial", 11, bold=True).render(
            label, True, (255, 255, 255))
        surf.blit(lbl_s, (cx - lbl_s.get_width() // 2, cy - CAR_H // 2 - 16))


def draw_traffic_light(surf, cx, cy, light_color, big=False):
    """Feu tricolore sur un mât."""
    mast_h = 120 if big else 80
    pole_w = 8 if big else 5
    box_w  = 44 if big else 30
    r      = 14 if big else 9
    # Mât
    pygame.draw.rect(surf, (80, 80, 80),
                     (cx - pole_w // 2, cy - mast_h, pole_w, mast_h))
    # Boîtier
    pygame.draw.rect(surf, (50, 50, 50),
                     (cx - box_w // 2, cy - mast_h, box_w, box_w * 3),
                     border_radius=6)
    # 3 lampes (rouge=0 haut, orange=2 milieu, vert=1 bas)
    for i, col_key in enumerate((0, 2, 1)):
        on  = (col_key == light_color)
        col = LIGHT_RGB[col_key] if on else (30, 30, 30)
        yy  = cy - mast_h + box_w // 2 + i * box_w
        pygame.draw.circle(surf, col, (cx, yy), r)
        if on:  # halo brillant
            pygame.draw.circle(surf, col, (cx, yy), r + 4, 2)


def draw_pedestrian(surf, px, py, phase):
    """Piéton simplifié (tête + corps + jambes)."""
    r = 12
    # Corps
    pygame.draw.circle(surf, C_PED, (int(px), py - r), r)
    pygame.draw.line(surf, C_PED, (int(px), py), (int(px), py + 28), 5)
    # Jambes animées
    leg = 12 * math.sin(phase * 8)
    pygame.draw.line(surf, C_PED, (int(px), py + 28),
                     (int(px) - 8, int(py + 28 + leg)), 4)
    pygame.draw.line(surf, C_PED, (int(px), py + 28),
                     (int(px) + 8, int(py + 28 - leg)), 4)


def draw_speed_bar(surf, x, y, w, h, speed, v_limit, font):
    """Barre de vitesse (verte → orange → rouge)."""
    ratio = min(1.5, speed / v_limit) if v_limit > 0 else 0
    filled = int(w * min(ratio, 1.0))
    over   = max(0, int(w * (ratio - 1.0)))
    col    = C_OK if ratio <= 0.7 else (C_WARN if ratio <= 1.0 else C_ERR)
    pygame.draw.rect(surf, (40, 40, 50), (x, y, w, h), border_radius=4)
    if filled > 0:
        pygame.draw.rect(surf, col, (x, y, filled, h), border_radius=4)
    if over > 0:
        pygame.draw.rect(surf, C_ERR, (x + int(w), y, over, h), border_radius=4)
    # Marqueur limite
    pygame.draw.rect(surf, (200, 200, 200), (x + w - 2, y - 2, 2, h + 4))
    lbl = font.render(f"v={speed:.2f} / {v_limit:.1f}", True, C_TEXT)
    surf.blit(lbl, (x, y + h + 3))


def draw_panel_left(surf, fonts, env, action_idx):
    """Panneau gauche : état complet de l'agent."""
    f_med, f_sml, f_big = fonts
    panel = pygame.Surface((RX - 4, H), pygame.SRCALPHA)
    panel.fill((8, 10, 18, 210))
    surf.blit(panel, (0, 0))

    t   = env.step_count
    lc  = env.light_now()      # le feu passe au vert après la ligne d'arrêt
    zn  = env._zone()
    ZONE_NAMES = {0: "Free route ", 1: "Traffic light zone", 2: "Obstacle zone", 3: "Pedestrian zone"}
    ACT_NAMES  = env.ACTION_NAMES

    y = 10
    def line(txt, col=C_TEXT, fn=None):
        nonlocal y
        if fn is None: fn = f_sml
        s = fn.render(txt, True, col)
        surf.blit(s, (8, y))
        y += fn.size("A")[1] + 4

    line("=== AGENT ===", C_DIM, f_sml)
    line(f"Step : {t} / {env.MAX_STEPS}", C_TEXT, f_med)
    line("")
    # Barre vitesse
    draw_speed_bar(surf, 8, y, RX - 20, 14, env.speed, env.V_LIMIT, f_sml)
    y += 36
    line("")
    line("CURRENT ZONE", C_DIM)
    zn_col = [C_DIM, LIGHT_RGB[lc], C_ERR, C_WARN][zn]
    line(env.PHASE_NAMES[env._phase()], zn_col, f_med)
    line("")
    line("LIGHT", C_DIM)
    if zn == 1:
        line(env.LIGHT_NAMES[lc], LIGHT_RGB[lc], f_med)
    else:
        line("--", C_DIM, f_med)
    line("")
    line("CAR ?", C_DIM)
    if zn != 2:
        line("--", C_DIM, f_med)
    elif not env.car_ahead:
        line("No (free)", C_OK, f_med)
    elif env.car_lane == env.lane:
        line("Yes - SAME LANE", C_ERR, f_med)
    else:
        line("Yes - OTHER LANE", C_WARN, f_med)
    line("")
    line("Pedestrian ?", C_DIM)
    if zn != 3:
        line("--", C_DIM, f_med)
    else:
        line("Yes  (danger!)" if env.pedestrian else "No",
             C_WARN if env.pedestrian else C_OK, f_med)
    line("")
    line("LANE", C_DIM)
    lane_name = ["Left", "Center", "Right"][env.lane]
    line(lane_name, C_TEXT, f_med)
    line("")
    line("ACTION", C_DIM)
    line(ACT_NAMES[action_idx], C_OK, f_med)
    line("")
    line("TOTAL REWARD", C_DIM)
    rc = C_OK if env.total_reward >= 0 else C_ERR
    line(f"{env.total_reward:+.2f}", rc, f_med)

    # Détail reward
    if env.last_r_detail:
        line("")
        line("-- detailed reward --", C_DIM)
        d = env.last_r_detail
        for k, v in d.items():
            vc = C_OK if v >= 0 else C_ERR
            line(f"  {k:<8} {v:+.4f}", vc)


def _wrap(text, maxlen=16):
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > maxlen and cur:
            out.append(cur.strip())
            cur = word + " "
        else:
            cur += word + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def draw_panel_right(surf, fonts, pos, timeline, results, episode, loop_n,
                     locked, paused, outcomes, fps):
    """Panneau droit : trajet chronologique, épisode en cours, résultats."""
    f_med, f_sml, f_big = fonts
    px  = RX + RW + 14
    pw  = W - px - 4
    panel = pygame.Surface((pw, H), pygame.SRCALPHA)
    panel.fill((8, 10, 18, 210))
    surf.blit(panel, (px, 0))

    y = 10
    def line(txt, col=C_TEXT, fn=None):
        nonlocal y
        if fn is None: fn = f_sml
        s = fn.render(str(txt), True, col)
        surf.blit(s, (px + 6, y))
        y += fn.size("A")[1] + 4

    line("=== ROUTE ===", C_DIM)
    line(f"Episode {episode}", C_TEXT, f_med)
    line(f"Lap {loop_n}  ({pos+1}/{len(timeline)})", C_DIM)
    if locked:
        line("[LOCKED]", C_WARN)
    line("")
    line("IN PROGRESS", C_DIM)
    for part in _wrap(timeline[pos][0]):
        line(part, C_TEXT, f_med)
    line("")
    ICON = {True: "OK", False: "XX", None: ".."}
    for k in ("light", "obstacle", "pedestrian"):
        v = outcomes.get(k)
        col = C_OK if v is True else (C_ERR if v is False else C_DIM)
        line(f"{k:<8} {ICON[v]}", col)

    # ── Chronologie (fenêtre glissante autour de l'épisode courant) ──────
    line("")
    line("-- chronologie --", C_DIM)
    n, visible = len(timeline), 10
    start = max(0, min(pos - visible // 2, n - visible))
    for i in range(start, min(n, start + visible)):
        r = results[i]
        if i == pos:
            col = C_TEXT
        else:
            col = C_OK if r is True else (C_ERR if r is False else C_DIM)
        name = timeline[i][0]
        name = name if len(name) <= 15 else name[:14] + "."
        line(f"{'>' if i == pos else ' '}{i+1:>2} {ICON[r]} {name}", col)

    # ── Contrôles ────────────────────────────────────────────────────────
    line("")
    line(f"[</>] Episode  [R] Replay", C_DIM)
    line(f"[SPC] Pause    [L] Lock", C_DIM)
    line(f"[^/v] Speed ({fps} fps)", C_DIM)
    line(f"[Q] Quit", C_DIM)

    if paused:
        ps = f_big.render("PAUSE", True, (255, 255, 80))
        surf.blit(ps, (px + pw // 2 - ps.get_width() // 2, H // 2 - 20))


def draw_event_countdown(surf, font, step, T_LIGHT, T_OBS, T_PED):
    """Barre de progression des événements en bas de l'écran."""
    total = 200
    bar_x, bar_y, bar_w, bar_h = RX, H - 22, RW, 12
    # Fond
    pygame.draw.rect(surf, (30, 30, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
    # Progression
    prog = int(bar_w * step / total)
    pygame.draw.rect(surf, (60, 100, 200), (bar_x, bar_y, prog, bar_h), border_radius=4)
    # Marqueurs
    for t_ev, name, col in [(T_LIGHT, "Feu", LIGHT_RGB[1]),
                             (T_OBS,   "Obs.", C_ERR),
                             (T_PED,   "Ped.", C_PED)]:
        mx = bar_x + int(bar_w * t_ev / total)
        pygame.draw.rect(surf, col, (mx - 2, bar_y - 2, 4, bar_h + 4))
        s = font.render(name, True, col)
        surf.blit(s, (mx - s.get_width() // 2, bar_y - 16))


def draw_toast(surf, font, text, color, life, life_max):
    """Bandeau NON bloquant (fondu) annonçant le résultat de l'épisode précédent.
    La simulation continue pendant qu'il s'affiche."""
    alpha = int(255 * min(1.0, life / (life_max * 0.4)))
    s = font.render(text, True, color)
    s.set_alpha(alpha)
    box = pygame.Surface((s.get_width() + 24, s.get_height() + 14), pygame.SRCALPHA)
    pygame.draw.rect(box, (0, 0, 0, min(200, alpha)), box.get_rect(), border_radius=10)
    box.blit(s, (12, 7))
    surf.blit(box, (RX + RW // 2 - box.get_width() // 2, 18))


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE  —  conduite continue, épisodes enchaînés sans coupure
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo():
    pygame.init()
    surf  = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Agent Q-Learning — Conduite Urbaine (trajet continu)")
    clock = pygame.time.Clock()

    f_big = pygame.font.SysFont("Arial", 24, bold=True)
    f_med = pygame.font.SysFont("Arial", 15, bold=True)
    f_sml = pygame.font.SysFont("Arial", 12)
    fonts = (f_med, f_sml, f_big)

    # ── Chargement Q-table ────────────────────────────────────────────────
    env   = UrbanCarEnv()
    agent = QLearningAgent(num_states=env.NUM_STATES,
                           num_actions=env.NUM_ACTIONS)
    try:
        agent.load("q_table.npy")
        if agent.Q.shape != (env.NUM_STATES, env.NUM_ACTIONS):
            raise ValueError(f"Forme {agent.Q.shape} inattendue.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur : {e}\nLancez 'python train.py' d'abord.")
        pygame.quit(); sys.exit()
    agent.epsilon = 0.0
    print("Q-table chargee, demarrage de la demonstration.")

    # ── Trajet chronologique ──────────────────────────────────────────────
    timeline = TIMELINE
    n_scen   = len(timeline)
    pos      = 0                    # position courante dans le trajet
    loop_n   = 1                    # nombre de passages complets
    episode  = 1                    # compteur global d'épisodes
    results  = [None] * n_scen      # True/False/None par position (tour courant)

    # ── État d'affichage (PERSISTANT d'un épisode à l'autre) ──────────────
    # Les épisodes sont indépendants côté env (reset), mais l'écran, lui,
    # ne se réinitialise jamais : route qui défile, vitesse et position
    # latérale lissées -> aucune coupure visuelle entre deux épisodes.
    scroll    = 0.0
    vis_speed = 0.0
    vis_x     = float(LCX[1])
    fps       = FPS
    locked    = False
    paused    = False
    action    = 3
    outcomes  = {}
    toast     = None                # [texte, couleur, vie_restante, vie_max]

    def _new_ep():
        nonlocal action, outcomes
        state    = env.reset(timeline[pos][1])
        action   = 3
        outcomes = {}
        return env.get_state_index(state)

    state_idx = _new_ep()

    running = True
    while running:
        restart = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_RIGHT:
                    pos = (pos + 1) % n_scen;  restart = True
                elif ev.key == pygame.K_LEFT:
                    pos = (pos - 1) % n_scen;  restart = True
                elif ev.key == pygame.K_r:
                    restart = True
                elif ev.key == pygame.K_l:
                    locked = not locked
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_UP:
                    fps = min(60, fps + 2)
                elif ev.key == pygame.K_DOWN:
                    fps = max(4, fps - 2)

        if restart:
            episode  += 1
            state_idx = _new_ep()

        # ── Avancer la simulation ─────────────────────────────────────────
        if not paused:
            action              = agent.choose_action(state_idx, is_training=False)
            next_state, _, done = env.step(action)
            state_idx           = env.get_state_index(next_state)
            outcomes            = env.outcomes.copy()

            if done:
                # Résultat annoncé par un bandeau non bloquant, puis on
                # enchaîne IMMÉDIATEMENT sur l'épisode suivant du trajet.
                ok = not env.crashed
                results[pos] = ok
                msg = (f"Ep.{episode} : OK" if ok
                       else f"Ep.{episode} : CRASH {env.crash_reason}")
                life = int(fps * 2.2)
                toast = [msg, C_OK if ok else C_ERR, life, life]

                episode += 1
                if not locked:
                    pos += 1
                    if pos >= n_scen:
                        pos, loop_n = 0, loop_n + 1
                        results = [None] * n_scen
                state_idx = _new_ep()

            # Vitesse affichée lissée -> pas de saut au reset de l'env
            vis_speed += (env.speed - vis_speed) * 0.35
            scroll = (scroll + vis_speed * SCROLL_SPEED) % (H + 80)

        # Position latérale lissée (changement de voie fluide + pas de saut au reset)
        vis_x += (LCX[env.lane] - vis_x) * 0.30

        step = env.step_count

        # ── DESSIN ───────────────────────────────────────────────────────
        surf.fill(C_BG)
        draw_road(surf, scroll)

        # ─ Zone FEU ───────────────────────────────────────────────────────
        if in_zone(env.T_LIGHT, step):
            light_cx = RX + RW - 32
            draw_traffic_light(surf, light_cx, 160, env.light_now(), big=True)
            lt = f_med.render(env.LIGHT_NAMES[env.light_now()], True,
                              LIGHT_RGB[env.light_now()])
            surf.blit(lt, (light_cx - lt.get_width() // 2, 180))

        # ─ Zone OBSTACLE ──────────────────────────────────────────────────
        if env.car_ahead == 1 and in_zone(env.T_OBSTACLE, step):
            obs_screen_y = _event_y(env.T_OBSTACLE, step)
            if -CAR_H <= obs_screen_y <= H + CAR_H:
                danger_flash = (env.car_lane == env.lane and
                                0 <= env.T_OBSTACLE - step <= 8)
                draw_car(surf, LCX[env.car_lane], obs_screen_y, C_OBSTACLE,
                         label="!", highlight=danger_flash)

        # ─ Zone PIETON ────────────────────────────────────────────────────
        if in_zone(env.T_PED, step):
            ped_y_road = H - 260
            for xi in range(RX, RX + RW, 22):
                pygame.draw.rect(surf, (200, 200, 180), (xi, ped_y_road - 20, 10, 40))
            if env.pedestrian == 1:
                prog  = (step - (env.T_PED - env.ZONE_HALF)) / (2 * env.ZONE_HALF)
                ped_x = (RX - 10) + prog * (RW + 20)
                draw_pedestrian(surf, ped_x, ped_y_road, step * 0.15)

        # ─ Voiture agent ──────────────────────────────────────────────────
        agent_cx = int(vis_x)
        draw_car(surf, agent_cx, AGENT_Y, C_AGENT)

        act_name = env.ACTION_NAMES[action]
        act_col  = {0: C_OK, 1: C_OK, 2: C_WARN, 3: C_TEXT, 4: (140, 100, 255)}
        at = f_sml.render(act_name, True, act_col.get(action, C_TEXT))
        surf.blit(at, (agent_cx - at.get_width() // 2, AGENT_Y + CAR_H // 2 + 5))

        if vis_speed > 0.05:
            arrow_h = int(vis_speed * 30)
            arrow_y = AGENT_Y - CAR_H // 2 - 10
            pygame.draw.polygon(surf, C_OK, [
                (agent_cx, arrow_y - arrow_h),
                (agent_cx - 10, arrow_y),
                (agent_cx + 10, arrow_y),
            ])

        if env.speed <= 0.01 and action == 2:
            st = f_med.render("STOP", True, C_ERR)
            surf.blit(st, (agent_cx - st.get_width() // 2, AGENT_Y - CAR_H // 2 - 32))

        # ─ Barre de progression (bas) ─────────────────────────────────────
        draw_event_countdown(surf, f_sml, step,
                             env.T_LIGHT, env.T_OBSTACLE, env.T_PED)

        # ─ Panneaux latéraux ─────────────────────────────────────────────
        draw_panel_left(surf, fonts, env, action)
        draw_panel_right(surf, fonts, pos, timeline, results, episode, loop_n,
                         locked, paused, outcomes, fps)

        # ─ Bandeau de résultat (non bloquant) ─────────────────────────────
        if toast and toast[2] > 0:
            draw_toast(surf, f_med, toast[0], toast[1], toast[2], toast[3])
            if not paused:
                toast[2] -= 1

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    run_demo()