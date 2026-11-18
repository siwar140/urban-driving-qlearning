# -*- coding: utf-8 -*-
"""
Environnement de conduite urbaine — Q-Learning  (v2)
=====================================================
Reward : R_t = R_progress + R_speed + R_lane + R_control + R_event + R_terminal

Actions :
  0 = Démarrer      (départ doux, n'agit que si v < START_MAX)
  1 = Accélérer     (forte accélération)
  2 = Freiner       (décélération forte)
  3 = Maintenir     (vitesse constante)
  4 = Changer voie  (évitement latéral : vers la gauche, ou retour au centre si déjà à gauche)

Chronologie d'un épisode (200 pas) — chaque événement a une ZONE de ±15 pas :
  Feu       T=50   zone 35..65   (rouge / orange / vert)
  Obstacle  T=100  zone 85..115  (voiture dans la voie de l'agent OU dans une autre voie)
  Piéton    T=150  zone 135..165 (occupe la chaussée de T-5 à T+5)

Ce que l'AGENT PERÇOIT (état) — c'est ce qui rend les épisodes « exacts » :
  • phase = où l'on en est dans l'événement : approche / critique / dépassé
      (sans cela l'agent ne sait pas s'il est à t=36 ou t=50 et ne peut pas
       « freiner à la distance précise »)
  • obstacle = aucun / dans MA voie / à GAUCHE / à DROITE
  • feu visible uniquement dans sa zone ; piéton visible uniquement dans sa zone

Règles comportementales (vérifiées par verify_scenarios.py) :
  • Feu ROUGE  → arrêté à t=T_LIGHT, puis le feu passe au vert et l'agent repart
  • Feu VERT   → en marche à T_LIGHT (Démarrer si à l'arrêt)
  • Feu ORANGE → Maintenir (pas de pénalité)
  • Obstacle dans MA voie   → Changer de voie avant T_OBSTACLE
  • Obstacle dans une autre voie → rester dans sa voie (ne pas s'en approcher)
  • Piéton → arrêté pendant toute sa traversée (T-5 .. T+5), freinage dès l'approche
"""
import math, random


class UrbanCarEnv:
    # ── Actions ───────────────────────────────────────────────────────────
    NUM_ACTIONS  = 5
    ACTION_NAMES = ["Start", "Accelerate", "Brake", "Maintain", "Change lane"]

    # Vecteur de contrôle (a_accel, a_steer, a_brake) — utilisé pour R_control
    CONTROL = {
        0: (0.5,  0.0,  0.0),   # Start
        1: (0.8,  0.0,  0.0),   # Accelerate
        2: (0.0,  0.0,  0.9),   # Brake
        3: (0.0,  0.0,  0.0),   # Maintain
        4: (0.0, -0.9,  0.0),   # Change lane
    }

    # ── Physique ──────────────────────────────────────────────────────────
    V_LIMIT   = 1.0        # vitesse limite normalisée
    V_MAX     = 1.5        # vitesse maximale (dépassement de limite pénalisé)
    START_MAX = 0.4        # "Start" n'agit que sous cette vitesse
    DV = {
        0: +0.12,          # Start
        1: +0.22,          # Accelerate
        2: -0.35,          # Brake
        3:  0.00,          # Maintain (vitesse constante)
        4: -0.05,          # Change lane (perte légère due au braquage)
    }

    # ── Coefficients de récompense ────────────────────────────────────────
    KP = 0.1   # R_progress
    KS = 0.2   # R_speed
    KL = 0.05  # R_lane
    KC = 0.01  # R_control
    OFF_CENTER = 0.05   # coût par pas hors de la voie centrale

    # ── Événements (pas de déclenchement) ─────────────────────────────────
    T_LIGHT    = 50
    T_OBSTACLE = 100
    T_PED      = 150
    MAX_STEPS  = 200
    ZONE_HALF  = 15    # rayon de la zone événement (en pas)
    CRIT       = 5     # fenêtre critique : les 5 pas qui précèdent l'événement
    PED_HALF   = 5     # le piéton occupe la chaussée de T_PED-5 à T_PED+5

    # ── Espace d'états discret ─────────────────────────────────────────────
    # speed(4) × light(3) × obstacle(4) × ped(2) × lane(3) × phase(10)
    N_SPEED, N_LIGHT, N_OBS, N_PED, N_LANE, N_PHASE = 4, 3, 4, 2, 3, 10
    NUM_STATES = N_SPEED * N_LIGHT * N_OBS * N_PED * N_LANE * N_PHASE   # 2880

    # phases : 0 route libre
    #   1/2/3 = feu      approche / critique / dépassé
    #   4/5/6 = obstacle approche / critique / dépassé
    #   7/8/9 = piéton   approche / critique / dépassé
    PHASE_NAMES = ["Clear road",
                   "Light: approach", "Light: critical", "Light: passed",
                   "Obstacle: approach", "Obstacle: critical", "Obstacle: passed",
                   "Pedestrian: approach", "Pedestrian: critical", "Pedestrian: passed"]
    LIGHT_NAMES = {0: "RED", 1: "GREEN", 2: "YELLOW"}

    # ─────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.reset()

    def reset(self, scenario=None):
        sc = scenario or {}
        self.step_count    = 0
        self.lane          = sc.get("lane", 1)      # départ voie centrale
        self.speed         = 0.0                    # toujours à l'arrêt au départ
        self.done          = False
        self.crashed       = False
        self.crash_reason  = ""
        self.success       = False
        self.outcomes      = {}
        self.total_reward  = 0.0
        self.last_r_detail = {}

        # Conditions de l'épisode
        self.light_color = sc.get("light_color", random.choice([0, 0, 1, 2]))
        self.car_ahead   = sc.get("car_ahead",   random.randint(0, 1))
        self.car_lane    = sc.get("car_lane",    random.randint(0, 2))  # 0 gauche, 1 centre, 2 droite
        self.pedestrian  = sc.get("pedestrian",  random.randint(0, 1))

        return self._get_state()

    # ── Temps / zones / phases ────────────────────────────────────────────
    def _zone(self, t=None):
        """0 libre, 1 zone feu, 2 zone obstacle, 3 zone piéton (pour l'affichage)."""
        t = t if t is not None else self.step_count
        if abs(t - self.T_LIGHT)    <= self.ZONE_HALF: return 1
        if abs(t - self.T_OBSTACLE) <= self.ZONE_HALF: return 2
        if abs(t - self.T_PED)      <= self.ZONE_HALF: return 3
        return 0

    def _phase(self, t=None):
        """Position dans l'événement : approche / critique / dépassé (0 = hors zone)."""
        t = t if t is not None else self.step_count
        for base, T, lo, hi in ((1, self.T_LIGHT,    -self.CRIT, 0),
                                (4, self.T_OBSTACLE, -self.CRIT, 0),
                                (7, self.T_PED,      -self.CRIT, self.PED_HALF)):
            d = t - T
            if abs(d) <= self.ZONE_HALF:
                if d < lo:  return base          # approche
                if d <= hi: return base + 1      # critique
                return base + 2                  # dépassé
        return 0

    def light_now(self, t=None):
        """Couleur du feu :
        - Par défaut hors zone : VERT (l'agent roule sur route libre).
        - Dans la zone feu :
            * light_color in (0, 2) (cycle complet) :
                t <= T_LIGHT            : ROUGE (0) -> l'agent s'arrête
                T_LIGHT < t <= T_LIGHT+3: JAUNE (2) -> démarre mais ne marche pas
                t > T_LIGHT+3           : VERT (1)  -> commence à marcher
            * light_color == 1          : VERT (1) permanent
        """
        t = t if t is not None else self.step_count
        if self.light_color in (0, 2):
            if t <= self.T_LIGHT:
                return 0
            elif t <= self.T_LIGHT + 9:
                return 2
            else:
                return 1
        else:
            return 1

    # ── Fonctions d'état ──────────────────────────────────────────────────
    def _speed_bin(self):
        v = self.speed
        if v <= 0.0:  return 0   # arrêt
        if v < 0.35:  return 1   # lent
        if v < 0.85:  return 2   # moyen
        return 3                  # rapide / à la limite

    def _obstacle_obs(self):
        """0 aucun · 1 dans MA voie · 2 à GAUCHE · 3 à DROITE (visible en approche/critique)."""
        if self.car_ahead != 1 or self._phase() not in (4, 5):
            return 0
        if self.car_lane == self.lane: return 1
        return 2 if self.car_lane < self.lane else 3

    def _ped_near(self):
        """1 si le piéton est visible (approche ou traversée)."""
        return 1 if (self.pedestrian == 1 and self._phase() in (7, 8)) else 0

    def _get_state(self):
        light_obs = self.light_now() if self._zone() == 1 else 1   # feu visible dans sa zone
        return (self._speed_bin(), light_obs,
                self._obstacle_obs(), self._ped_near(),
                self.lane, self._phase())

    def get_state_index(self, state):
        sb, lc, ob, pn, ln, ph = state
        idx = sb
        idx = idx * self.N_LIGHT + lc
        idx = idx * self.N_OBS   + ob
        idx = idx * self.N_PED   + pn
        idx = idx * self.N_LANE  + ln
        idx = idx * self.N_PHASE + ph
        assert 0 <= idx < self.NUM_STATES, f"Index hors limite: {idx}"
        return idx

    # ── Pas de simulation ─────────────────────────────────────────────────
    def step(self, action):
        t     = self.step_count
        phase = self._phase(t)

        old_lane  = self.lane
        old_speed = self.speed
        a_acc, a_str, a_brk = self.CONTROL[action]
        r_event = 0.0

        # ── Appliquer l'action ────────────────────────────────────────────
        if action == 4:   # Changer de voie : gauche, ou retour au centre si déjà à gauche
            self.lane = self.lane - 1 if self.lane > 0 else 1

        dv = self.DV[action]
        if action == 1:                                   # Accélérer
            self.speed = min(self.V_MAX, self.speed + dv)
        elif action == 0 and old_speed >= self.START_MAX: # Démarrer en roulant : sans effet
            self.speed = min(self.V_LIMIT, self.speed)
            r_event   -= 0.05
        else:
            self.speed = max(0.0, min(self.V_LIMIT, self.speed + dv))

        changed = (old_lane != self.lane)
        theta  = 0.15 if changed else 0.0
        d_lane = 0.6  if changed else 0.0

        # ── 1. R_progress ─────────────────────────────────────────────────
        r_progress = self.KP * self.speed * math.cos(theta)

        # ── 2. R_speed ────────────────────────────────────────────────────
        if self.speed <= self.V_LIMIT:
            r_speed = self.KS * (self.speed / self.V_LIMIT)
        else:
            r_speed = -self.KS * (self.speed - self.V_LIMIT) / self.V_LIMIT

        # ── 3. R_lane ─────────────────────────────────────────────────────
        r_lane = -self.KL * (d_lane ** 2)
        if self.lane != 1:
            # Rester dans sa voie (centrale) sauf si contournement d'un obstacle
            avoiding = (self.car_ahead == 1 and self.car_lane == 1 and phase in (4, 5, 6))
            if not avoiding:
                r_lane -= 0.15              # pénalité nette hors de la voie centrale

        # ── 4. R_control ──────────────────────────────────────────────────
        r_control = -self.KC * (a_acc**2 + a_str**2 + a_brk**2)

        # ── 5. R_event ────────────────────────────────────────────────────
        # ─ Changement de voie (action 4) : strictement réservé à l'évitement ─
        if action == 4:
            legitimate_avoid = (self.car_ahead == 1 and self.car_lane == old_lane and phase in (4, 5) and old_lane == 1)
            legitimate_return = (old_lane != 1 and phase == 6)
            if legitimate_avoid:
                r_event += 4.0              # évitement réussi
            elif legitimate_return:
                r_event += 2.0              # retour propre au centre
            else:
                r_event -= 4.0              # changement de voie inutile / bizarre !

        # ─ Freinage injustifié (action 2) ─────────────────────────────────
        if action == 2:
            need_brake = (
                (self._zone(t) == 1 and self.light_now(t) == 0 and phase in (1, 2) and self.speed > 0.0) or
                (self.pedestrian == 1 and phase in (7, 8) and self.speed > 0.0) or
                (self.speed > self.V_LIMIT)
            )
            stopped_at_danger = (
                self.speed == 0.0 and (
                    (self._zone(t) == 1 and self.light_now(t) == 0 and phase in (1, 2)) or
                    (self.pedestrian == 1 and phase in (7, 8))
                )
            )
            if not need_brake and not stopped_at_danger:
                r_event -= 1.5              # freinage inutile / injustifié (au feu vert, route libre)

        # ─ Feu de circulation (zone 1) ──────────────────────────────────
        if self._zone(t) == 1:
            cur_light = self.light_now(t)

            # 1. ROUGE : doit ralentir puis s'arrêter
            if cur_light == 0:
                if phase == 1:              # approche
                    if action == 2:
                        r_event += 1.0      # bon freinage anticipé
                    elif action in (0, 1):
                        r_event -= 1.5      # n'accélère pas au rouge
                elif phase == 2:            # critique
                    if action in (0, 1):
                        r_event -= 3.0      # interdit d'avancer vers feu rouge
                    elif self.speed <= 0.01:
                        r_event += 1.0      # arrêt maîtrisé
                if t == self.T_LIGHT:
                    if self.speed > 0.01:
                        r_event -= 5.0      # infraction grave feu rouge
                        self.outcomes["feu"] = False
                    else:
                        r_event += 2.0      # arrêt parfait
                        self.outcomes["feu"] = True

            # 2. JAUNE : démarre mais ne marche pas
            elif cur_light == 2:
                if t == self.T_LIGHT and self.light_color == 2:
                    self.outcomes["feu"] = True
                if action == 0:
                    r_event += 2.0          # démarre au feu jaune (Action 0)
                    self.outcomes["feu_jaune"] = True
                elif action == 1:
                    r_event -= 2.5          # interdit d'accélérer ("ne marche pas")
                elif action == 2 and self.speed <= 0.01:
                    r_event -= 1.0          # reste immobile avec frein
                if self.speed > self.START_MAX:
                    r_event -= 2.0          # vitesse excessive au feu jaune

            # 3. VERT : commence à marcher / roule normalement
            elif cur_light == 1:
                if phase in (1, 2):         # approche feu vert : roule droit sans freiner
                    if action == 3 and self.speed >= 0.7:
                        r_event += 0.5      # maintien de vitesse au vert
                    elif action == 1 and self.speed < self.V_LIMIT:
                        r_event += 0.5      # accélération au vert
                elif phase == 3:            # redémarrage après le feu rouge/jaune
                    if action == 1:
                        r_event += 2.0      # redémarrage dynamique au vert
                    elif action in (2, 3) and self.speed <= 0.01:
                        r_event -= 2.0      # reste bloqué au vert
                if t == self.T_LIGHT and self.light_color == 1:
                    if self.speed < 0.05:
                        r_event -= 1.0      # bloque la circulation
                        self.outcomes["feu"] = False
                    else:
                        self.outcomes["feu"] = True

        # ─ Obstacle (zone 2) ──────────────────────────────────────────────
        if self._zone(t) == 2 and self.car_ahead == 1:
            if phase in (4, 5):
                obs = self._obstacle_obs()
                if obs == 1:
                    if action == 4:
                        r_event += 2.0
                    else:
                        r_event -= 2.0
                elif obs in (2, 3):
                    if action in (1, 3):
                        r_event += 1.0       # maintien de trajectoire
                    elif action == 4:
                        r_event -= 4.0       # braquage interdit

        if t == self.T_OBSTACLE:
            if self.car_ahead == 1 and self.lane == self.car_lane:
                r_event -= 20               # collision véhicule
                self.crashed      = True
                self.crash_reason = "COLLISION VOITURE"
                self.outcomes["obstacle"] = False
                self.done = True
            else:
                self.outcomes["obstacle"] = True

        # ─ Piéton : il occupe la chaussée de T_PED-5 à T_PED+5 ────────────
        if self._zone(t) == 3 and self.pedestrian == 1:
            if phase in (7, 8):
                if self.speed > 0.01 and action == 2:
                    r_event += 1.5          # bon freinage pour le piéton
                elif self.speed <= 0.01 and action in (2, 3):
                    r_event += 1.5          # arrêt respectueux du piéton
                elif action in (0, 1):
                    r_event -= 3.0          # interdit d'avancer pendant traversée
            elif phase == 9:
                if self.speed < 0.8:
                    if action == 1:
                        r_event += 3.0      # accélération franche après le piéton
                    elif action == 0 and self.speed <= 0.01:
                        r_event += 1.5      # premier pas de démarrage
                    elif action in (2, 3):
                        r_event -= 2.0      # ne traîne pas après le piéton

        if self.pedestrian == 1:
            if abs(t - self.T_PED) <= self.PED_HALF and not self.done:
                if self.speed > 0.01:
                    r_event -= 50           # piéton percuté
                    self.crashed      = True
                    self.crash_reason = "PIETON PERCUTE"
                    self.outcomes["pieton"] = False
                    self.done = True
                elif t == self.T_PED + self.PED_HALF:
                    self.outcomes["pieton"] = True
        elif t == self.T_PED:
            self.outcomes["pieton"] = True

        # ── 6. R_terminal ─────────────────────────────────────────────────
        r_terminal = 0.0
        self.step_count += 1
        if not self.done and self.step_count >= self.MAX_STEPS:
            self.done     = True
            self.success  = True
            r_terminal    = 5.0

        reward = r_progress + r_speed + r_lane + r_control + r_event + r_terminal
        self.total_reward += reward

        # Mémorise détail pour le HUD
        self.last_r_detail = dict(
            progress=round(r_progress, 4),
            speed   =round(r_speed,    4),
            lane    =round(r_lane,     4),
            control =round(r_control,  4),
            event   =round(r_event,    2),
            terminal=round(r_terminal, 1),
        )

        return self._get_state(), reward, self.done