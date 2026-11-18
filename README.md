# Agent de conduite urbaine par Q-Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un agent de Reinforcement Learning tabulaire qui apprend a conduire
dans un environnement urbain simule : feux tricolores, obstacles, pietons.
Environnement type Gym code from scratch, Q-Learning implemente de A a Z
(pas de stable-baselines3), visualisation Pygame temps reel.

![Demo](docs/demo.gif)

## Resultats

| Agent | Recompense moy. | Taux de collision |
|-------|-----------------|-------------------|
| Aleatoire | -80 | 55 % |
| Regles fixes | +60 | 12 % |
| **Q-Learning** | **+105** | **2.8 %** |

![Courbes d'entrainement](docs/training_curves.png)

Entrainement : 80 000 episodes, Q-table de 2 880 etats x 5 actions.

## Comment ca marche

### Etat (2 880 etats discrets)

| Composante | Valeurs | Description |
|------------|---------|-------------|
| Vitesse | 4 | arret / lent / moyen / rapide |
| Feu | 3 | rouge / vert / jaune |
| Obstacle | 4 | aucun / meme voie / gauche / droite |
| Pieton | 2 | absent / present |
| Voie | 3 | gauche / centrale / droite |
| Phase | 10 | route libre + approche/critique/depasse |

### Actions (5)

0. Demarrer - acceleration douce (si v < 0.4)
1. Accelerer - acceleration forte
2. Freiner - deceleration forte
3. Maintenir - vitesse constante
4. Changer de voie - gauche, ou retour au centre

### Recompense

    R_t = R_progress + R_speed + R_lane + R_control + R_event + R_terminal

- R_progress : avancement x vitesse
- R_speed : proche de la limite (penalite si depassement)
- R_lane : penalite hors voie centrale
- R_control : penalite pour les actions couteuses
- R_event : recompenses/penalites pour les evenements
- R_terminal : +5 si survie complete (200 pas)

### Algorithme

Q-Learning tabulaire (Watkins & Dayan, 1992).

    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

- alpha : 0.20 -> 0.04 (decroissance lineaire)
- gamma : 0.95
- epsilon : 1.0 -> 0.01 (epsilon-greedy)
- Initialisation optimiste : Q0 = 1.0

## Scenarios testes

14 scenarios couvrant toutes les regles comportementales :

- Feux : rouge (arret), vert (avance), jaune (demarre mais ne roule pas)
- Obstacles : meme voie (evitement), gauche/droite (rester en voie)
- Pieton : freinage a distance precise
- Combines : rouge + obstacle + pieton
- Aleatoire : conditions reelles

## Lancer le projet

    pip install -r requirements.txt
    python train.py
    python test_pygame.py
    python test_scenarios.py
    python verify_scenarios.py

## Structure

| Fichier | Role |
|---------|------|
| car_env.py | Environnement (reset, step, state, reward, done) |
| q_agent.py | Agent Q-Learning (epsilon-greedy, alpha/gamma decroissants) |
| train.py | Boucle d'entrainement + courbes |
| test_pygame.py | Visualisation temps reel |
| test_scenarios.py | Evaluation sur scenarios |
| verify_scenarios.py | Verification des regles comportementales |

## Limites

- Q-Learning tabulaire : ne passe pas a l'echelle (espace d'etats discret)
  -> pour un etat continu, il faudrait DQN, PPO, ou SAC
- Environnement simplifie : pas de physique realiste, pas de trafic
  multi-agents, scenarios deterministes
- Q-table non versionnee : regenerez-la avec python train.py

## Stack

Python, NumPy, Pygame, Matplotlib.

## Licence

MIT - voir LICENSE
