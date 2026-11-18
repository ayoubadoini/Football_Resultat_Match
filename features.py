# Construction des variables d'un match à partir de l'état des équipes (même définitions que dans le notebook)
import math
from datetime import datetime

FEATURES_NUM = ['elo_dom', 'elo_ext', 'elo_diff', 'pts_dom', 'pts_ext', 'pts_diff', 'bp_dom', 'bc_dom', 'bp_ext', 'bc_ext',
                'val_diff', 'val_moy', 'repos_dom', 'repos_ext']
FEATURES = FEATURES_NUM + ['val_connue', 'competition_type']      # ordre = ordre d'entraînement
CLASSES = ['Victoire extérieur', 'Nul', 'Victoire domicile']       # classes 0, 1, 2
MIN_MATCHS = 5                                                      # historique minimal (comme à l'entraînement)


def variables_valeur(x, z):
    # (écart, niveau moyen, indicateur) en échelle log10 ; valeur inconnue -> valeurs neutres
    if x is None or z is None or x <= 0 or z <= 0:
        return 0.0, 7.0, 0
    lx, lz = math.log10(x), math.log10(z)
    return lx - lz, (lx + lz) / 2, 1


def repos(equipe, jour):
    if not equipe.get('dernier'):
        return 30.0
    return float(max(0, min(30, (jour - datetime.fromisoformat(equipe['dernier'])).days)))


def ligne_depuis_etat(etat, dom, ext, competition, jour=None):
    e1, e2 = etat['equipes'][str(dom)], etat['equipes'][str(ext)]
    j = datetime.fromisoformat(jour or etat['date_reference'])
    vd, vm, vc = variables_valeur(e1.get('valeur'), e2.get('valeur'))
    return {'elo_dom': e1['elo'], 'elo_ext': e2['elo'], 'elo_diff': e1['elo'] - e2['elo'],
            'pts_dom': e1['pts'], 'pts_ext': e2['pts'], 'pts_diff': e1['pts'] - e2['pts'],
            'bp_dom': e1['bp'], 'bc_dom': e1['bc'], 'bp_ext': e2['bp'], 'bc_ext': e2['bc'],
            'val_diff': vd, 'val_moy': vm, 'repos_dom': repos(e1, j), 'repos_ext': repos(e2, j),
            'val_connue': vc, 'competition_type': competition}
