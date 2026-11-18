import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, etat, pipeline
from features import FEATURES, MIN_MATCHS, ligne_depuis_etat

client = TestClient(app)
ACTIVES = [i for i, e in etat['equipes'].items() if e['nb'] >= MIN_MATCHS and not e['selection']]
DOM, EXT = int(ACTIVES[0]), int(ACTIVES[1])
LIGNE = ligne_depuis_etat(etat, DOM, EXT, 'domestic_league')


def test_health():
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_predict_identique_au_pipeline():
    r = client.post('/predict', json=LIGNE)
    assert r.status_code == 200
    attendu = pipeline.predict_proba(pd.DataFrame([LIGNE])[FEATURES])[0]
    assert list(r.json()['probabilites'].values()) == pytest.approx([round(float(v), 4) for v in attendu], abs=1e-4)
    assert sum(r.json()['probabilites'].values()) == pytest.approx(1, abs=1e-3)


def test_predict_match():
    r = client.post('/predict-match', json={'equipe_dom': DOM, 'equipe_ext': EXT, 'competition_type': 'domestic_league'})
    assert r.status_code == 200
    assert r.json()['prediction'] in ('Victoire extérieur', 'Nul', 'Victoire domicile')


@pytest.mark.parametrize('modif', [
    {'elo_dom': 99999},                       # hors bornes
    {'pts_dom': -1},                          # hors bornes
    {'elo_dom': 'beaucoup'},                  # mauvais type
    {'val_connue': 2},                        # valeur non autorisée
    {'competition_type': 'tournoi_inconnu'},  # catégorie inconnue
    {'champ_en_trop': 1},                     # champ inconnu refusé
])
def test_entrees_invalides_422(modif):
    assert client.post('/predict', json={**LIGNE, **modif}).status_code == 422


def test_champ_manquant_422():
    mauvais = dict(LIGNE); mauvais.pop('elo_dom')
    assert client.post('/predict', json=mauvais).status_code == 422


def test_corps_trop_gros_413():
    r = client.post('/predict', content=b'x' * 20000, headers={'content-type': 'application/json'})
    assert r.status_code == 413


def test_equipe_inconnue_404():
    r = client.post('/predict-match', json={'equipe_dom': -1, 'equipe_ext': EXT, 'competition_type': 'domestic_league'})
    assert r.status_code == 404


def test_memes_equipes_400():
    r = client.post('/predict-match', json={'equipe_dom': DOM, 'equipe_ext': DOM, 'competition_type': 'domestic_league'})
    assert r.status_code == 400


def test_recherche_equipes():
    r = client.get('/equipes', params={'q': etat['equipes'][str(DOM)]['nom'][:4], 'limite': 5})
    assert r.status_code == 200 and len(r.json()) <= 5
