# Service de prédiction du résultat d'un match : python -m uvicorn app:app
import hashlib, json, logging, os, time
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from features import CLASSES, FEATURES, MIN_MATCHS, ligne_depuis_etat

BASE = Path(__file__).parent
MODELE = BASE / 'model'
LOG_DIR = Path(os.environ.get('LOG_DIR', BASE / 'logs'))
LOG_DIR.mkdir(exist_ok=True)
MAX_CORPS = 10_000                                                   # octets : au-delà -> 413

# ---------- chargement unique au démarrage + contrôle d'intégrité ----------
meta = json.loads((MODELE / 'metadata.json').read_text(encoding='utf-8'))
if hashlib.sha256((MODELE / 'pipeline.joblib').read_bytes()).hexdigest() != meta['sha256']:
    raise RuntimeError('pipeline.joblib modifié ou corrompu : empreinte SHA-256 différente')
pipeline = joblib.load(MODELE / 'pipeline.joblib')
etat = json.loads((MODELE / 'etat_equipes.json').read_text(encoding='utf-8'))
COMPETITIONS = meta['competitions']
Competition = Literal[tuple(COMPETITIONS)]
BORNES = meta['bornes']

# ---------- journal JSON (une ligne par évènement, sans donnée personnelle) ----------
journal = logging.getLogger('football-api')
journal.setLevel(logging.INFO)
if not journal.handlers:
    h = logging.FileHandler(LOG_DIR / 'service.jsonl', encoding='utf-8')
    h.setFormatter(logging.Formatter('%(message)s'))
    journal.addHandler(h)


def log(**kw):
    journal.info(json.dumps({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'version': meta['version'], **kw}, ensure_ascii=False))


# ---------- schémas (le contrat de données) ----------
def borne(nom):
    lo, hi = BORNES[nom]
    return Field(ge=lo, le=hi)


class MatchFeatures(BaseModel):
    model_config = ConfigDict(extra='forbid')                         # champ inconnu -> 422
    elo_dom: float = borne('elo_dom')
    elo_ext: float = borne('elo_ext')
    elo_diff: float = borne('elo_diff')
    pts_dom: float = borne('pts_dom')
    pts_ext: float = borne('pts_ext')
    pts_diff: float = borne('pts_diff')
    bp_dom: float = borne('bp_dom')
    bc_dom: float = borne('bc_dom')
    bp_ext: float = borne('bp_ext')
    bc_ext: float = borne('bc_ext')
    val_diff: float = borne('val_diff')
    val_moy: float = borne('val_moy')
    repos_dom: float = borne('repos_dom')
    repos_ext: float = borne('repos_ext')
    val_connue: Literal[0, 1]
    competition_type: Competition


class MatchEquipes(BaseModel):
    model_config = ConfigDict(extra='forbid')
    equipe_dom: int
    equipe_ext: int
    competition_type: Competition
    date: Optional[str] = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$')   # AAAA-MM-JJ (par défaut : fin des données)


app = FastAPI(title='Prédiction de résultat de match', version=meta['version'])


@app.middleware('http')
async def garde(request: Request, call_next):
    t0 = time.perf_counter()
    if int(request.headers.get('content-length') or 0) > MAX_CORPS:
        reponse = JSONResponse({'detail': 'Corps de requête trop volumineux'}, status_code=413)
    else:
        reponse = await call_next(request)
    log(evt='requete', methode=request.method, chemin=request.url.path, statut=reponse.status_code,
        ms=round((time.perf_counter() - t0) * 1000, 2))
    return reponse


@app.exception_handler(Exception)
async def erreur_interne(request: Request, exc: Exception):
    log(evt='erreur', chemin=request.url.path, type=type(exc).__name__)       # le détail technique reste dans les journaux
    return JSONResponse({'detail': 'Erreur interne'}, status_code=500)


def predire(ligne: dict) -> dict:
    proba = pipeline.predict_proba(pd.DataFrame([ligne])[FEATURES])[0]       # mêmes colonnes, même ordre qu'à l'entraînement
    p = {CLASSES[int(c)]: round(float(v), 4) for c, v in zip(pipeline.classes_, proba)}
    classe = max(p, key=p.get)
    log(evt='prediction', classe=classe, **{f'p_{i}': v for i, v in enumerate(p.values())})
    return {'version': meta['version'], 'prediction': classe, 'probabilites': p}


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': meta['version'], 'modele': meta['modele'], 'equipes': len(etat['equipes'])}


@app.get('/competitions')
def competitions():
    return COMPETITIONS


@app.get('/equipes')
def equipes(q: str = Query('', max_length=50), limite: int = Query(20, ge=1, le=50)):
    q = q.lower().strip()
    res = [{'id': int(i), 'nom': e['nom'], 'elo': round(e['elo'])} for i, e in etat['equipes'].items() if q in e['nom'].lower()]
    return sorted(res, key=lambda e: -e['elo'])[:limite]


@app.post('/predict')
def predict(m: MatchFeatures):
    return predire(m.model_dump())


@app.post('/predict-match')
def predict_match(m: MatchEquipes):
    if m.equipe_dom == m.equipe_ext:
        raise HTTPException(400, 'Les deux équipes doivent être différentes')
    for i in (m.equipe_dom, m.equipe_ext):
        e = etat['equipes'].get(str(i))
        if e is None:
            raise HTTPException(404, f'Équipe {i} inconnue ou inactive')
        if e['nb'] < MIN_MATCHS:
            raise HTTPException(400, f"Historique insuffisant pour {e['nom']} (minimum {MIN_MATCHS} matchs)")
    try:
        ligne = ligne_depuis_etat(etat, m.equipe_dom, m.equipe_ext, m.competition_type, m.date)
    except ValueError:
        raise HTTPException(400, 'Date invalide')
    e1, e2 = etat['equipes'][str(m.equipe_dom)], etat['equipes'][str(m.equipe_ext)]
    return {**predire(ligne), 'dom': {'nom': e1['nom'], 'elo': round(e1['elo']), 'valeur': e1.get('valeur'), 'bp': round(e1['bp'], 2), 'bc': round(e1['bc'], 2), 'pts': round(e1['pts'], 2)},
            'ext': {'nom': e2['nom'], 'elo': round(e2['elo']), 'valeur': e2.get('valeur'), 'bp': round(e2['bp'], 2), 'bc': round(e2['bc'], 2), 'pts': round(e2['pts'], 2)},
            'variables': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ligne.items()}}


# ---------- tableau de bord : statistiques d'entraînement (stats.json) et supervision en direct (journal) ----------
STATS_FICHIER = MODELE / 'stats.json'
STATS = json.loads(STATS_FICHIER.read_text(encoding='utf-8')) if STATS_FICHIER.exists() else {}


@app.get('/stats')
def stats():
    return {**STATS, 'meta': {'version': meta['version'], 'modele': meta['modele'], 'entraine_le': meta['entraine_le'],
                              'metriques': meta['metriques_test_temporel'], 'lignes': meta['lignes_entrainement'],
                              'classes': meta['classes'], 'prior': meta['prior_classes'], 'sklearn': meta['sklearn']}}


def lire_journal(max_octets=400_000):
    f = LOG_DIR / 'service.jsonl'
    if not f.exists():
        return []
    with open(f, 'rb') as fh:                                   # on ne lit que la fin du fichier (rapide même si le journal grossit)
        fh.seek(0, 2)
        taille = fh.tell()
        fh.seek(max(0, taille - max_octets))
        lignes = fh.read().decode('utf-8', errors='ignore').splitlines()
    sortie = []
    for l in lignes[1:] if taille > max_octets else lignes:
        try:
            sortie.append(json.loads(l))
        except ValueError:
            pass
    return sortie


@app.get('/stats/live')
def stats_live():
    lg = lire_journal()
    reqs = [e for e in lg if e.get('evt') == 'requete']
    preds = [e for e in lg if e.get('evt') == 'prediction']
    api = [e['ms'] for e in reqs if e.get('chemin') in ('/predict', '/predict-match')]
    api.sort()

    def pct(p):
        return round(api[min(len(api) - 1, int(len(api) * p))], 2) if api else None

    cl = {'2xx': 0, '4xx': 0, '5xx': 0}
    for e in reqs:
        c = e.get('statut', 0)
        cl['2xx' if c < 400 else '4xx' if c < 500 else '5xx'] += 1
    now = time.time()
    minutes = {}                                                # requêtes par minute sur les 30 dernières minutes
    for e in reqs:
        try:
            age = int((now - time.mktime(time.strptime(e['t'], '%Y-%m-%dT%H:%M:%S'))) // 60)
        except (ValueError, KeyError):
            continue
        if 0 <= age < 30:
            minutes[age] = minutes.get(age, 0) + 1
    classes = {c: 0 for c in CLASSES}
    for e in preds:
        if e.get('classe') in classes:
            classes[e['classe']] += 1
    bornes = [0, 2, 5, 10, 20, 50, 100, 1000]                   # histogramme de latence (ms)
    hist = [sum(1 for v in api if bornes[i] <= v < bornes[i + 1]) for i in range(len(bornes) - 1)]
    return {'total': len(reqs), 'predictions': len(preds), 'statuts': cl, 'latence': {'p50': pct(0.5), 'p95': pct(0.95), 'p99': pct(0.99)},
            'par_minute': [minutes.get(29 - i, 0) for i in range(30)], 'classes_predites': classes,
            'hist_latence': {'bornes': bornes, 'n': hist},
            'recentes': [{'t': e['t'][11:], 'classe': e['classe'], 'p': [e.get('p_0'), e.get('p_1'), e.get('p_2')]} for e in preds[-8:]][::-1]}


@app.get('/')
def accueil():
    return FileResponse(BASE / 'static' / 'index.html')
