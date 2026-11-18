# football-api

API FastAPI de prédiction du résultat d'un match de football. Le modèle est une **régression logistique** entraînée sur les indicateurs Elo, la forme récente des équipes et la valeur estimée des effectifs.

## Arborescence

```text
football-api/
├── app.py                 # API FastAPI
├── features.py            # Construction des variables
├── model/                 # Pipeline, métadonnées et état des équipes
├── static/index.html      # Interface web
├── test_service.py        # Tests pytest
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .dockerignore
├── .gitignore
└── logs/service.jsonl     # Journal JSON applicatif
```

## Installation locale sous Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Lancement local

```powershell
python -m uvicorn app:app --port 8000
```

- Interface web: http://127.0.0.1:8000/
- Documentation Swagger: http://127.0.0.1:8000/docs

## Tests

```powershell
python -m pytest -q
```

## Docker Compose

```powershell
docker compose up -d --build
docker compose logs -f
docker compose down
```

Le service écoute sur `127.0.0.1:8000`, monte `./logs` dans `/app/logs` et redémarre automatiquement sauf arrêt explicite.

## Endpoints

| Endpoint | Méthode | Exemple curl.exe |
|---|---|---|
| `/health` | GET | `curl.exe -s http://127.0.0.1:8000/health` |
| `/competitions` | GET | `curl.exe -s http://127.0.0.1:8000/competitions` |
| `/equipes?q=` | GET | `curl.exe -s "http://127.0.0.1:8000/equipes?q=Paris&limite=20"` |
| `/predict` | POST | `curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@payload-predict.json"` |
| `/predict-match` | POST | `curl.exe -s -X POST http://127.0.0.1:8000/predict-match -H "Content-Type: application/json" -d "@payload.json"` |

Exemple de fichier `payload.json`:

```json
{"equipe_dom":27,"equipe_ext":15,"competition_type":"domestic_league"}
```

Exemple complet via `payload-predict.json`:

```json
{"elo_dom":2036,"elo_ext":1843,"elo_diff":193,"pts_dom":1.6,"pts_ext":1.4,"pts_diff":0.2,"bp_dom":2.8,"bc_dom":2.0,"bp_ext":1.6,"bc_ext":1.6,"val_diff":0.34,"val_moy":8.8,"repos_dom":30,"repos_ext":30,"val_connue":1,"competition_type":"domestic_league"}
```

## Codes d'erreur

- `200`: requête réussie.
- `400`: règles métier non respectées, par exemple deux équipes identiques ou historique insuffisant.
- `404`: équipe inconnue ou inactive.
- `413`: corps JSON supérieur à la taille maximale acceptée.
- `422`: données invalides ou champs hors contraintes.
- `500`: erreur interne, avec détail masqué au client et conservé dans le journal.

## Limites

Le modèle prédit principalement une victoire à domicile ou à l'extérieur; la classe **nul** est rarement prédite. Avant une exposition publique, ajouter une authentification et une limitation de débit (rate limiting).

## Supervision

Les événements sont enregistrés en une ligne JSON dans `logs/service.jsonl`. Le journal contient notamment le modèle utilisé, le type d'événement, la méthode, le chemin, le code HTTP, la durée de traitement, la classe prédite et les probabilités agrégées. Aucun secret ni donnée personnelle n'est écrit volontairement dans le journal.
