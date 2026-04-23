# MLOps Heart Disease Prediction

## Projet Final — MLOps II | SupNum DEML M2

Pipeline MLOps complet pour la prédiction de maladies cardiaques à partir de données cliniques.

---

## Sujet

Ce projet implémente un pipeline MLOps de bout en bout autour d'un modèle de classification binaire qui prédit la présence ou l'absence de maladie cardiaque chez un patient, à partir de 13 caractéristiques cliniques (âge, sexe, type de douleur thoracique, tension artérielle, cholestérol, etc.).

## Membres du groupe

| Nom | Matricule |
|-----|-----------|
| Membre 1 | 21029 |
| Membre 2 | 21011 |
| Membre 3 | 21061 |


## Services

| Service | URL |
|---------|-----|
| MLflow UI | http://16.16.184.117:5000 |
| API de prédiction | http://16.16.184.117:8000 |
| Health Check | http://16.16.184.117:8000/ |

## Architecture

```
GitHub Repository (main / staging / dev-MATRICULE)
         │                        │
    merge → main            .dvc modifié
         │                        │
         ▼                        ▼
    Workflow 1               Workflow 2
    (Code CI)                (Data CI)
    → SSH → EC2              → SSH → EC2
    → docker build           → dvc repro
    → restart API            → MLflow log
    → health check           → restart API
         │
         ▼
    AWS EC2 (t3.medium)
    ┌─────────────┬──────────────┐
    │ MLflow :5000 │ Docker API   │
    │ (Registry +  │ FastAPI      │
    │ Experiments) │ Interface Web│
    └──────┬───────┴──────┬───────┘
           │              │
           ▼              ▼
    AWS S3 Bucket
    ├── data/raw/
    ├── data/processed/
    ├── mlflow-artifacts/
    └── dvc-store/
```

## Structure du projet

```
mlops-heart-disease/
├── src/
│   ├── preprocess.py       # Nettoyage, encodage, normalisation, split
│   ├── train.py            # Entraînement RandomForest + logging MLflow
│   └── evaluate.py         # Évaluation sur jeu de test + métriques MLflow
├── app/
│   ├── main.py             # API FastAPI (endpoints /predict et /)
│   ├── Dockerfile          # Conteneurisation de l'API
│   └── templates/
│       └── index.html      # Interface utilisateur web
├── data/
│   ├── raw/                # Dataset brut (géré par DVC)
│   └── processed/          # Données preprocessées (géré par DVC)
├── models/                 # Modèle entraîné (géré par DVC)
├── .github/
│   └── workflows/
│       ├── deploy.yml      # Workflow 1 : CI/CD sur merge main
│       └── retrain.yml     # Workflow 2 : Retraining sur données
├── dvc.yaml                # Définition du pipeline DVC
├── dvc.lock                # Hash des données versionnées
├── data/raw.dvc            # Référence DVC vers S3
├── params.yaml             # Paramètres du pipeline (non versionné)
├── requirements.txt        # Dépendances Python
├── .gitignore
└── README.md
```

## Pipeline DVC

Le pipeline ML est découpé en 3 stages DVC, exécutables avec une seule commande :

| Stage | Description |
|-------|-------------|
| `preprocess` | Chargement depuis S3, nettoyage, normalisation, split train/test (80/20) |
| `train` | Entraînement RandomForest avec GridSearchCV, logging dans MLflow |
| `evaluate` | Évaluation sur jeu de test, logging accuracy et F1-score dans MLflow |

```bash
# Exécuter tout le pipeline
dvc repro

# Pousser les données vers S3
dvc push

# Récupérer les données depuis S3
dvc pull
```

## Modèle

- **Algorithme** : RandomForestClassifier avec GridSearchCV
- **Features** : 13 caractéristiques cliniques (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target** : Présence de maladie cardiaque (0 = non, 1 = oui)
- **Métriques** : Accuracy ~85%, F1-score ~86%

## Déploiement

### Prérequis

- Python 3.10+
- Docker
- AWS CLI configuré
- Accès au bucket S3 `mlops-heart-disease-groupe-2`

### Installation locale

```bash
git clone https://github.com/TON-USERNAME/mlops-heart-disease.git
cd mlops-heart-disease
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configurer params.yaml

Créer un fichier `params.yaml` à la racine avec les paramètres et credentials AWS (ce fichier n'est pas versionné).

### Récupérer les données et lancer le pipeline

```bash
dvc pull
dvc repro
```

### Lancer l'API en local

```bash
cd app
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## CI/CD

### Workflow 1 — Déploiement (merge sur main)

Déclenché automatiquement à chaque merge vers `main`. Se connecte à l'EC2 via SSH, met à jour le code, reconstruit l'image Docker, redémarre l'API et vérifie le health check.

### Workflow 2 — Retraining (manuel / données modifiées)

Déclenché manuellement ou sur modification des données. Relance `dvc repro`, enregistre le nouveau modèle dans MLflow, et redémarre l'API.

## Branches

| Branche | Rôle |
|---------|------|
| `main` | Production — ne reçoit que des merges depuis staging |
| `staging` | Intégration — reçoit les merges des branches dev |
| `dev-XXXXX` | Branche individuelle par membre du groupe |

## Encadrant

Professeur Yehdhih ANNA — Institut Supérieur du Numérique (SupNum)










