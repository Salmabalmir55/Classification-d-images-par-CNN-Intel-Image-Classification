# TP CNN — Application Streamlit

Projet de classification des paysages Intel Image Classification, basé sur `TP_CNN_final.ipynb`.

## Contenu

- `app.py` : interface Streamlit pour importer et classifier une image.
- `train_model.py` : entraînement MobileNetV2, validation, fine-tuning, early stopping et sauvegarde.
- `TP_CNN_final.ipynb` : notebook fourni pour le TP.
- `requirements.txt` : dépendances Python.
- `data/` : emplacement local du dataset (non inclus dans l’archive).
- `artifacts/` : emplacement du modèle et des étiquettes (non inclus par défaut).

## Utiliser le modèle déjà entraîné du notebook

Le notebook contient les sorties de l’entraînement, mais ne sauvegarde pas le modèle final dans un fichier. Si le kernel du notebook possède encore `final_model` et `class_names`, exécute cette cellule après avoir extrait le projet. Remplace le chemin par le dossier extrait :

```python
from pathlib import Path
import json

APP_DIR = Path(r"C:\chemin\vers\TP_CNN_Streamlit")
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
final_model.save(ARTIFACT_DIR / "mobilenetv2_intel.keras")
(ARTIFACT_DIR / "class_names.json").write_text(
    json.dumps(class_names, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

L’application lit la taille d’entrée du modèle sauvegardé, donc elle peut utiliser le modèle exporté du notebook (150 × 150) ou celui entraîné par le script du projet (224 × 224). Si le notebook tourne dans Colab, télécharge les deux fichiers créés puis place-les dans le dossier `artifacts/`.

## Installation

Python 3.10 à 3.12 est recommandé. Depuis le dossier du projet :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Télécharge et décompresse Intel Image Classification dans `data/`. Structure attendue :

```text
data/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_test/        # facultatif pour l’application
```

Les structures Kaggle imbriquées `seg_train/seg_train/classe/…` sont également détectées. Le dossier de test n’est pas utilisé pour entraîner l’application.

## Démarrer

```powershell
streamlit run app.py
```

L’application s’ouvre dans le navigateur. Elle charge le modèle sauvegardé à la première demande de classification, puis le garde en cache. Si aucun modèle n’est présent dans `artifacts/`, vérifie le dossier du dataset dans la barre latérale et clique sur **Entraîner le modèle**. Cet entraînement initial peut prendre plusieurs minutes, surtout sans GPU. Le modèle et les classes sont ensuite enregistrés dans `artifacts/`; les lancements suivants ne réentraînent pas le modèle.

Pour éviter les recherches répétées dans le dataset, la vérification des dossiers de classes se lance seulement après le clic sur **Vérifier le dossier**. Le code prend en charge le double dossier `seg_train/seg_train`.

## Entraîner depuis le terminal

```powershell
python train_model.py data --epochs 10
```

## Interpréter le score

La softmax partage ses probabilités entre les classes connues. Un score élevé n’assure pas qu’une image appartient réellement au dataset : une image hors sujet peut quand même recevoir l’une des six classes.
