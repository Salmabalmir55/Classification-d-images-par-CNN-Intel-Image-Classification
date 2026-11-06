"""Interface Streamlit pour le TP de classification Intel Image Classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from train_model import ARTIFACT_DIR, discover_train_dir, train_model


st.set_page_config(page_title="Nature & paysages | TP CNN", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(145deg, #f4f8f6 0%, #eef3f8 100%); }
    .block-container { max-width: 1120px; padding-top: 2.1rem; }
    [data-testid="stMetric"] { background: white; padding: 1rem; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 Reconnaissance de paysages")
st.caption("TP CNN · Intel Image Classification · Transfert d’apprentissage MobileNetV2")

MODEL_PATH = ARTIFACT_DIR / "mobilenetv2_intel.keras"
LABELS_PATH = ARTIFACT_DIR / "class_names.json"


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def load_labels() -> list[str]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=300, show_spinner=False)
def find_dataset_classes(dataset_path: str) -> str | None:
    found = discover_train_dir(Path(dataset_path).expanduser())
    return str(found) if found else None


def render_training_controls(button_label: str, widget_key: str) -> None:
    dataset_path = st.text_input(
        "Dossier du dataset",
        value="data",
        key=f"dataset_path_{widget_key}",
        help="Choisissez le dossier Intel Image Classification ou son dossier seg_train.",
    )
    dataset_root = Path(dataset_path).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parent / dataset_root

    if st.button("Vérifier le dossier", key=f"detect_{widget_key}"):
        with st.spinner("Vérification de la structure du dataset…"):
            found = find_dataset_classes(str(dataset_root.resolve()))
        st.session_state[f"detected_train_dir_{widget_key}"] = found
        st.session_state[f"detected_root_{widget_key}"] = str(dataset_root.resolve())

    detected = st.session_state.get(f"detected_train_dir_{widget_key}")
    if st.session_state.get(f"detected_root_{widget_key}") != str(dataset_root.resolve()):
        detected = None
    if not detected:
        st.caption("Placez les images par classe dans `seg_train/buildings/…`, `seg_train/forest/…`, etc., puis vérifiez le dossier.")
        return

    train_dir = Path(detected)
    st.caption(f"Classes détectées : `{train_dir}`")
    epochs = st.slider("Époques maximum", 5, 30, 10, key=f"epochs_{widget_key}")

    if st.button(button_label, type="primary", width="stretch", key=f"train_{widget_key}"):
        progress = st.progress(0, text="Chargement des données et du modèle…")
        status = st.empty()
        status.write("La préparation initiale peut prendre quelques minutes.")

        def report(epoch: int, total: int, val_acc: float) -> None:
            progress.progress(min(0.96, epoch / total), text=f"Époque {epoch}/{total}")
            status.write(f"Précision validation : **{val_acc:.1%}**")

        def report_batch(
            phase: str, epoch: int, total: int, batch: int, batches: int, epoch_fraction: float
        ) -> None:
            fraction = ((epoch - 1) + epoch_fraction) / total
            progress.progress(
                min(0.96, fraction),
                text=f"{phase} — époque {epoch}/{total}, lot {batch}/{batches}",
            )

        try:
            with st.spinner("Entraînement MobileNetV2 en cours…"):
                train_model(train_dir, max_epochs=epochs, on_epoch=report, on_batch=report_batch)
            progress.progress(1.0, text="Modèle enregistré")
            load_model.clear()
            st.success("Entraînement terminé.")
            st.rerun()
        except Exception as exc:
            st.error(f"Échec de l’entraînement : {exc}")


with st.sidebar:
    st.header("Modèle")
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        st.success("Modèle entraîné prêt")
        st.caption("Le modèle enregistré est chargé uniquement au premier classement.")
        if st.toggle("Afficher les options de réentraînement", key="show_retrain_options"):
            render_training_controls("Réentraîner le modèle", "retrain")
    else:
        st.info("Aucun modèle entraîné trouvé.")
        render_training_controls("Entraîner le modèle", "train")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.info(
        "Pour démarrer, placez le jeu Intel Image Classification dans `data/` avec son dossier "
        "`seg_train`, puis lancez l’entraînement depuis le panneau à gauche."
    )
    st.markdown(
        "Le modèle utilise un backbone **MobileNetV2 préentraîné sur ImageNet**, puis un fine-tuning "
        "léger sur les classes du TP. L’early stopping conserve les poids ayant la meilleure précision "
        "de validation."
    )
    st.stop()

labels = load_labels()

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("Analyser une image")
    uploaded = st.file_uploader("Importer une photo de paysage", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Image importée", width="stretch")
        if st.button("Classifier l’image", type="primary", width="stretch"):
            with st.spinner("Chargement du modèle et analyse…"):
                model = load_model(str(MODEL_PATH))
                height, width = model.input_shape[1:3]
                resized = image.resize((int(width), int(height)))
                batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
                probabilities = model.predict(batch, verbose=0)[0]
                order = np.argsort(probabilities)[::-1]
                st.session_state["prediction"] = {
                    "label": labels[int(order[0])],
                    "confidence": float(probabilities[order[0]]),
                    "scores": {labels[int(i)]: float(probabilities[i]) for i in order},
                }
with right:
    st.subheader("Résultat")
    result = st.session_state.get("prediction")
    if result:
        st.metric("Classe prédite", result["label"].replace("_", " ").title())
        st.metric("Score du modèle", f"{result['confidence']:.1%}")
        st.caption("Le score mesure la confiance parmi les classes connues. Une image hors du dataset peut tout de même recevoir une classe.")
        st.markdown("**Probabilités par classe**")
        for name, score in result["scores"].items():
            st.progress(score, text=f"{name.replace('_', ' ').title()} · {score:.1%}")
    else:
        st.write("Importez une image puis cliquez sur **Classifier l’image**.")

st.divider()
st.caption("Classes du dataset : " + " · ".join(name.replace("_", " ") for name in labels))
