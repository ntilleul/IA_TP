import csv
from pathlib import Path

import keras
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "Data_set_AntiSpam" / "Uses_Cases" / "Spam"


def load_data(path, delimiter, target_column):
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        if reader.fieldnames is None or target_column not in reader.fieldnames:
            raise ValueError(f"Colonne cible absente : {target_column}")

        feature_names = [name for name in reader.fieldnames if name != target_column]
        rows = list(reader)

    label_mapping = {"no": 0.0, "yes": 1.0, "0": 0.0, "1": 1.0}
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    labels = np.asarray(
        [label_mapping[row[target_column].strip().lower()] for row in rows], dtype=float
    )
    return features, labels, feature_names

def build_model_m1(training_features):
    normalization = keras.layers.Normalization(axis=-1)
    normalization.adapt(training_features)

    model = keras.Sequential(
        [
            keras.Input(shape=(training_features.shape[1],)),
            normalization,
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="M1_sans_couche_cachee",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_m2(training_features, hidden_units=32):
    normalization = keras.layers.Normalization(axis=-1)
    normalization.adapt(training_features)

    model = keras.Sequential(
        [
            keras.Input(shape=(training_features.shape[1],)),
            normalization,
            keras.layers.Dense(hidden_units, activation="tanh"),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="M2_avec_couche_cachee",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, training_features, training_labels, test_features, test_labels):
    model.fit(
        training_features,
        training_labels.reshape(-1, 1),
        epochs=100,
        batch_size=32,
        verbose=0,
    )
    probabilities = model.predict(test_features, verbose=0).ravel()
    return float(np.mean((probabilities >= 0.5) == test_labels))

if __name__ == "__main__":
    training_features, training_labels, training_names = load_data(
        DATA_DIR / "Spam detection - For model creation.csv", ";", "GOAL-Spam"
    )
    test_features, test_labels, test_names = load_data(
        DATA_DIR / "Spam detection - For prediction.csv", ",", "Spam"
    )
    if training_names != test_names:
        raise ValueError("Les variables d'entraînement et de test ne sont pas identiques.")

    keras.utils.set_random_seed(42)
    model_m1 = build_model_m1(training_features)
    accuracy_m1 = evaluate_model(
        model_m1, training_features, training_labels, test_features, test_labels
    )
    print("Architecture M1 : 57 entrées -> 1 neurone sigmoïde (aucune couche cachée)")
    print(f"Exactitude de M1 sur le jeu de test : {accuracy_m1:.2%}")

    model_m2 = build_model_m2(training_features)
    accuracy_m2 = evaluate_model(
        model_m2, training_features, training_labels, test_features, test_labels
    )
    print("Architecture M2 : 57 entrées -> 32 neurones tanh -> 1 neurone sigmoïde")
    print(f"Exactitude de M2 sur le jeu de test : {accuracy_m2:.2%}")