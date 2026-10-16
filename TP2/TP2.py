import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "Data_set_AntiSPam" / "Uses_Cases" / "Spam"


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

def initialisation(features):
    """Initialise les poids et le biais du neurone de sortie."""
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    return weights, bias

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))

def train_model(features, labels, learning_rate=0.1, iterations=2000):
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    normalized_features = (features - feature_mean) / feature_scale

    weights, bias = initialisation(normalized_features)
    for _ in range(iterations):
        probabilities = sigmoid(normalized_features @ weights + bias)
        weight_gradient = normalized_features.T @ (probabilities - labels) / len(labels)
        bias_gradient = np.mean(probabilities - labels)
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return {
        "weights": weights,
        "bias": bias,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
    }


def train_model_m2(features, labels, hidden_units=32, learning_rate=0.01, iterations=1000):
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    normalized_features = (features - feature_mean) / feature_scale

    rng = np.random.default_rng(42)
    input_weights = rng.normal(
        0.0, np.sqrt(1.0 / normalized_features.shape[1]),
        (normalized_features.shape[1], hidden_units),
    )
    hidden_bias = np.zeros(hidden_units)
    output_weights = rng.normal(0.0, np.sqrt(1.0 / hidden_units), hidden_units)
    output_bias = 0.0

    parameters = [input_weights, hidden_bias, output_weights, output_bias]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for iteration in range(1, iterations + 1):
        hidden_values = np.tanh(normalized_features @ input_weights + hidden_bias)
        probabilities = sigmoid(hidden_values @ output_weights + output_bias)

        output_error = (probabilities - labels) / len(labels)
        output_weights_gradient = hidden_values.T @ output_error
        output_bias_gradient = np.sum(output_error)
        hidden_error = np.outer(output_error, output_weights) * (1.0 - hidden_values**2)
        input_weights_gradient = normalized_features.T @ hidden_error
        hidden_bias_gradient = np.sum(hidden_error, axis=0)

        gradients = [
            input_weights_gradient,
            hidden_bias_gradient,
            output_weights_gradient,
            output_bias_gradient,
        ]
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first_moments[index] = beta1 * first_moments[index] + (1.0 - beta1) * gradient
            second_moments[index] = beta2 * second_moments[index] + (1.0 - beta2) * gradient**2
            corrected_first = first_moments[index] / (1.0 - beta1**iteration)
            corrected_second = second_moments[index] / (1.0 - beta2**iteration)
            parameters[index] -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )

    return {
        "input_weights": input_weights,
        "hidden_bias": hidden_bias,
        "output_weights": output_weights,
        "output_bias": output_bias,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
    }


def predict_m2(features, model):
    normalized_features = (features - model["feature_mean"]) / model["feature_scale"]
    hidden_values = np.tanh(normalized_features @ model["input_weights"] + model["hidden_bias"])
    probabilities = sigmoid(hidden_values @ model["output_weights"] + model["output_bias"])
    return (probabilities >= 0.5).astype(int)


def predict(features, model):
    normalized_features = (features - model["feature_mean"]) / model["feature_scale"]
    probabilities = sigmoid(normalized_features @ model["weights"] + model["bias"])
    return (probabilities >= 0.5).astype(int)

if __name__ == "__main__":
    training_features, training_labels, training_names = load_data(
        DATA_DIR / "Spam detection - For model creation.csv", ";", "GOAL-Spam"
    )
    test_features, test_labels, test_names = load_data(
        DATA_DIR / "Spam detection - For prediction.csv", ",", "Spam"
    )
    if training_names != test_names:
        raise ValueError("Les variables d'entraînement et de test ne sont pas identiques.")

    model = train_model(training_features, training_labels)
    predictions = predict(test_features, model)
    accuracy = np.mean(predictions == test_labels)
    print("Architecture M1 : 57 entrées -> 1 neurone sigmoïde (aucune couche cachée)")
    print(f"Exactitude sur le jeu de test : {accuracy:.2%}")

    model_m2 = train_model_m2(training_features, training_labels)
    predictions_m2 = predict_m2(test_features, model_m2)
    accuracy_m2 = np.mean(predictions_m2 == test_labels)
    print("Architecture M2 : 57 entrées -> 32 neurones tanh -> 1 neurone sigmoïde")
    print(f"Exactitude de M2 sur le jeu de test : {accuracy_m2:.2%}")