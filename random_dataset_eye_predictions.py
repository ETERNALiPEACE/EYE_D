import argparse
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from eye_detection_facial_keypoints import (
    IMG_SIZE,
    TRAINING_CSV,
    download_and_extract_dataset,
    load_keypoints_dataset,
)


DEFAULT_MODEL_PATH = "models/facial_keypoints_eye_detector.keras"
DEFAULT_OUTPUT_DIR = "outputs/random_dataset_predictions"


def ensure_dataset(data_dir, download):
    data_dir = Path(data_dir)
    dataset_path = data_dir / TRAINING_CSV

    if download or not dataset_path.exists():
        if not dataset_path.exists():
            print(f"Dataset file {dataset_path} not found. Forcing download.")
        download_and_extract_dataset(data_dir)


def load_trained_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {model_path}. "
            "Najpierw uruchom trening, np. "
            "`python eye_detection_facial_keypoints.py --download --epochs 50`."
        )

    return keras.models.load_model(model_path)


def normalized_to_pixels(coordinates):
    coordinates = np.asarray(coordinates, dtype="float32")
    return coordinates.reshape(2, 2) * IMG_SIZE


def plot_single_prediction(image, prediction, target, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_points = normalized_to_pixels(prediction)
    target_points = normalized_to_pixels(target)

    plt.figure(figsize=(4, 4))
    plt.imshow(image.astype("uint8"))
    plt.scatter(
        predicted_points[:, 0],
        predicted_points[:, 1],
        c=["lime", "red"],
        marker="o",
        s=60,
        label="predykcja",
    )
    plt.scatter(
        target_points[:, 0],
        target_points[:, 1],
        c=["cyan", "magenta"],
        marker="x",
        s=70,
        label="etykieta z bazy",
    )
    plt.title(title)
    plt.axis("off")
    plt.legend(loc="lower center", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()


def plot_prediction_grid(images, predictions, targets, indices, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = len(images)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    plt.figure(figsize=(columns * 3, rows * 3))

    for plot_idx, image in enumerate(images):
        ax = plt.subplot(rows, columns, plot_idx + 1)
        predicted_points = normalized_to_pixels(predictions[plot_idx])
        target_points = normalized_to_pixels(targets[plot_idx])

        ax.imshow(image.astype("uint8"))
        ax.scatter(
            predicted_points[:, 0],
            predicted_points[:, 1],
            c=["lime", "red"],
            marker="o",
            s=45,
        )
        ax.scatter(
            target_points[:, 0],
            target_points[:, 1],
            c=["cyan", "magenta"],
            marker="x",
            s=55,
        )
        ax.set_title(f"idx {indices[plot_idx]}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Kolko = predykcja modelu, X = etykieta z bazy", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"Zapisano siatke predykcji: {output_path}")


def save_random_dataset_predictions(
    model,
    images,
    labels,
    count,
    output_dir,
    seed=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if count <= 0:
        raise ValueError("--count musi byc wieksze od 0.")
    if len(images) == 0:
        raise ValueError("Dataset nie zawiera zadnych obrazow do predykcji.")

    count = min(count, len(images))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=count, replace=False)
    selected_images = images[indices]
    selected_labels = labels[indices]

    predictions = model.predict(selected_images, batch_size=min(32, count), verbose=0)
    predictions = np.clip(predictions, 0.0, 1.0)

    for local_idx, dataset_idx in enumerate(indices):
        output_path = output_dir / f"random_{local_idx + 1:02d}_idx_{dataset_idx}.png"
        plot_single_prediction(
            selected_images[local_idx],
            predictions[local_idx],
            selected_labels[local_idx],
            output_path,
            title=f"Dataset index: {dataset_idx}",
        )
        print(f"Zapisano predykcje: {output_path}")

    plot_prediction_grid(
        selected_images,
        predictions,
        selected_labels,
        indices,
        output_dir / "random_predictions_grid.png",
    )
    print(f"Wylosowane indeksy z datasetu: {indices.tolist()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Losuje zdjecia z bazy Facial Keypoints i wykonuje predykcje oczu "
            "zapisanym modelem."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/facial-keypoints",
        help="Katalog zawierajacy training.csv.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pobierz dataset, jesli chcesz wymusic pobranie/rozpakowanie.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Sciezka do wytrenowanego modelu .keras.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Liczba losowych zdjec z datasetu do predykcji.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed losowania. Ustaw np. 42, jesli chcesz powtarzalne wyniki.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Opcjonalny limit wczytywanych probek z datasetu.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Katalog zapisu losowych predykcji.",
    )
    # Colab/Jupyter dodaje wlasne argumenty uruchomieniowe.
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    ensure_dataset(args.data_dir, args.download)
    images, labels = load_keypoints_dataset(
        args.data_dir,
        max_samples=args.max_samples,
    )
    model = load_trained_model(args.model_path)
    save_random_dataset_predictions(
        model=model,
        images=images,
        labels=labels,
        count=args.count,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
