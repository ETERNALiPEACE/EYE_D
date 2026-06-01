import argparse
import urllib.request
import zipfile
from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.applications.densenet import DenseNet121


IMG_SIZE = 128
NUM_COORDINATES = 4
EPOCHS = 50
BATCH_SIZE = 16
RANDOM_SEED = 42

BIOID_BASE_URL = "https://ftp.uni-erlangen.de/pub/facedb"
BIOID_IMAGE_ZIP = "BioID-FaceDatabase-V1.2.zip"
BIOID_EYE_ZIP = "BioID-FD-Eyepos-V1.2.zip"


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Plik juz istnieje: {destination}")
        return

    print(f"Pobieranie: {url}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path, output_dir):
    marker = output_dir / f".{zip_path.stem}.extracted"
    if marker.exists():
        print(f"Archiwum juz rozpakowane: {zip_path.name}")
        return

    print(f"Rozpakowywanie: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    marker.write_text("ok", encoding="utf-8")


def download_and_extract_bioid(data_dir):
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_zip = raw_dir / BIOID_IMAGE_ZIP
    eye_zip = raw_dir / BIOID_EYE_ZIP
    download_file(f"{BIOID_BASE_URL}/{BIOID_IMAGE_ZIP}", image_zip)
    download_file(f"{BIOID_BASE_URL}/{BIOID_EYE_ZIP}", eye_zip)
    extract_zip(image_zip, data_dir)
    extract_zip(eye_zip, data_dir)


def parse_eye_file(path):
    """Zwraca [left_x, left_y, right_x, right_y] z pliku BioID .eye."""
    values = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.extend(float(part) for part in line.split())

    if len(values) < NUM_COORDINATES:
        raise ValueError(f"Niepoprawny plik adnotacji oczu: {path}")
    return np.array(values[:NUM_COORDINATES], dtype="float32")


def find_bioid_pairs(data_dir):
    data_dir = Path(data_dir)
    image_paths = {path.stem: path for path in data_dir.rglob("BioID_*.pgm")}
    eye_paths = {path.stem: path for path in data_dir.rglob("BioID_*.eye")}

    common_stems = sorted(set(image_paths) & set(eye_paths))
    if not common_stems:
        raise FileNotFoundError(
            "Nie znaleziono par BioID_*.pgm + BioID_*.eye. "
            "Uruchom skrypt z --download albo sprawdz sciezke --data-dir."
        )

    return [(image_paths[stem], eye_paths[stem]) for stem in common_stems]


def load_bioid_dataset(data_dir):
    pairs = find_bioid_pairs(data_dir)
    images, labels = [], []

    for image_path, eye_path in pairs:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Nie mozna wczytac obrazu: {image_path}")

        height, width = image.shape[:2]
        eye_coordinates = parse_eye_file(eye_path)
        normalized_coordinates = np.array(
            [
                eye_coordinates[0] / width,
                eye_coordinates[1] / height,
                eye_coordinates[2] / width,
                eye_coordinates[3] / height,
            ],
            dtype="float32",
        )

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        images.append(image.astype("float32"))
        labels.append(normalized_coordinates)

    return np.stack(images), np.stack(labels)


def split_dataset(images, labels, validation_fraction=0.15, test_fraction=0.15):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    test_size = int(len(images) * test_fraction)
    validation_size = int(len(images) * validation_fraction)

    test_images = images[:test_size]
    test_labels = labels[:test_size]
    validation_images = images[test_size : test_size + validation_size]
    validation_labels = labels[test_size : test_size + validation_size]
    train_images = images[test_size + validation_size :]
    train_labels = labels[test_size + validation_size :]

    return (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    )


def build_eye_detector():
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="face_image")
    x = keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation=keras.activations.gelu)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(
        NUM_COORDINATES, activation="sigmoid", name="eye_coordinates"
    )(x)

    model = keras.Model(inputs, outputs, name="bioid_eye_detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def plot_training_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_model(data_dir, model_path, epochs):
    images, labels = load_bioid_dataset(data_dir)
    print(f"Liczba zdjec BioID: {len(images)}")
    print(f"Ksztalt danych: {images.shape}; ksztalt etykiet: {labels.shape}")

    (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    ) = split_dataset(images, labels)

    model = build_eye_detector()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_path.with_suffix(".weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    test_loss, test_mae = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test MAE wspolrzednych znormalizowanych: {test_mae:.6f}")

    model.save(model_path)
    plot_training_history(history, Path("outputs") / "training_history.png")
    print(f"Zapisano model: {model_path}")
    return model


def load_image_for_prediction(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Nie mozna wczytac obrazu: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    batch = resized.astype("float32")[None, ...]
    return image_rgb, batch, original_width, original_height


def draw_eye_prediction(image_rgb, coordinates, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = image_rgb.shape[:2]
    left_eye = (int(coordinates[0] * width), int(coordinates[1] * height))
    right_eye = (int(coordinates[2] * width), int(coordinates[3] * height))

    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    box_width = max(12, int(width * 0.10))
    box_height = max(8, int(height * 0.06))

    for label, center, color in [
        ("left eye", left_eye, (0, 255, 0)),
        ("right eye", right_eye, (255, 0, 0)),
    ]:
        cv2.circle(annotated, center, radius=4, color=color, thickness=-1)
        top_left = (center[0] - box_width // 2, center[1] - box_height // 2)
        bottom_right = (center[0] + box_width // 2, center[1] + box_height // 2)
        cv2.rectangle(annotated, top_left, bottom_right, color=color, thickness=2)
        cv2.putText(
            annotated,
            label,
            (top_left[0], max(12, top_left[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated)
    return left_eye, right_eye


def predict_eyes(model, image_paths, output_dir):
    for image_path in image_paths:
        image_path = Path(image_path)
        image_rgb, batch, _, _ = load_image_for_prediction(image_path)
        coordinates = model.predict(batch, verbose=0)[0]
        coordinates = np.clip(coordinates, 0.0, 1.0)

        output_path = Path(output_dir) / f"{image_path.stem}_eyes.jpg"
        left_eye, right_eye = draw_eye_prediction(image_rgb, coordinates, output_path)
        print(f"{image_path}: lewe oko={left_eye}, prawe oko={right_eye}")
        print(f"Zapisano obraz z detekcja: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trening modelu DenseNet121 do wykrywania oczu ludzi na zdjeciach."
    )
    parser.add_argument("--data-dir", default="data/bioid", help="Katalog na zbior BioID.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pobierz i rozpakuj BioID Face Database oraz pliki .eye.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Liczba epok treningu. Domyslnie 50.",
    )
    parser.add_argument(
        "--model-path",
        default="models/bioid_eye_detector.keras",
        help="Sciezka zapisu lub odczytu modelu.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Sciezki do zdjec, na ktorych zaznaczyc oczy.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/predictions",
        help="Katalog zapisu zdjec z zaznaczonymi oczami.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Nie trenuj modelu, tylko wczytaj --model-path i wykonaj predykcje.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download:
        download_and_extract_bioid(args.data_dir)

    if args.skip_train:
        model = keras.models.load_model(args.model_path)
    else:
        model = train_model(args.data_dir, args.model_path, args.epochs)

    if args.images:
        predict_eyes(model, args.images, args.output_dir)


if __name__ == "__main__":
    main()
