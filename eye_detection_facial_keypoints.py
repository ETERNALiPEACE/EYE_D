import argparse
import urllib.request
import zipfile
from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.applications.densenet import DenseNet121


IMG_SIZE = 128
SOURCE_IMG_SIZE = 96
NUM_COORDINATES = 4
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_SEED = 42

DATASET_URL = (
    "https://raw.githubusercontent.com/ruchawaghulde/"
    "Facial-Keypoints-Detection/master/Data/training.zip"
)
DATASET_ZIP = "training.zip"
TRAINING_CSV = "training.csv"

EYE_COLUMNS = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
]


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Plik juz istnieje: {destination}")
        return

    print(f"Pobieranie: {url}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path, output_dir):
    training_csv = output_dir / TRAINING_CSV
    if training_csv.exists():
        print(f"Plik CSV juz istnieje: {training_csv}")
        return

    print(f"Rozpakowywanie: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def download_and_extract_dataset(data_dir):
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_zip = raw_dir / DATASET_ZIP
    download_file(DATASET_URL, dataset_zip)
    extract_zip(dataset_zip, data_dir)


def image_string_to_array(image_string):
    image = np.fromstring(image_string, sep=" ", dtype="float32")
    if image.size != SOURCE_IMG_SIZE * SOURCE_IMG_SIZE:
        raise ValueError("Niepoprawna liczba pikseli w kolumnie Image.")
    return image.reshape(SOURCE_IMG_SIZE, SOURCE_IMG_SIZE)


def load_keypoints_dataset(data_dir, max_samples=None):
    csv_path = Path(data_dir) / TRAINING_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {csv_path}. Uruchom skrypt z --download."
        )

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=EYE_COLUMNS + ["Image"]).reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)

    images, labels = [], []
    for _, row in df.iterrows():
        image = image_string_to_array(row["Image"])
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_GRAY2RGB)
        images.append(image.astype("float32"))

        labels.append(row[EYE_COLUMNS].to_numpy(dtype="float32") / SOURCE_IMG_SIZE)

    images = np.stack(images)
    labels = np.stack(labels)
    print(f"Liczba zdjec twarzy: {len(images)}")
    print(f"Ksztalt obrazow: {images.shape}; ksztalt etykiet: {labels.shape}")
    return images, labels


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

    model = keras.Model(inputs, outputs, name="facial_keypoints_eye_detector")
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


def save_dataset_preview(images, labels, output_path, count=9):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(count, len(images))

    columns = 3
    rows = int(np.ceil(count / columns))
    plt.figure(figsize=(columns * 3, rows * 3))

    for idx in range(count):
        ax = plt.subplot(rows, columns, idx + 1)
        image = images[idx].astype("uint8")
        left_eye = labels[idx, :2] * IMG_SIZE
        right_eye = labels[idx, 2:] * IMG_SIZE
        ax.imshow(image)
        ax.scatter([left_eye[0], right_eye[0]], [left_eye[1], right_eye[1]], c=["lime", "red"])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Zapisano podglad datasetu: {output_path}")


def train_model(data_dir, model_path, epochs, max_samples=None):
    images, labels = load_keypoints_dataset(data_dir, max_samples=max_samples)
    save_dataset_preview(images, labels, Path("outputs") / "dataset_preview.png")

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


def find_largest_face(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda box: box[2] * box[3])


def load_face_for_prediction(image_path, detect_face=True):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Nie mozna wczytac obrazu: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    face_box = None

    if detect_face:
        face_box = find_largest_face(image_rgb)

    if face_box is None:
        face_box = (0, 0, width, height)

    x, y, w, h = face_box
    padding = int(0.08 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    face_rgb = image_rgb[y1:y2, x1:x2]
    resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    batch = resized.astype("float32")[None, ...]
    return image_rgb, batch, (x1, y1, x2 - x1, y2 - y1)


def draw_eye_prediction(image_rgb, face_box, coordinates, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x, y, w, h = face_box
    left_eye = (int(x + coordinates[0] * w), int(y + coordinates[1] * h))
    right_eye = (int(x + coordinates[2] * w), int(y + coordinates[3] * h))

    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)
    box_width = max(12, int(w * 0.18))
    box_height = max(8, int(h * 0.10))

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


def predict_eyes(model, image_paths, output_dir, detect_face=True):
    for image_path in image_paths:
        image_path = Path(image_path)
        image_rgb, batch, face_box = load_face_for_prediction(
            image_path, detect_face=detect_face
        )
        coordinates = model.predict(batch, verbose=0)[0]
        coordinates = np.clip(coordinates, 0.0, 1.0)

        output_path = Path(output_dir) / f"{image_path.stem}_eyes.jpg"
        left_eye, right_eye = draw_eye_prediction(
            image_rgb, face_box, coordinates, output_path
        )
        print(f"{image_path}: lewe oko={left_eye}, prawe oko={right_eye}")
        print(f"Zapisano obraz z detekcja: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trening modelu DenseNet121 do wykrywania oczu na bazie "
            "Kaggle Facial Keypoints Detection."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/facial-keypoints",
        help="Katalog na training.csv.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pobierz i rozpakuj publiczny mirror training.zip.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Liczba epok treningu. Domyslnie 50.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Opcjonalny limit probek do szybkiego testu kodu.",
    )
    parser.add_argument(
        "--model-path",
        default="models/facial_keypoints_eye_detector.keras",
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
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Nie kadruj twarzy detektorem OpenCV przy predykcji.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download:
        download_and_extract_dataset(args.data_dir)

    if args.skip_train:
        model = keras.models.load_model(args.model_path)
    else:
        model = train_model(
            args.data_dir,
            args.model_path,
            args.epochs,
            max_samples=args.max_samples,
        )

    if args.images:
        predict_eyes(
            model,
            args.images,
            args.output_dir,
            detect_face=not args.no_face_detect,
        )


if __name__ == "__main__":
    main()
