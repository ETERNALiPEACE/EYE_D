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
CELEBA_WIDTH = 178
CELEBA_HEIGHT = 218
NUM_COORDINATES = 4
EPOCHS = 50
BATCH_SIZE = 16
RANDOM_SEED = 42
DEFAULT_MAX_SAMPLES = 2000
RANDOM_DATASET_OUTPUT_DIR = "outputs/random_dataset_predictions"
DEFAULT_EYE_CROP_OUTPUT_DIR = "outputs/eye_crops"

CELEBA_BASE_URL = "https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba"
CELEBA_IMAGES_DIR = "img_align_celeba"
CELEBA_IMAGE_ZIP = "img_align_celeba.zip"
CELEBA_LANDMARKS_FILE = "list_landmarks_align_celeba.txt"

# Prosta konfiguracja do Colab/Jupyter. Zmien te wartosci przed uruchomieniem.
DATA_DIR = "data/celeba"
MODEL_PATH = "models/celeba_eye_detector.keras"
PREDICTION_OUTPUT_DIR = "outputs/predictions"
DOWNLOAD_DATA = False
SKIP_TRAIN = False
MAX_SAMPLES = DEFAULT_MAX_SAMPLES
IMAGE_PATHS = []
NO_FACE_DETECT = False

RUN_RANDOM_DATASET = False
RANDOM_COUNT = 8
RANDOM_DATASET_SEED = None
RANDOM_OUTPUT_DIR = RANDOM_DATASET_OUTPUT_DIR

SAVE_EYE_CROPS = False
EYE_CROP_COUNT = 32
EYE_CROP_PADDING = 0.45
EYE_CROP_OUTPUT_DIR = DEFAULT_EYE_CROP_OUTPUT_DIR


def normalize_max_samples(max_samples):
    if max_samples is None or max_samples <= 0:
        return None
    return max_samples


def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Plik juz istnieje: {destination}")
        return

    print(f"Pobieranie: {url}")
    urllib.request.urlretrieve(url, destination)


def download_celeba_landmarks(data_dir):
    data_dir = Path(data_dir)
    landmarks_path = data_dir / CELEBA_LANDMARKS_FILE
    download_file(f"{CELEBA_BASE_URL}/{CELEBA_LANDMARKS_FILE}", landmarks_path)
    return landmarks_path


def download_celeba_image_zip(data_dir):
    data_dir = Path(data_dir)
    zip_path = data_dir / "raw" / CELEBA_IMAGE_ZIP
    download_file(f"{CELEBA_BASE_URL}/{CELEBA_IMAGE_ZIP}", zip_path)
    return zip_path


def read_celeba_landmark_records(data_dir, max_samples=None):
    landmarks_path = Path(data_dir) / CELEBA_LANDMARKS_FILE
    if not landmarks_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {landmarks_path}. Wlacz DOWNLOAD_DATA "
            "albo pozwol na automatyczne pobranie danych."
        )

    max_samples = normalize_max_samples(max_samples)
    records = []
    with open(landmarks_path, "r", encoding="utf-8") as handle:
        total_images = int(handle.readline().strip())
        header = handle.readline().split()
        expected_header = [
            "lefteye_x",
            "lefteye_y",
            "righteye_x",
            "righteye_y",
            "nose_x",
            "nose_y",
            "leftmouth_x",
            "leftmouth_y",
            "rightmouth_x",
            "rightmouth_y",
        ]
        if header != expected_header:
            raise ValueError(f"Nieoczekiwany naglowek landmarkow CelebA: {header}")

        for line in handle:
            parts = line.split()
            if len(parts) != 11:
                continue

            filename = parts[0]
            values = np.array([float(value) for value in parts[1:]], dtype="float32")
            eye_coordinates = np.array(
                [
                    values[0] / CELEBA_WIDTH,
                    values[1] / CELEBA_HEIGHT,
                    values[2] / CELEBA_WIDTH,
                    values[3] / CELEBA_HEIGHT,
                ],
                dtype="float32",
            )
            records.append((filename, eye_coordinates))

            if max_samples and len(records) >= max_samples:
                break

    print(f"CelebA landmark file contains {total_images} images.")
    print(f"Using {len(records)} color face images.")
    return records


def extract_celeba_images(data_dir, records):
    data_dir = Path(data_dir)
    images_dir = data_dir / CELEBA_IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    missing_filenames = [
        filename for filename, _ in records if not (images_dir / filename).exists()
    ]
    if not missing_filenames:
        print("Wszystkie wymagane obrazy CelebA sa juz rozpakowane.")
        return

    zip_path = download_celeba_image_zip(data_dir)
    print(f"Rozpakowywanie {len(missing_filenames)} obrazow z {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive_members = set(archive.namelist())
        for idx, filename in enumerate(missing_filenames, start=1):
            member_path = f"{CELEBA_IMAGES_DIR}/{filename}"
            if member_path in archive_members:
                archive.extract(member_path, data_dir)
            elif filename in archive_members:
                archive.extract(filename, images_dir)
            else:
                raise FileNotFoundError(
                    f"Nie znaleziono {filename} w archiwum {zip_path}."
                )

            if idx % 100 == 0 or idx == len(missing_filenames):
                print(f"Rozpakowano {idx}/{len(missing_filenames)} obrazow")


def download_and_prepare_celeba(data_dir, max_samples=None):
    download_celeba_landmarks(data_dir)
    records = read_celeba_landmark_records(data_dir, max_samples=max_samples)
    extract_celeba_images(data_dir, records)


def load_celeba_dataset(data_dir, max_samples=None):
    data_dir = Path(data_dir)
    if not (data_dir / CELEBA_LANDMARKS_FILE).exists():
        download_celeba_landmarks(data_dir)

    records = read_celeba_landmark_records(data_dir, max_samples=max_samples)
    extract_celeba_images(data_dir, records)

    images, labels = [], []
    images_dir = data_dir / CELEBA_IMAGES_DIR
    for filename, eye_coordinates in records:
        image_path = images_dir / filename
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Nie mozna wczytac obrazu: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        images.append(image_rgb.astype("float32"))
        labels.append(eye_coordinates)

    images = np.stack(images)
    labels = np.stack(labels)
    print(f"Liczba kolorowych zdjec CelebA: {len(images)}")
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

    model = keras.Model(inputs, outputs, name="celeba_eye_detector")
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
    plt.savefig(str(output_path))
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
        ax.scatter(
            [left_eye[0], right_eye[0]],
            [left_eye[1], right_eye[1]],
            c=["lime", "red"],
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    if not output_path.exists():
        print(
            f"Blad: plik {output_path} nie zostal zapisany. "
            "Sprawdz uprawnienia albo dostepna pamiec."
        )
    print(f"Zapisano podglad datasetu: {output_path}")


def eye_crop_bounds(image_shape, coordinates, padding=0.45):
    height, width = image_shape[:2]
    points = normalized_coordinates_to_pixels_for_shape(coordinates, width, height)
    left_eye, right_eye = points[0], points[1]

    min_x = min(left_eye[0], right_eye[0])
    max_x = max(left_eye[0], right_eye[0])
    min_y = min(left_eye[1], right_eye[1])
    max_y = max(left_eye[1], right_eye[1])
    eye_distance = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)

    x_padding = eye_distance * padding
    y_padding = max(eye_distance * padding * 0.6, height * 0.04)

    x1 = max(0, int(np.floor(min_x - x_padding)))
    y1 = max(0, int(np.floor(min_y - y_padding)))
    x2 = min(width, int(np.ceil(max_x + x_padding)))
    y2 = min(height, int(np.ceil(max_y + y_padding)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Nie mozna wyznaczyc poprawnego prostokata oczu.")
    return x1, y1, x2, y2


def normalized_coordinates_to_pixels_for_shape(coordinates, width, height):
    coordinates = np.asarray(coordinates, dtype="float32").reshape(2, 2)
    scale = np.array([width, height], dtype="float32")
    return coordinates * scale


def crop_eye_region(image, coordinates, padding=0.45):
    x1, y1, x2, y2 = eye_crop_bounds(image.shape, coordinates, padding=padding)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def save_eye_crop_grid(crops, output_path, columns=4):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not crops:
        return

    columns = min(columns, len(crops))
    rows = int(np.ceil(len(crops) / columns))
    plt.figure(figsize=(columns * 3, rows * 2))

    for idx, crop in enumerate(crops):
        ax = plt.subplot(rows, columns, idx + 1)
        ax.imshow(crop.astype("uint8"))
        ax.set_title(f"crop {idx + 1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    print(f"Zapisano siatke wycinkow oczu: {output_path}")


def save_dataset_eye_crops(
    data_dir,
    output_dir=DEFAULT_EYE_CROP_OUTPUT_DIR,
    count=32,
    padding=0.45,
    seed=None,
    max_samples=None,
):
    data_dir = Path(data_dir)
    if not (data_dir / CELEBA_LANDMARKS_FILE).exists():
        download_celeba_landmarks(data_dir)

    records = read_celeba_landmark_records(data_dir, max_samples=max_samples)
    if not records:
        raise ValueError("Dataset nie zawiera obrazow do wycinania oczu.")

    extract_celeba_images(data_dir, records)
    images_dir = data_dir / CELEBA_IMAGES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if count is None or count <= 0:
        count = len(records)
    count = min(count, len(records))

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=count, replace=False)
    preview_crops = []

    for local_idx, dataset_idx in enumerate(indices):
        filename, coordinates = records[dataset_idx]
        image_path = images_dir / filename
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Nie mozna wczytac obrazu: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop, bounds = crop_eye_region(image_rgb, coordinates, padding=padding)
        output_path = output_dir / f"eye_crop_{local_idx + 1:04d}_idx_{dataset_idx}.png"
        crop_bgr = cv2.cvtColor(crop.astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), crop_bgr)
        preview_crops.append(crop)
        print(f"Zapisano wycinek oczu: {output_path}; bounds={bounds}")

    save_eye_crop_grid(
        preview_crops[: min(16, len(preview_crops))],
        output_dir / "eye_crops_grid.png",
    )
    print(f"Wylosowane indeksy do wycinkow oczu: {indices.tolist()}")


def train_model(data_dir, model_path, epochs, max_samples=None):
    images, labels = load_celeba_dataset(data_dir, max_samples=max_samples)
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


def normalized_coordinates_to_pixels(coordinates):
    coordinates = np.asarray(coordinates, dtype="float32")
    return coordinates.reshape(2, 2) * IMG_SIZE


def save_random_dataset_prediction(image, prediction, target, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_points = normalized_coordinates_to_pixels(prediction)
    target_points = normalized_coordinates_to_pixels(target)

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


def save_random_dataset_grid(images, predictions, targets, indices, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = len(images)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    plt.figure(figsize=(columns * 3, rows * 3))

    for plot_idx, image in enumerate(images):
        ax = plt.subplot(rows, columns, plot_idx + 1)
        predicted_points = normalized_coordinates_to_pixels(predictions[plot_idx])
        target_points = normalized_coordinates_to_pixels(targets[plot_idx])

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
    print(f"Zapisano siatke losowych predykcji: {output_path}")


def predict_random_dataset_eyes(
    model,
    data_dir,
    count=8,
    output_dir=RANDOM_DATASET_OUTPUT_DIR,
    seed=None,
    max_samples=None,
):
    """Losuje osoby z kolorowego CelebA i zapisuje detekcje oczu modelem."""
    if count <= 0:
        raise ValueError("RANDOM_COUNT musi byc wieksze od 0.")

    images, labels = load_celeba_dataset(data_dir, max_samples=max_samples)
    if len(images) == 0:
        raise ValueError("Dataset nie zawiera zadnych obrazow do losowania.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(count, len(images))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=count, replace=False)
    selected_images = images[indices]
    selected_labels = labels[indices]

    predictions = model.predict(selected_images, batch_size=min(32, count), verbose=0)
    predictions = np.clip(predictions, 0.0, 1.0)

    for local_idx, dataset_idx in enumerate(indices):
        output_path = output_dir / f"random_{local_idx + 1:02d}_idx_{dataset_idx}.png"
        save_random_dataset_prediction(
            selected_images[local_idx],
            predictions[local_idx],
            selected_labels[local_idx],
            output_path,
            title=f"CelebA index: {dataset_idx}",
        )
        print(f"Zapisano losowa predykcje: {output_path}")

    save_random_dataset_grid(
        selected_images,
        predictions,
        selected_labels,
        indices,
        output_dir / "random_predictions_grid.png",
    )
    print(f"Wylosowane indeksy z datasetu: {indices.tolist()}")


def main():
    max_samples = normalize_max_samples(MAX_SAMPLES)

    needs_dataset = not SKIP_TRAIN or RUN_RANDOM_DATASET or SAVE_EYE_CROPS or DOWNLOAD_DATA
    if needs_dataset:
        landmarks_path = Path(DATA_DIR) / CELEBA_LANDMARKS_FILE
        if DOWNLOAD_DATA or not landmarks_path.exists():
            download_and_prepare_celeba(DATA_DIR, max_samples=max_samples)

    if SAVE_EYE_CROPS:
        save_dataset_eye_crops(
            DATA_DIR,
            output_dir=EYE_CROP_OUTPUT_DIR,
            count=EYE_CROP_COUNT,
            padding=EYE_CROP_PADDING,
            seed=RANDOM_DATASET_SEED,
            max_samples=max_samples,
        )

    model = None
    if SKIP_TRAIN:
        if IMAGE_PATHS or RUN_RANDOM_DATASET:
            model = keras.models.load_model(MODEL_PATH)
    else:
        model = train_model(
            DATA_DIR,
            MODEL_PATH,
            EPOCHS,
            max_samples=max_samples,
        )

    if IMAGE_PATHS:
        predict_eyes(
            model,
            IMAGE_PATHS,
            PREDICTION_OUTPUT_DIR,
            detect_face=not NO_FACE_DETECT,
        )

    if RUN_RANDOM_DATASET:
        predict_random_dataset_eyes(
            model,
            DATA_DIR,
            count=RANDOM_COUNT,
            output_dir=RANDOM_OUTPUT_DIR,
            seed=RANDOM_DATASET_SEED,
            max_samples=max_samples,
        )


if __name__ == "__main__":
    main()
