# Eye detector - CelebA + DenseNet121

Projekt trenuje model Keras/DenseNet121 do wykrywania oczu ludzi na kolorowych
zdjeciach twarzy z datasetu **CelebA**.

Model przewiduje:

```text
left_eye_x, left_eye_y, right_eye_x, right_eye_y
```

Dataset:

- kolorowe, wyrownane twarze JPG 218x178,
- landmarki twarzy w `list_landmarks_align_celeba.txt`,
- do treningu uzywane sa punkty lewego i prawego oka,
- mirror: https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba/

Zdjecia sa pobierane jako ZIP:

```text
data/celeba/raw/img_align_celeba.zip
```

a potem rozpakowywane do:

```text
data/celeba/img_align_celeba/
```

## Instalacja

```bash
pip install -r requirements.txt
```

## Konfiguracja

W Colab/Jupyter najprosciej zmieniac stale na gorze
`eye_detection_facial_keypoints.py`:

```python
DOWNLOAD_DATA = False
SKIP_TRAIN = False
MAX_SAMPLES = DEFAULT_MAX_SAMPLES

RUN_RANDOM_DATASET = False
SAVE_EYE_CROPS = False
IMAGE_PATHS = []
```

## Trening

Domyslnie skrypt trenuje model przez 50 epok na 2000 obrazach:

```bash
python eye_detection_facial_keypoints.py
```

Model zostanie zapisany do:

```text
models/celeba_eye_detector.keras
```

Szybki test wymaga zmiany stalych:

```python
EPOCHS = 1
MAX_SAMPLES = 128
```

## Wycinanie prostokata obejmujacego oba oczy

Aby tylko wyciac prostokaty oczu bez trenowania modelu, ustaw:

```python
SKIP_TRAIN = True
SAVE_EYE_CROPS = True
EYE_CROP_COUNT = 32
RANDOM_DATASET_SEED = 42
```

Wyniki:

```text
outputs/eye_crops/
outputs/eye_crops/eye_crops_grid.png
```

W Colab:

```python
from IPython.display import Image, display

display(Image(filename="/content/outputs/eye_crops/eye_crops_grid.png"))
```

## Losowe predykcje z bazy

Po wytrenowaniu modelu ustaw:

```python
SKIP_TRAIN = True
RUN_RANDOM_DATASET = True
RANDOM_COUNT = 8
RANDOM_DATASET_SEED = 42
```

Wynik:

```text
outputs/random_dataset_predictions/random_predictions_grid.png
```

## Predykcja na wlasnych zdjeciach

Po wytrenowaniu modelu ustaw:

```python
SKIP_TRAIN = True
IMAGE_PATHS = ["zdjecie1.jpg", "zdjecie2.jpg"]
```

Wyniki zapisza sie w:

```text
outputs/predictions/
```

## Podglad datasetu

Po treningu skrypt zapisuje:

```text
outputs/dataset_preview.png
```

W Colab:

```python
from IPython.display import Image, display

display(Image(filename="/content/outputs/dataset_preview.png"))
```
