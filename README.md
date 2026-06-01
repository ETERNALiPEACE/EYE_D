# Eye detector - CelebA + DenseNet121

Projekt zawiera kod podobny organizacyjnie do podanego przykladu Keras, ale
zamiast klasyfikacji wideo trenuje model do wykrywania oczu ludzi na kolorowych
zdjeciach twarzy. Model przewiduje cztery znormalizowane wspolrzedne:

```text
left_eye_x, left_eye_y, right_eye_x, right_eye_y
```

Jako baza danych uzywany jest publiczny mirror **CelebA**:

- ponad 200 tys. kolorowych, wyrownanych zdjec twarzy,
- obrazy JPG 218x178,
- plik `list_landmarks_align_celeba.txt` z 5 landmarkami twarzy,
- do treningu wykorzystywane sa landmarki lewego i prawego oka,
- mirror: https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba/
- opis datasetu: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Skrypt pobiera zdjecia w archiwum `img_align_celeba.zip`, zapisuje je w
`data/celeba/raw/`, a potem rozpakowuje z ZIP-a obrazy potrzebne dla aktualnego
limitu `--max-samples`.

## Instalacja

```bash
pip install -r requirements.txt
```

## Trening modelu przez 50 epok

Domyslnie skrypt pobiera archiwum ZIP CelebA i wypakowuje 2000 kolorowych zdjec:

```bash
python eye_detection_facial_keypoints.py --download --epochs 50
```

Mozesz ustawic inny limit:

```bash
python eye_detection_facial_keypoints.py --download --epochs 50 --max-samples 5000
```

Ustawienie `--max-samples 0` oznacza probe uzycia wszystkich rekordow CelebA,
czyli wypakowanie wszystkich obrazow z archiwum ZIP.

Skrypt:

1. pobierze `list_landmarks_align_celeba.txt`,
2. pobierze `img_align_celeba.zip`,
3. rozpakowuje z ZIP-a potrzebne zdjecia twarzy i wczyta adnotacje oczu,
4. wytrenuje model DenseNet121 z wlasna glowa regresyjna,
5. zapisze model do `models/celeba_eye_detector.keras`,
6. zapisze wykres treningu do `outputs/training_history.png`,
7. zapisze podglad kolorowych zdjec i punktow oczu do
   `outputs/dataset_preview.png`.

Do szybkiego sprawdzenia kodu bez pelnego treningu:

```bash
python eye_detection_facial_keypoints.py --download --epochs 1 --max-samples 128
```

## Podglad datasetu w Colab/Jupyter

Po uruchomieniu treningu lub szybkiego testu skrypt zapisuje podglad danych do
`outputs/dataset_preview.png`. W notebooku wyswietlisz go tak:

```python
from IPython.display import Image, display

display(Image(filename="outputs/dataset_preview.png"))
```

Jesli uruchamiasz kod w katalogu `/content`, poprawna sciezka bedzie tez:

```python
display(Image(filename="/content/outputs/dataset_preview.png"))
```

## Wycinanie prostokata obejmujacego oba oczy

Glowny program moze wycinac z oryginalnych kolorowych JPG CelebA jeden prostokat,
ktory obejmuje lewe i prawe oko naraz. Do samego wycinania nie trzeba ladowac modelu,
wiec mozna uzyc `--skip-train`:

```bash
python eye_detection_facial_keypoints.py \
  --download \
  --skip-train \
  --save-eye-crops \
  --eye-crop-count 32 \
  --random-seed 42
```

Wycinki zostana zapisane w:

```text
outputs/eye_crops/
```

Najwazniejszy podglad:

```text
outputs/eye_crops/eye_crops_grid.png
```

W Colab/Jupyter wyswietlisz go tak:

```python
from IPython.display import Image, display

display(Image(filename="/content/outputs/eye_crops/eye_crops_grid.png"))
```

Parametr `--eye-crop-padding` reguluje margines wokol oczu, np.:

```bash
python eye_detection_facial_keypoints.py --skip-train --save-eye-crops --eye-crop-padding 0.6
```

Ustawienie `--eye-crop-count 0` zapisze wycinki dla wszystkich wczytanych obrazow
z zakresu `--max-samples`.

## Losowe predykcje na zdjeciach z bazy

Glowny program potrafi wylosowac przykladowe kolorowe twarze bezposrednio z
CelebA i porownac predykcje modelu z etykietami oczu z bazy:

```bash
python eye_detection_facial_keypoints.py \
  --skip-train \
  --model-path models/celeba_eye_detector.keras \
  --random-dataset \
  --random-count 8 \
  --random-seed 42
```

Jesli chcesz po treningu od razu wykonac losowe predykcje, pomin `--skip-train`:

```bash
python eye_detection_facial_keypoints.py --download --epochs 50 --random-dataset
```

Wyniki trafia do:

```text
outputs/random_dataset_predictions/
```

Najwazniejszy plik wynikowy to:

```text
outputs/random_dataset_predictions/random_predictions_grid.png
```

W Colab/Jupyter wyswietlisz go tak:

```python
from IPython.display import Image, display

display(Image(filename="/content/outputs/random_dataset_predictions/random_predictions_grid.png"))
```

Na obrazach:

- kolka oznaczaja predykcje modelu,
- znaki `X` oznaczaja prawdziwe etykiety oczu z datasetu.

## Predykcja na podeslanych zdjeciach

Po treningu mozna zaznaczyc oczy na wlasnych zdjeciach:

```bash
python eye_detection_facial_keypoints.py \
  --skip-train \
  --model-path models/celeba_eye_detector.keras \
  --images zdjecie1.jpg zdjecie2.jpg
```

Wyniki zostana zapisane w `outputs/predictions/`. Przy predykcji skrypt najpierw
probuje znalezc najwieksza twarz detektorem OpenCV, a dopiero potem wyznacza
pozycje oczu na wycinku twarzy. Jesli chcesz pominac detekcje twarzy:

```bash
python eye_detection_facial_keypoints.py \
  --skip-train \
  --model-path models/celeba_eye_detector.keras \
  --images zdjecie1.jpg \
  --no-face-detect
```

## Uwaga

CelebA zawiera glownie wykadrowane twarze. Model bedzie dzialal najlepiej, gdy
twarz jest widoczna i niezbyt mocno obrocona. Dla zdjec z wieloma osobami obecny
kod wybiera najwieksza wykryta twarz.
