# Eye detector - Facial Keypoints + DenseNet121

Projekt zawiera kod podobny organizacyjnie do podanego przykladu Keras, ale
zamiast klasyfikacji wideo trenuje model do wykrywania oczu ludzi na zdjeciach.
Model przewiduje cztery znormalizowane wspolrzedne:

```text
left_eye_center_x, left_eye_center_y, right_eye_center_x, right_eye_center_y
```

Jako baza danych uzywany jest publiczny mirror **Kaggle Facial Keypoints
Detection**:

- 7049 zdjec twarzy ludzi,
- obrazy 96x96 zapisane w `training.csv` jako piksele,
- recznie oznaczone punkty twarzy, w tym srodki lewego i prawego oka,
- mirror: https://github.com/ruchawaghulde/Facial-Keypoints-Detection
- opis konkursu: https://www.kaggle.com/c/facial-keypoints-detection/data

## Instalacja

```bash
pip install -r requirements.txt
```

## Trening modelu przez 50 epok

```bash
python eye_detection_facial_keypoints.py --download --epochs 50
```

Skrypt:

1. pobierze `training.zip` z publicznego mirroru,
2. rozpakowuje `training.csv`,
3. wczyta obrazy twarzy i adnotacje oczu,
4. wytrenuje model DenseNet121 z wlasna glowa regresyjna,
5. zapisze model do `models/facial_keypoints_eye_detector.keras`,
6. zapisze wykres treningu do `outputs/training_history.png`,
7. zapisze podglad rzeczywistych zdjec i punktow oczu do
   `outputs/dataset_preview.png`.

Do szybkiego sprawdzenia kodu bez pelnego treningu mozna ograniczyc liczbe probek:

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

## Losowe predykcje na zdjeciach z bazy

Jesli model jest juz wytrenowany, osobny skrypt moze wylosowac przykladowe
twarze bezposrednio z datasetu i porownac predykcje modelu z etykietami z bazy:

```bash
python random_dataset_eye_predictions.py --count 8 --seed 42
```

Skrypt jest powiazany z glownym programem przez import funkcji z
`eye_detection_facial_keypoints.py`, ale pozostaje odizolowany: nie uruchamia
treningu i zapisuje swoje wyniki osobno w
`outputs/random_dataset_predictions/`.

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
  --model-path models/facial_keypoints_eye_detector.keras \
  --images zdjecie1.jpg zdjecie2.jpg
```

Wyniki zostana zapisane w `outputs/predictions/`. Przy predykcji skrypt najpierw
probuje znalezc najwieksza twarz detektorem OpenCV, a dopiero potem wyznacza
pozycje oczu na wycinku twarzy. Jesli chcesz pominac detekcje twarzy:

```bash
python eye_detection_facial_keypoints.py \
  --skip-train \
  --model-path models/facial_keypoints_eye_detector.keras \
  --images zdjecie1.jpg \
  --no-face-detect
```

## Uwaga

Ten dataset zawiera glownie wykadrowane twarze. Model bedzie dzialal najlepiej,
gdy twarz jest widoczna i niezbyt mocno obrocona. Dla zdjec z wieloma osobami
obecny kod wybiera najwieksza wykryta twarz.
