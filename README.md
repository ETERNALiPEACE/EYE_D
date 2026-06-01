# Eye detector - BioID + DenseNet121

Projekt zawiera kod podobny organizacyjnie do podanego przykladu Keras, ale
zamiast klasyfikacji wideo trenuje model do wykrywania oczu ludzi na zdjeciach.
Model przewiduje cztery znormalizowane wspolrzedne:

```text
left_eye_x, left_eye_y, right_eye_x, right_eye_y
```

Jako baza danych uzywany jest publiczny **BioID Face Database**:

- 1521 zdjec twarzy ludzi,
- rozdzielczosc oryginalna 384x286,
- recznie oznaczone pozycje lewego i prawego oka w plikach `.eye`,
- zrodlo: https://ftp.uni-erlangen.de/pub/facedb/readme.html

## Instalacja

```bash
pip install -r requirements.txt
```

## Trening modelu przez 50 epok

```bash
python eye_detection_bioid.py --download --epochs 50
```

Skrypt:

1. pobierze `BioID-FaceDatabase-V1.2.zip`,
2. pobierze `BioID-FD-Eyepos-V1.2.zip`,
3. wczyta obrazy i adnotacje oczu,
4. wytrenuje model DenseNet121 z wlasna glowa regresyjna,
5. zapisze model do `models/bioid_eye_detector.keras`,
6. zapisze wykres treningu do `outputs/training_history.png`.

## Predykcja na podeslanych zdjeciach

Po treningu mozna zaznaczyc oczy na wlasnych zdjeciach:

```bash
python eye_detection_bioid.py \
  --skip-train \
  --model-path models/bioid_eye_detector.keras \
  --images zdjecie1.jpg zdjecie2.jpg
```

Wyniki zostana zapisane w `outputs/predictions/`.

## Uwaga

BioID zawiera pojedyncze, frontalne twarze. Model bedzie dzialal najlepiej na
podobnych zdjeciach: jedna osoba, widoczna twarz, oczy bez duzych zasloniec.
Jesli zdjecie zawiera wiele osob, najpierw warto dodac detektor twarzy i uruchamiac
ten model osobno na kazdym wycieciu twarzy.
