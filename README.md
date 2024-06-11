# Analiza zdjęć lotniczych

## Wstęp
Poniższa instrukcja opisuje:
1. Przygotowanie środowiska i uruchamianie notebooków
3. Funkcje i przeznaczenie notebooków
2. Źródła i krótką charakterystykę danych

## Przygotowanie środowiska i uruchamianie notebooków
1. Utwórz środowisko wirtualne (conda, venv, etc.), np.: `python -m venv aerial_images`
2. Aktywuj środowisko wirtualne (w zależności od systemu operacyjnego):
    - Windows: `.\aerial_images\Scripts\activate`
    - Linux: `source aerial_images/bin/activate`
3. Pobierz i zainstaluj bibliotekę `torch` zgodnie z instrukcją na stronie [pytorch.org](https://pytorch.org/get-started/locally/)
4. Zainstaluj pozostałe wymagane biblioteki: `pip install -r requirements.txt`

## Kod i notebooki
Ogólnie repozytorium jest zorganizowane w taki sposób, że katalog `data` zawiera tylko surowe dane, katalog `src` kod i funkcje pomocnicze, a katalog `notebooks` notebooki z kodem. Dla lepszej przejrzystości notebooki nie są przechowywane w root directory, ale przed uruchomieniem należy je do niego przenieść. Poszczególne sub-katalogi zawierają to na co wskazuje ich nazwa, poniżej szczegółowy opis:

* `data` - katalog zawierający tylko dane, katalogi zakończone na `Patches` zawierają zbiory danych podzielone na *patche*
* `src` - katalog zawierający kod źródłowy, w tym funkcje pomocnicze
    * `callbacks` - katalog zawierający funkcje callbacków, pomocniczne przy zarządzaniu treningiem modelu
    * `datasets` - katalog zawierający klasy datasetów (dziedziczace po `torch.utils.data.Dataset`)
        * `utils` - katalog zawierający funkcje pomocnicze dla datasetów, głównie do konwersji masek na etykiety typu {0, 1, 2, ...} oraz transormacje (`torchvision.transforms`)
    * `evaluation` - katalog zawierający funkcje pomocnicze do ewaluacji modelu
    * `models` - katalog zawierający pomocniczą klasę baseline modelu (dziedziczace po `torch.nn.Module`)
    * `utils.py` - ogólne funkcje pomocnicze
* `notebooks` - katalog zawierający notebooki z kodem
    * `datasets_to_patches` - notebooki demonstrujące podział zbiorów danych na *patche*
    * `masks_conversion` - notebooki demonstrujące konwersję masek na etykiety typu {0, 1, 2, ...}
    * `no_finetune` - notebooki demonstrujące próby użycia modeli bez treningu do segmentacji obrazów lotniczych (baseline, wagi wytrenowane na ImageNet nie przenoszą się na nowy zbiór danych)
    * `sanity_checks` - notebooki demonstrujące sanity checks dla datasetów
    * `with_finetune` - notebooki demonstrujące faktyczny trening modeli na nowych zbiorach danych

## Dane

### Pobieranie
* `INRIA`: [link](https://project.inria.fr/aerialimagelabeling/)
* `Dubai`: [link](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data)
* `Aerial Drone`: [link](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data)
* `UAVid`: [link](https://www.kaggle.com/code/alexalex02/semantic-segmentation-of-aerial-images)

### Ilość klas w poszczególnych zbiorach
* `INRIA`: 2 (binary - *building* i *non-building*) [source](https://project.inria.fr/aerialimagelabeling/)
* `Dubai`: 6  [source](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data)
    - Building: #3C1098
    - Land (unpaved area): #8429F6
    - Road: #6EC1E4
    - Vegetation: #FEDD3A
    - Water: #E2A929
    - Unlabeled: #9B9B9B     
* `Aerial Drone`: 20 (tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle) [source](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data)   

> [!WARNING]  
> Tak naprawdę, wygląda na to, że są 23 klasy.

* `UAVid`: 8 [source](https://www.kaggle.com/code/alexalex02/semantic-segmentation-of-aerial-images)
    1. *building*: living houses, garages, skyscrapers, security booths, and buildings under construction.
    2. *road*: road or bridge surface that cars can run on legally. Parking lots are not included.
    3. *tree*: tall trees that have canopies and main trunks.
    4. *low vegetation*: grass, bushes and shrubs.
    5. *static car*: cars that are not moving, including static buses, trucks, automobiles, and tractors. Bicycles and motorcycles are not included.
    6. *moving car*: cars that are moving, including moving buses, trucks, automobiles, and tractors. Bicycles and motorcycles are not included.
    7. *human*: pedestrians, bikers, and all other humans occupied by different activities.
    8. *clutter*: all objects not belonging to any of the classes above.

### Ilość obrazków w poszczególnych zbiorach
* `INRIA`
    - `train`: 180 (dane są etykiety, trzeba ręcznie podzielić na train i val)
    - `test`: 144 (brak etykiet)
* `Dubai`
    - `train`: 72 (dane są etykiety, trzeba ręcznie podzielić na train, val i test)
* `Aerial Drone`
    - `train`: 400 (dane są etykiety, trzeba ręcznie podzielić na train, val i test)
* `UAVid`
    - `train`: 200
    - `val`: 70
    - `test`: 10
