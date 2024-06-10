# Aerial Segmentation
Repo for TWM (Machine Vision Techniques) project @ WUT 24L semester

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## TODO
- dokończenie sprawka
- gałąź report
- [ ] przygotowanie wstępne ('pocięcie' obrazków) w większych datasetach
    - [ ] UAVid
        - [ ] debug
    - [x] INRIA
- [ ] RETRO
    - kanały, konwersja, etc. w PIL
    - wymiary obrazków, przemyslenie tego na etapie wejścia
    - rastervision
    - rasterio
    - funkcja straty 'na kartce' (upewnić się co i jak dokładnie działa, f straty 
    zawsze min. i tak dalej)
    - Compose zamiast for transform in self.transforms
    - ... 
- [ ] trening na jednym datasecie + test na jednym datasecie
- [ ] użycie `Crop` lub `Pad` zamiast `Resize` - może będą lepsze wyniki?
- [ ] upewnienie się, że maski nie zostały (za bardzo) zaburzone - np. bilinear i progowanie niskim progiem (będzie mniejszy latent w UNet)
- [ ] literatura
    - [ ] jakie jest SOTA w tym problemie? (top 5)
    - [ ] dobre modele z Kaggle (po jednym dla każdego datasetu)
    - [ ] inne rzeczy warte uwagi
- [ ] inne
    - [x] włączenie torch-lightning, żeby mieć logi i przebieg eksperymentów
    - [ ] dodanie wymiarów tensorów w annotacjach / *type hints*
    - [ ] (opcjonalnie) publikacja na Kaggle
- [x] *deep dive* UNet
- [x] rescale (downsample) -> INRIA i może inne
- [x] stworzenie funkcji ewaluacyjnej
- [x] ile klas w UAVid?
- [x] czy lepiej robić segmentację na podstawie jednego kanału czy trzech?
- [x] zapoznanie z libką *segmentation_models.pytorch*
- [x] zapoznanie z datasetem *INRIA*
- [x] zapoznanie z datasetem *Aerial Image Segmentation from Online Maps*
- [x] zapoznanie z datasetem *UAVid Semantic Segmentation Dataset*
- [x] zapoznanie z datasetem *Aerial Semantic Segmentation Drone Dataset*
- [x] doinstalować torcha z CUDA (skill issue xd)
- [x] dokończenie prezentacji
- [x] problem konwersji danych RGB -> maska
    - [x] tak samo w UAVid - dane są zakodowane, tak, żeby dało się je wyświetlić, a nie do modelu
    - [x] rozwiązać problem ze sposobem w jaki jest zakodowane gt w Dubai (kolorowe obrazki zamiast po prostu [0...5]) - tu jest chyba jakiś bug, bo w notebooku `example-masks-conversion` wychodzi inaczej niż w `dubai-no-finetune`
    - [x] czy w AerialDrone używamy tylko maski z jednym kanałem czy kolorów z wieloma?
    - [x] w jaki sposób, w AerialDrone, jest oznaczane to co trzeba przewidzieć (RGB classes czy to drugie)?
    - [x] w AerialDrone, jak działa przetworzenie maski na tensor / PIL.Image (tzn. czy nie ma np. jakiegoś rescale, itd.)?

## Intro
Proponuję pójść w stronę przeglądu / ensemble różnych modeli i/lub datasetów, porównać, itd.

## Dane

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

### Wymiary wejściowych obrazków bez Resize
* `INRIA` - potrzebne *Resize*
```python
torch.Size([1, 3, 5024, 5024])
torch.Size([1, 1, 5024, 5024])
```

* `Dubai` - nie ma potrzeby *Resize*
```python
torch.Size([1, 3, 544, 512])
torch.Size([1, 1, 544, 512])
```
* `UAVid` - trzeba zrobić *Resize*, różne wymiary obrazków
```python
torch.Size([1, 3, 2176, 4096])
torch.Size([1, 3, 2176, 4096])
```
albo 
```python
torch.Size([1, 3, 2176, 3840])
torch.Size([1, 3, 2176, 3840])
```
* `Aerial Drone` - potrzebne *Resize*, możliwe, że są różne rozmiary obrazków w zbiorze
```python
torch.Size([1, 3, 4000, 6016])
torch.Size([1, 1, 4000, 6016])
```

## Modele

### Bez finetune
*sprawdzenie tylko czy się odpalają, tzn. model prawidłowo przetwarza dane, wyniki mogą być (is zazwyczaj są) bardzo słabe na początku*
| Model      | INRIA | UAVid | Dubai | AerialDrone |  
| ----------- | ----------- | ----------- | ----------- | ----------- |  
| UNet      | :heavy_check_mark:       | :heavy_check_mark:   | :heavy_check_mark:   |  :heavy_check_mark:   | 
| UNet++   | :heavy_check_mark:        | :heavy_check_mark:      | :heavy_check_mark:      | :heavy_check_mark:   | 
| DeepLabV3   | :heavy_check_mark:        | :heavy_check_mark:      | :heavy_check_mark:      | :heavy_check_mark:   | 
| DeepLabV3+   | :heavy_check_mark:        | :heavy_check_mark:      | :heavy_check_mark:      | :heavy_check_mark:   | 

## Wyniki

### IoU

### Bez finetune
*wyniki bez finetune bardzo słabe, więc nie będą nawet wpisywane, TBA już przy normalnym treningu*
| Model      | INRIA | UAVid | Dubai | AerialDrone |  
| ----------- | ----------- | ----------- | ----------- | ----------- |  
| UNet      | 0.0106      | TBA   | TBA   |  0.0043   | 
| UNet++   | 0.0146        | TBA      | TBA      | TBA   | 
| DeepLabV3   | TBA        | TBA      | TBA      | TBA   | 
| DeepLabV3+   | TBA        | TBA      | TBA      | TBA   | 

### Acc

### Bez finetune
*wyniki bez finetune bardzo słabe, więc nie będą nawet wpisywane, TBA już przy normalnym treningu*
| Model      | INRIA | UAVid | Dubai | AerialDrone |  
| ----------- | ----------- | ----------- | ----------- | ----------- |  
| UNet      | 0.8448      | TBA   | TBA   |  0.9138   | 
| UNet++   | 0.9170        | TBA      | TBA      | TBA   | 
| DeepLabV3   | TBA        | TBA      | TBA      | TBA   | 
| DeepLabV3+   | TBA        | TBA      | TBA      | TBA   | 

### Misc
* `UNet` - `INRIA` -  `no finetune`  
{'iou': 0.010615132, 'f1': 0.020689072, 'accuracy': 0.8447621, 'recall': 0.0944909}
* `UNet++` - `INRIA` -  `no finetune`  
{'iou': tensor(0.0146), 'f1': tensor(0.0289), 'accuracy': tensor(0.9170), 'recall': tensor(0.0147)}
* `Unet` - `Aerial Dron` - `no finetune`
```
Mean metrics
iou 0.004345709
f1 0.008359963
accuracy 0.91377044
recall 0.008359963
```

## Problemy
* model musi przyjmować dowolny (albo z dużego zbioru) rozmiar obrazka, a nie stały, bo datasety mają różne rozmiary obrazków, a nawet mogą być różne w ramach datasetu
* w większości zbiorów danych, `groundtruth` jest zakodowane w postaci obrazków RGB, gdzie każdy kolor odpowiada innej klasie, trzeba je konwertować na tensor z etykietami, bo taki zwracają modele [related gh issue](https://github.com/qubvel/segmentation_models/issues/137)
    - `UAVid` - maska ma 3 kanały, zapisane jako arbitralne wartości RGB
    - `Aerial Drone` - maska ma 1 kanał, który potem jest jakoś przepisywany przy wyświetlaniu
    - `Dubai` - maska ma 1 kanał, który potem jest jakoś przepisywany przy wyświetlaniu

## Prezka
* opisane w [readme](./docs/README.md)

## Uwagi
* ze względu na architekturę UNet, której używamy, ważne jest, żeby wymiary danych wejściowych były wielokrotnością 32 (zob. [ta funkcja](/src/datasets/utils/ResizeToDivisibleBy32.py))
![unet arch](assets/unet-arch.png)

> [!WARNING]  
> W przypadku `Resize` trzeba uważać na wybrany interpolant, bo może zaburzyć maskę, np. `BILINEAR` zaburza maskę, bo robi wartości zmiennoprzecinkowe, które potem trzeba progować, a `NEAREST` i `NEAREST_EXACT` (chyba) nie zaburzają.

* problem z `runtimeerror: element 0 of tensors does not require grad and does not have a grad_fn` - trzeba zwrócić uwagę na to, żeby nie próbować robić `backward` na tensorach, które nie wymagają `grad` (np. maski, które są `int`), często po prostu dodanie `.float()` (konwersja do float) lub `loss.required_grad=True` ('wymuszenie' gradientu) rozwiązuje problem, zobacz [SO post](https://stackoverflow.com/questions/61808965/pytorch-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-ha)

* problem z `smp.losses.JaccardLoss` jest związany z [github discussion](https://github.com/qubvel/segmentation_models.pytorch/discussions/454) i nadal nie został rozwiązany

> [!WARNING]  
> Chyba lepiej nie używać `smp.losses.JaccardLoss`.  

[Ten docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html) i [ten post SO](https://stackoverflow.com/questions/66543659/one-hot-encoding-in-pytorch) pomagają rozwiązać problem, choć dziwne, że akurat `torch.LongTensor` jest wymagany w `one_hot`

## Materiały
* repozytoria
    * [praca z INRIA dataset](https://github.com/margokhokhlova/aerial_segmentation)
    * [praca z Aerial Image Segmentation from Online Maps](https://github.com/alpemek/aerial-segmentation/tree/master)
* datasety
    * [UAVid Semantic Segmentation Dataset](https://www.kaggle.com/datasets/titan15555/uavid-semantic-segmentation-dataset)
    * [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)
    * [Semantic segmentation of aerial imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
* rozwiązania z kaggle
    * [rozwiązanie UAVid Semantic Segmentation Dataset](https://www.kaggle.com/code/alexalex02/semantic-segmentation-of-aerial-images/notebook)
* libki
    * [potężna libka z gotowymi modelami do segmentacji](https://github.com/qubvel/segmentation_models.pytorch)
* papiery
    * [Learning Aerial Image Segmentation from Online Maps](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Papers/Learning%20Aerial%20Image.pdf)
    * [Drone Depth and Obstacle Segmentation Dataset](https://arxiv.org/pdf/2312.12494.pdf)
    * [Varied Drone Dataset for Semantic Segmentation](https://arxiv.org/pdf/2305.13608.pdf)
