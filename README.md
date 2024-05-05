# Aerial Segmentation
Repo for TWM (Machine Vision Techniques) project @ WUT 24L semester

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## TODO
- [ ] zapoznanie z libką *segmentation_models.pytorch*
- [ ] czy lepiej robić segmentację na podstawie jednego kanału czy trzech?
- [ ] w jaki sposób, w AerialDrone, jest oznaczane to co trzeba przewidzieć (RGB classes czy to drugie)?
- [ ] w AerialDrone, jak działa przetworzenie maski na tensor / PIL.Image (tzn. czy nie ma np. jakiegoś rescale, itd.)?
- [ ] literatura
    - [ ] jakie jest SOTA w tym problemie? (top 5)
    - [ ] dobre modele z Kaggle (po jednym dla każdego datasetu)
    - [ ] inne rzeczy warte uwagi
- [ ] (opcjonalnie) publikacja na Kaggle
- [x] zapoznanie z datasetem *INRIA*
- [x] zapoznanie z datasetem *Aerial Image Segmentation from Online Maps*
- [x] zapoznanie z datasetem *UAVid Semantic Segmentation Dataset*
- [x] zapoznanie z datasetem *Aerial Semantic Segmentation Drone Dataset*
- [x] doinstalować torcha z CUDA (skill issue xd)
- [x] dokończenie prezentacji

## Problemy
* model musi przyjmować dowolny (albo z dużego zbioru) rozmiar obrazka, a nie stały, bo datasety mają różne rozmiary obrazków, a nawet mogą być różne w ramach datasetu

## Prezka
* opisane w [readme](./docs/README.md)

## Intro
Dobra, żeby zacząć już coś robić w projekcie, proponuję pójść w stronę przeglądu / ensemble różnych modeli i/lub datasetów, porównać, itd.
* 'na zewnątrz' sprzedamy to jako właśnie taki przegląd, porównanie np. czy modele dobre dla zdj z satelitów, są dobre też dla zdj z dronów, biorąc też pod uwagę, np. koszt treningu / inferencji
* 'do wewnątrz', czyli dla nas, to będzie po prostu zapoznanie się z aktualnym stanem dziedziny, nie będziemy wymyślać nowych rzeczy, jak na PBAD xd 

## Uwagi
* ze względu na architekturę UNet, której używamy, ważne jest, żeby wymiary danych wejściowych były wielokrotnością 32 (zob. [ta funkcja](/src/datasets/utils/ResizeToDivisibleBy32.py))
![unet arch](assets/unet-arch.png)

## Materiały
Wstępnie zebrałem trochę materiałów, proponuję od nich zacząć zapoznawanie się z rzeczami. Kolejne etapy projektu możemy spokojnie zrobić wcześniej niż termin i potem tylko oddawać

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
