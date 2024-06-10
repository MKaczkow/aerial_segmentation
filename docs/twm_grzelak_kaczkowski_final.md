# TWM PROJEKT SPRAWOZDANIE KOŃCOWE
*Mateusz Grzelak* 305780
*Maciej Kaczkowski* 300660  

# Wstęp

# Metody

## Narzędzia
* pytorch
* pytorch-lightning
* segmentation-models-pytorch
* torchmetrics

## INRIA

### Ustawienia ogólne
* batch_size = 16
* lr = 0.001
* resnet18
* imagenet init

`smp.losses.` lub `smp.` ...  
* MCCLoss, DeepLabV3
* MCCLoss, UNet
* SoftBCEWithLogitsLoss, UNET
* DiceLoss, UNET

## Dubai

### Ustawienia ogólne
* batch_size = 1
* lr = 0.001
* resnet18
* imagenet init

`smp.losses.` lub `smp.` ...  
* JaccardLoss, UNet

# Wyniki

* tabelka z podsumowaniem INRIA
    * przykładowe obrazki

* tabelka z podsumowaniem Dubai
    * przykładowe obrazki

## INRIA
* MCCLoss, DeepLabV3 (wykres)
* MCCLoss, UNet (wykres)
* SoftBCEWithLogitsLoss, UNET (wykres)
* DiceLoss, UNET (wykres)

* porównanie końcowych lossów walidacyjnych
* porównanie końcowych IoU

## Dubai
* JaccardLoss, UNet (wykres)

# Dyskusja

## Trudności
* duże dane [5000, 5000], itp.
* .tif
* długi czas treningu
* maski

## Co pomogło
* podział na małe kawałki [500, 500]
* MCCLoss (jeżeli będzie lepsza)

## Inne pomysły
* rasterio, rastervision
* geotorch
