# DessiAdam
An Adam version of DessiLBI optimizer put forward in  paper Yanwei Fu, Chen Liu, Donghao Li, Xinwei Sun, Jinshan Zeng, and Yuan Yao. Dessilbi: Exploring structural sparsity of deep networks via differential inclusion paths. In International Conference on Machine Learning, pages 3315–3326. PMLR, 2020.

To draw a loss landscape of VGG-A network (with or without BatchNorm), please run ```VGG_Loss_Landscape.py```. <br>
To train a VGG-A network using DessiLBI or DessiAdam, please run ```VGG_trainer.py```. Please edit the optimizer you like to employ before running it.
To train a LeNet-5 network using DessiLBI or DessiAdam, please run ```LeNet_trainer.py```.
