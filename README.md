# Neighboring Region Dropout for Hyperspectral Image Classification
The Code for "Neighboring Region Dropout for Hyperspectral Image Classification". [https://ieeexplore.ieee.org/document/8848492]
```
M. E. Paoletti, J. M. Haut, J. Plaza and A. Plaza
Neighboring Region Dropout for Hyperspectral Image Classification.
IEEE Geoscience and Remote Sensing Letters.
DOI: 10.1109/LGRS.2019.2940467
vol. 17, no. 6, pp. 1032-1036, June 2020.
```

![DeepNRD](https://github.com/mhaut/DeepNRD/blob/master/images/featuremap.png)



## Example of use
```
# Without datasets
git clone https://github.com/mhaut/DeepNRD

# With datasets
git clone --recursive https://github.com/mhaut/DeepNRD
cd HSI-datasets
python join_dsets.py
```

### Run code

```
# without data augmentation
python main.py --dataset IP --tr_percent 0.1 --verbose # without data augmentation

# Hyperspectral Image Classification Using Random Occlusion Data Augmentation
python main.py --dataset IP --tr_percent 0.1 --p 0.25 --verbose # with data augmentation
python main.py --dataset IP --tr_percent 0.1 --p 0.50 --verbose # with data augmentation

# Hyperspectral Image Classification Using Simple Dropout
python main.py --dataset IP --tr_percent 0.1 --simple_dropout 0.2 --verbose
python main.py --dataset IP --tr_percent 0.1 --simple_dropout 0.4 --verbose
python main.py --dataset IP --tr_percent 0.1 --simple_dropout 0.8 --verbose

# Hyperspectral Image Classification Using NDP
python main.py --dataset IP --tr_percent 0.1 --dropprob 0.2 --verbose
python main.py --dataset IP --tr_percent 0.1 --dropprob 0.4 --verbose
python main.py --dataset IP --tr_percent 0.1 --dropprob 0.8 --verbose

```
