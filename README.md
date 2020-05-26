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
python main.py --tr_percent 0.15 --dataset IP --verbose # without data augmentation

# Hyperspectral Image Classification Using Random Occlusion Data Augmentation
python main.py --p 0.25 --tr_percent 0.15 --dataset IP --verbose # with data augmentation
python main.py --p 0.5 --tr_percent 0.15 --dataset IP --verbose # with data augmentation


```
