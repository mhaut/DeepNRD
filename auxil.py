import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


def random_unison(a,b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]

def random_single(a, rstate=None):
    return a[np.random.RandomState(seed=rstate).permutation(len(a))]

def split_data(pixels, labels, percent, splitdset="sklearn", rand_state=32):
    pixels_number = np.unique(labels, return_counts=1)[1]
    train_set_size = [int(np.ceil(a*percent)) for a in pixels_number]
    #train_set_size = [int(a*percent) for a in pixels_number]
    train_set_size = [1 if n == 0 else n for n in train_set_size]
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number))- int(sum(train_set_size))
    sizetr = np.array([tr_size]+list(pixels.shape)[1:])
    sizete = np.array([te_size]+list(pixels.shape)[1:])
    train_x = np.empty((sizetr)); train_y = np.empty((tr_size)); test_x = np.empty((sizete)); test_y = np.empty((te_size))
    trcont = 0; tecont = 0;
    for cl in np.unique(labels).astype("int"):
        pixels_cl = pixels[labels==cl]
        labels_cl = labels[labels==cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a,b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_x[trcont] = a
                train_y[trcont] = b
                trcont += 1
            else:
                test_x[tecont,] = a
                test_y[tecont] = b
                tecont += 1
    train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
    test_x, test_y   = random_unison(test_x, test_y, rstate=rand_state)
    return train_x, test_x, train_y, test_y
    #if splitdset == "sklearn":
        #X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=(1-percent), stratify=labels, random_state=rand_state)
        #print(np.unique(y_train, return_counts=1))
        #exit()
        ##return 
    #elif splitdset == "custom":
        #X_train = []; X_test = []; y_train = []; y_test = [];
        #values = np.ceil(np.unique(labels, return_counts=1)[1]*percent).astype("int")
        #for idi, i in enumerate(values):
            #samples = pixels[labels==idi]
            #samples = random_single(samples, rstate=rand_state)
            #for a in samples[:i]: 
                #X_train.append(a); y_train.append(idi)
            #for a in samples[i:]:
                #X_test.append(a); y_test.append(idi)
        #X_train = np.array(X_train); X_test = np.array(X_test)
        #y_train = np.array(y_train); y_test = np.array(y_test)
        #X_train, y_train = random_unison(X_train,y_train, rstate=rand_state)
        #return X_train, X_test, y_train, y_test

def loadData(name, num_components=None, preprocess="minmax"):
    data_path = os.path.join(os.getcwd(),'HSI-datasets/Classification/')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected_gt.mat'))['indian_pines_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    else:
        print("NO DATASET TESTED")
        exit()

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    if preprocess == "minmax":
        data = MinMaxScaler().fit_transform(data)
    elif preprocess == "standard":
        data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def getaccuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred) * 100

def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred, labels=range(len(np.unique(y_pred))))
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
