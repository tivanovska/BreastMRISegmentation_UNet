import os
import re
import numpy as np
import nibabel as nib
import dicom
import cv2
import glob
import itertools
import random 

# Test
import matplotlib.pyplot as plt

PATH_DICOM = './../../Mamma_Segmentation/'
PATH_DICOM_NEW = './../../Mamma_Segmentation_corrected/'
PATH_N4ITK_OWN_MASK_NEW = './../data/preprocessed_N4ITK_own_mask/'

def getSegmentationArr(file_dcm, n_classes):
    ds = dicom.read_file(file_dcm)
    array = ds.pixel_array
    return array.reshape(1, array.shape[0], array.shape[1])


def getImageArr(filename_n4itk):
    n4itk = nib.load(filename_n4itk)
    array = n4itk.get_data()
    return array.reshape(1, array.shape[0], array.shape[1])

def cut_and_mirror(img):
    lr_splitted = np.split(img, np.array(2), axis=2)
    return np.asarray([lr_splitted[0], np.fliplr(lr_splitted[1])])




def imageSegmentationGenerator(images_path, segs_path, subj_nr, batch_size, n_classes, mean=0, shuffle=True):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    subj_nr = subj_nr.astype(int)
    subj_nr = subj_nr.astype(str)
    for i in range(subj_nr.shape[0]):
        if(len(subj_nr[i])<2):
            subj_nr[i] = '0' + subj_nr[i]
    images = []
    segmentations = []
    for i in range(len(subj_nr)):
        #print(subj_nr[i])
        images += glob.glob(images_path + 'subject_' + subj_nr[i] + '/*')
        segmentations += glob.glob(segs_path + 'subject_' + subj_nr[i] + '/*')
    segmentations.sort()
    images.sort()
    if shuffle == True:
        data = np.random.permutation(list(zip(images, segmentations)))
    else:
        data = list(zip(images, segmentations))
    assert len( images ) == len(segmentations)
    for im , seg in data:
        assert(im.split('/')[-1].split(".")[0].split('_')[1:] ==  seg.split('/')[-1].split(".")[0].split('_')[1:])

    zipped = itertools.cycle(data)

    while True:
        X = []
        Y = []
        for _ in range(int(batch_size/2)):
            im , seg = next(zipped)
            X.append(cut_and_mirror(getImageArr(im) - mean))
            Y.append(cut_and_mirror(np.where(getSegmentationArr(seg, n_classes) == 255, 1, 0)))
        X = np.array(X)
        Y = np.array(Y)
        yield X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]) , Y.reshape(Y.shape[0] * Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])

def imageGenerator(images_path, segs_path, subj_nr, batch_size, n_classes, mean=0, shuffle=True):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    subj_nr = subj_nr.astype(int)
    subj_nr = subj_nr.astype(str)
    for i in range(subj_nr.shape[0]):
        if(len(subj_nr[i])<2):
            subj_nr[i] = '0' + subj_nr[i]
    images = []
    for i in range(len(subj_nr)):
        images += glob.glob(images_path + 'subject_' + subj_nr[i] + '/*')
    images.sort()
    if shuffle == True:
        data = np.random.permutation(list(zip(images)))
    else:
        data = list(zip(images))
    
    zipped = itertools.cycle(data)

    while True:
        X = []
        Y = []
        for _ in range(int(batch_size/2)):
            im = next(zipped)
            X.append(cut_and_mirror(getImageArr(im) - mean))
        X = np.array(X)
        yield X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]) 


def imageSegmentationFilenameGenerator(images_path, segs_path, subj_nr, batch_size, n_classes, mean=0, shuffle=False):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    subj_nr = subj_nr.astype(int)
    subj_nr = subj_nr.astype(str)
    for i in range(subj_nr.shape[0]):
        if(len(subj_nr[i])<2):
            subj_nr[i] = '0' + subj_nr[i]
    images = []
    segmentations = []
    for i in range(len(subj_nr)):
        images += glob.glob(images_path + 'subject_' + subj_nr[i] + '/*')
        segmentations += glob.glob(segs_path + 'subject_' + subj_nr[i] + '/*')
    segmentations.sort()
    images.sort()
    if shuffle == True:
        data = np.random.permutation(list(zip(images, segmentations)))
    else:
        data = list(zip(images, segmentations))
    assert len( images ) == len(segmentations)

    zipped = itertools.cycle(data)

    while True:
        X = []
        Y = []
        for _ in range(int(batch_size)):
            im , seg = next(zipped)
            X.append(im)
            Y.append(seg)
        X = np.array(X)
        Y = np.array(Y)
        yield X, Y

if __name__ == "__main__":
    PATH_DATA = './../data/'
    PATH_SEGS = PATH_DATA + 'Mamma_Segmentation_corrected/MammaVolume/'
    #PATH_IMAGES = './../../Mamma_Segmentation_corrected/MammaVolume/'
    PATH_IMAGES = PATH_DATA + 'preprocessed_N4ITK_own_mask/'
    subj_nr = np.asarray(['01', '10', '40'])
#     G = imageSegmentationGenerator(images_path, segs_path,  batch_size,  n_classes, input_height, input_width, output_height, output_width)
    g =imageSegmentationGenerator(PATH_IMAGES, PATH_SEGS, subj_nr, 2, 1, shuffle=False)
    counter = 0
    for i in g:
        #print(i[0][0])
        counter += 1
        if (counter > 60): 
            plt.imshow(i[1][0,0])
            plt.show()
            plt.close()
            break

