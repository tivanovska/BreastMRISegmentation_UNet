import os
import argparse
import dicom
import numpy as np
import tensorflow as tf
import cv2

from keras.models import Model, load_model


from keras import backend as K

from LoadBatches import imageSegmentationGenerator, imageSegmentationFilenameGenerator
from unet import dice_coef_loss, dice_coef, get_subj_allocation, get_mean_and_shape

import matplotlib.pyplot as plt

PATH_RESULT = './../results/'
PATH_DATA = './../data/'
PATH_SEGS = PATH_DATA + 'Mamma_Segmentation_corrected/MammaVolume/'
#PATH_IMAGES = './../../Mamma_Segmentation_corrected/MammaVolume/'
PATH_IMAGES = PATH_DATA + 'preprocessed_N4ITK_own_mask/'
NUMBER_OF_SUBJ_PER_ACR = 10
PATH_PREDICTIONS = PATH_RESULT + 'Predictions/'

#https://arxiv.org/pdf/1606.04797.pdf
smooth = 1.0 # preventing 0/0
def np_dice_coef_gen(g, y_pred, batch_size):
    y_true = []
    count = 0
    for i in g:
        y_true.append(i[1])
        count += 1
        if count >= int(256 / batch_size):
            break
    y_true = np.asarray(y_true)
    y_true = y_true.reshape(256, 1, 512, 256)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def np_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def cut_and_mirror_transpose(img):
    left = np.where(img[0,0] == 1, 255, 0)
    right = np.where(img[1,0] == 1, 255, 0)
    return np.concatenate((left, np.flipud(right)), axis=1).astype('uint16')

def load_and_predict(model, subj, mean, batch_size=2):
    g = imageSegmentationGenerator(PATH_IMAGES, PATH_SEGS, subj, batch_size, 1, mean=mean, shuffle=False)
    filename_generator = imageSegmentationFilenameGenerator(PATH_IMAGES, PATH_SEGS, subj, 1, 1, shuffle=False)
    counter = 0
    y_true = []
    x = []
    for i in g:
        counter += 1
        x.append(i[0])
        y_true.append(i[1])
        if counter >= int(256 / batch_size):
            break
    x = np.asarray(x)
    y_true = np.asarray(y_true)
    x = x.reshape((256, 1, 512, 256))
    y_true = y_true.reshape((256, 1, 512, 256))
    predictions = model.predict(x, verbose=1)
    predictions = np.where(predictions > 0.5, 1, 0)
    predictions = correct_mask(predictions)
    counter = 0
    for i in filename_generator:
        save_pred(predictions[counter * 2 : counter * 2 + 2], i[1][0], subj)
        counter += 1
        if counter >= int(128):
            break
    return np_dice_coef(y_true, predictions)

def correct_mask(img):
    img = img.astype('uint8', casting='unsafe', copy=True)
    img = np.where(img == 1, 255, 0)
    img_out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        img_floodfill = img[i, 0].copy().astype('uint8')
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = img[i,0].shape[:2]
        mask = np.zeros((h+2, w+2)).astype('uint8')
        
        # Floodfill from point (0, 0)
        cv2.floodFill(img_floodfill, mask, (0, 0), 255);
        
        # Invert floodfilled image
        img_floodfill_inv = cv2.bitwise_not(img_floodfill)
        
        # Combine the two images to get the foreground.
        img_out[i, 0] = img[i, 0] | img_floodfill_inv
    img_out = np.where(img_out==255, 1, 0)
    return img_out

def save_pred(pred, original_filename, subj):
    subj = subj.astype(int)
    subj = subj.astype(str)
    for i in range(subj.shape[0]):
        if(len(subj[i])<2):
            subj[i] = '0' + subj[i]
    ds = dicom.read_file(original_filename)
    pred = cut_and_mirror_transpose(pred)
    ds.PixelData = pred.tostring()
    #print(PATH_PREDICTIONS, 'subj_', subj[0], '/', original_filename.split('/')[-1])
    if not os.path.exists(PATH_PREDICTIONS + 'subj_' + subj[0] + '/'):
        os.makedirs(PATH_PREDICTIONS + 'subj_' + subj[0] + '/')
    ds.save_as(str(PATH_PREDICTIONS + '/subj_' + subj[0] + '/' + original_filename.split('/')[-1]))
    return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict and save predicted train data for a given model')
    parser.add_argument('model_path', type=str, help='set path to model')
    args = parser.parse_args()
    batch_size = 2
    path_model = args.model_path
    model = load_model(path_model, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    fold = int(path_model.split('/')[-2].split('_')[-1])
    testset_subj, trainset_subj = get_subj_allocation(fold)
    print(testset_subj)
    mean, data_shape = get_mean_and_shape(trainset_subj, batch_size)
    for i in testset_subj:
        dsc = load_and_predict(model, np.asarray([i]), mean)
        f = open(PATH_RESULT + "dsc_fold_%d.txt"%(fold), 'a+')
        f.write('%d \t %f \n' %(i, dsc))
        f.close()
        
        
        
    #print(np.asarray([testset_subj[0]]))
    #filename_generator = imageSegmentationFilenameGenerator(PATH_IMAGES, PATH_SEGS, np.asarray([testset_subj[0]]), 1, 1, shuffle=False)
    #g = imageSegmentationGenerator(PATH_IMAGES, PATH_SEGS, np.asarray([testset_subj[0]]), batch_size, 1, mean=mean, shuffle=False)
    #counter = 0
    #y_true = []
    #x = []
    #for i in g:
        #counter += 1
        #x.append(i[0])
        #y_true.append(i[1])
        #if counter >= int(256 / batch_size):
            #break
    #y_true = np.asarray(y_true)
    #print(y_true.shape)
    #y_true = y_true.reshape((2 * 128, 1, 512, 256))
    #counter = 0
    #for i in filename_generator:
        #print(i[1][0])
        #save_pred(y_true[2 * counter : 2 * (counter + 1)], i[1][0], np.asarray([testset_subj[0]]))
        #counter += 1
        #if counter >= int(128):
            #break
    #save_pred(x, './../data/Mamma_Segmentation_corrected/MammaVolume/subject_01/MammaVolume_subj_01_slice_001.dcm', '01')
        
    #batch_size = 8
    #subj = np.asarray(['01'])
    #g = imageSegmentationGenerator(PATH_IMAGES, PATH_SEGS, subj, batch_size, 1, shuffle=False)
    #filename_generator = imageSegmentationFilenameGenerator(PATH_IMAGES, PATH_SEGS, subj, batch_size, 1, shuffle=False)
    #a = []
    #b = []
    #counter = 0
    #for i, j in zip(g, filename_generator):
        #a.append(i[1])
        #counter += 1
        #if counter >= int(256 / batch_size):
            #break
    #a = np.asarray(a)
    #print(a.shape)
    #print(np_dice_coef_gen(g, a, batch_size))