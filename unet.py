import os
import glob
import argparse
import warnings
import sys
import random
import numpy as np
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, History, EarlyStopping
from keras.initializers import glorot_uniform
from keras.regularizers import l2
#from keras_contrib.callbacks import DeadReluDetector


from keras import backend as K

from LoadBatches import imageSegmentationGenerator

PATH_RESULT = './../results/'
NUMBER_OF_SUBJ_PER_ACR = 10


#https://arxiv.org/pdf/1606.04797.pdf
smooth = 1.0 # preventing 0/0
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_subj_allocation(fold):
    if os.path.isfile(PATH_RESULT + 'subj_allocation.npy'):
        a = np.load(PATH_RESULT + 'subj_allocation.npy')
    else:
        print('file: subj_allocation.npy not found. Creating new one.')
        a = np.zeros((4,NUMBER_OF_SUBJ_PER_ACR))
        a[0] = np.random.permutation(np.arange(NUMBER_OF_SUBJ_PER_ACR))
        a[1] = np.random.permutation(np.arange(NUMBER_OF_SUBJ_PER_ACR)) + NUMBER_OF_SUBJ_PER_ACR
        a[2] = np.random.permutation(np.arange(NUMBER_OF_SUBJ_PER_ACR)) + NUMBER_OF_SUBJ_PER_ACR * 2
        a[3] = np.random.permutation(np.arange(NUMBER_OF_SUBJ_PER_ACR)) + NUMBER_OF_SUBJ_PER_ACR * 3
        if not os.path.exists(PATH_RESULT):
            os.mkdir(PATH_RESULT)
        np.save(PATH_RESULT + 'subj_allocation.npy', a)
    return a[:, 2 * fold : 2 * fold + 2].flatten() + 1, np.delete(np.arange(4 * NUMBER_OF_SUBJ_PER_ACR), a[:, 2 * fold : 2 * fold + 2].flatten()) + 1

def get_validation_set_subjects(trainingset):
    a = np.zeros((4), dtype='int')
    a[0] = np.random.randint(0, high = int(trainingset.shape[0]/4))
    a[1] = np.random.randint(0, high = int(trainingset.shape[0]/4)) + int(trainingset.shape[0]/4)
    a[2] = np.random.randint(0, high = int(trainingset.shape[0]/4)) + int(trainingset.shape[0]/4) * 2
    a[3] = np.random.randint(0, high = int(trainingset.shape[0]/4)) + int(trainingset.shape[0]/4) * 3
    val = list(trainingset[a])
    trainingset = np.delete(trainingset, a)
    b = np.random.randint(0, high = trainingset.shape[0], size=(2))
    while b[0]==b[1]:
        b = np.random.randint(0, high = trainingset.shape[0] - 4, size=(2))
    val.append(trainingset[b[0]])
    val.append(trainingset[b[1]])
    return np.delete(trainingset, b), np.asarray(val)

def get_mean_and_shape(trainset_subj, batch_size, images_per_subject):
    print(trainset_subj)
    G = imageSegmentationGenerator(path_images, path_segs, trainset_subj, batch_size, 1)
    steps = images_per_subject * trainset_subj.shape[0] / batch_size
    counter = 0
    summe = 0
    mean = 0
    shape = 0
    for i in G:
        summe += np.sum(i[0])
        counter += 1
        if (counter > steps):
            mean = summe / (images_per_subject * trainset_subj.shape[0] * i[0].shape[2] * i[0].shape[3])
            shape = i[0].shape[1:]
            break
    return mean, shape
        

def get_unet(img_shape, num_classes, learning_rate=1e-4, reg=1e-4, first_layer_depth=32, dropout_p=0.0):
    inputs = Input((img_shape))
    conv1 = Conv2D(first_layer_depth, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(inputs)
    #conv1 = Dropout(dropout_p)(conv1)
    conv1 = Conv2D(first_layer_depth, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv1)
    #conv1 = Dropout(dropout_p)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(first_layer_depth * 2, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(pool1)
    #conv2 = Dropout(dropout_p)(conv2)
    conv2 = Conv2D(first_layer_depth * 2, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv2)
    #conv2 = Dropout(dropout_p)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(first_layer_depth * 4, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(pool2)
    #conv3 = Dropout(dropout_p)(conv3)
    conv3 = Conv2D(first_layer_depth * 4, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv3)
    #conv3 = Dropout(dropout_p)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(first_layer_depth * 8, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(pool3)
    #conv4 = Dropout(dropout_p)(conv4)
    conv4 = Conv2D(first_layer_depth * 8, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv4)
    #conv4 = Dropout(dropout_p)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(first_layer_depth * 16, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(pool4)
    #conv5 = Dropout(dropout_p)(conv5)
    conv5 = Conv2D(first_layer_depth * 16, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv5)
    #conv5 = Dropout(dropout_p)(conv5)
    up6 = concatenate([Conv2DTranspose(first_layer_depth * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv5), conv4], axis=1)
    conv6 = Conv2D(first_layer_depth * 8, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(up6)
    #conv6 = Dropout(dropout_p)(conv6)
    conv6 = Conv2D(first_layer_depth * 8, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv6)
    #conv6 = Dropout(dropout_p)(conv6)
    up7 = concatenate([Conv2DTranspose(first_layer_depth * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv6), conv3], axis=1)
    conv7 = Conv2D(first_layer_depth * 4, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(up7)
    #conv7 = Dropout(dropout_p)(conv7)
    conv7 = Conv2D(first_layer_depth * 4, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv7)
    #conv7 = Dropout(dropout_p)(conv7)
    up8 = concatenate([Conv2DTranspose(first_layer_depth * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv7), conv2], axis=1)
    conv8 = Conv2D(first_layer_depth * 2, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(up8)
    #conv8 = Dropout(dropout_p)(conv8)
    conv8 = Conv2D(first_layer_depth * 2, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv8)
    #conv8 = Dropout(dropout_p)(conv8)
    up9 = concatenate([Conv2DTranspose(first_layer_depth, (3, 3), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv8), conv1], axis=1)
    conv9 = Conv2D(first_layer_depth, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(up9)
    #conv9 = Dropout(dropout_p)(conv9)
    conv9 = Conv2D(first_layer_depth, (3, 3), activation='relu', padding='same', kernel_initializer=glorot_uniform(seed=None), kernel_regularizer=l2(reg))(conv9)
    #conv9 = Dropout(dropout_p)(conv9)
    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid', kernel_initializer=glorot_uniform(seed=None))(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=RMSprop(learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unet')
    parser.add_argument('--first_layer_depth', help='Depth of the first layer - the size of the other layers will be set automatically', default=32)
    parser.add_argument('--epochs',
                        default=10,
                        help='Number of epochs')
    parser.add_argument('--reg', default=1e-4, help='Regularization')
    parser.add_argument('--debug', default=False, help='Set true to get debug informations and shorter runtime')
    parser.add_argument('--dropout_p', default=0.0, help='Set dropout probability')
    parser.add_argument('--load_model', type=str, help='Set dropout probability')
    parser.add_argument('--train_mode', type=str, help='Set Volume for Volumesegmentation, Set FGT for FGT-Segmentation')
    parser.add_argument('--data', type=str, help='Use axial, coronal or sagittal data')
    parser.add_argument('--fold', default=0, help='Set fold of cross validation. Possible numbers: 0-4')
    args = parser.parse_args()
    first_layer_depth = int(args.first_layer_depth)
    batch_size = int(8 * 64 / first_layer_depth)
    reg = float(args.reg)
    epochs = int(args.epochs)
    fold = int(args.fold)
    debug = bool(args.debug)
    dropout_p = float(args.dropout_p)
    data = args.data
    train_mode = args.train_mode
    testset_subjects = 8
    #images_per_subject = 128
    num_classes = 1
    fold_per_fold = 5
    
    if data == 'Axial':
        images_per_subject = 
        images_per_subject = 
        path_data = './../data/'
        path_segs = path_data + 'Mamma_Segmentation_corrected/MammaVolume/'
        #path_images = './../../Mamma_Segmentation_corrected/MammaVolume/'
        path_images = path_data + 'preprocessed_N4ITK_own_mask/'
    for fold in range(2,3):
        testset_subj, trainset_subj = get_subj_allocation(fold)
        print('testsetgroesse:',  testset_subj.shape, 'trainingssetgroesse:',  trainset_subj.shape)
        mean, data_shape = get_mean_and_shape(trainset_subj, batch_size, images_per_subject)
        print('mean:', mean, 'shape:', data_shape)
        if(args.load_model):
            sess = tf.Session()
            K.set_session(sess)
            model_path = glob.glob(args.load_model + 'fold_' + str(fold) + '/*.h5')[0]
            model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
            #model = load_model(model_path)
            learning_rates = np.random.uniform(K.eval(model.optimizer.lr), K.eval(model.optimizer.lr) * 0.3, fold_per_fold)
            sess.close()
        else:
            learning_rates = np.random.uniform(10**-4.3, 10**-5, fold_per_fold)
        if not os.path.exists(PATH_RESULT + 'fold_' + str(fold)):
            os.makedirs(PATH_RESULT + 'fold_' + str(fold))
        np.save(PATH_RESULT + 'fold_' + str(fold) + '/lr.npy', learning_rates)
        for i in range(fold_per_fold):
            sess = tf.Session()
            K.set_session(sess)
            print("Running Fold", fold + 1, "/", 5, '\t', i + 1 , '/', fold_per_fold)
            train, val =  get_validation_set_subjects(trainset_subj)
            print('train:',  train.shape, 'val:',  val.shape, 'lr:', learning_rates[i])
            G = imageSegmentationGenerator(path_images, path_segs, train, batch_size, 1, mean=mean)
            G_val = imageSegmentationGenerator(path_images, path_segs, val, batch_size, 1, mean=mean)
            if(args.load_model):
                model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
                #model = load_model(model_path)
                K.set_value(model.optimizer.lr, learning_rates[i])
            else:
                model = get_unet(data_shape, num_classes, learning_rate=learning_rates[i], reg=reg, first_layer_depth=first_layer_depth, dropout_p=dropout_p)
            best_vall_loss = 1e6
            for ep in range(epochs):
                history = model.fit_generator(G, steps_per_epoch=int(train.shape[0] * images_per_subject / batch_size), validation_data=G_val, validation_steps=int(val.shape[0] * images_per_subject / batch_size), epochs=1)
                numpy_loss_history = np.array([history.history['loss'], history.history['val_loss'], history.history['dice_coef'], history.history['val_dice_coef'], np.arange(len(history.history['loss']))]).transpose(1,0)
                f = open(PATH_RESULT + 'fold_' + str(fold) + "/val_loss_history_fold_%d.txt"%(i), 'a+')
                f.write('%d \t %f \t %f \t %f \t %f\n' %(ep, history.history['loss'][-1], history.history['val_loss'][-1], history.history['dice_coef'][-1], history.history['val_dice_coef'][-1]))
                if (history.history['val_loss'][-1] < best_vall_loss):
                    best_vall_loss = history.history['val_loss']
                    model.save( PATH_RESULT + 'fold_' + str(fold) + '/fold_' + str(i) + '.h5')
                else: 
                    f.close()
                    break
            sess.close()
            