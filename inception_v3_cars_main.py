# -*- coding: utf-8 -*-

#import required libraries
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D,  Flatten, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
from glob import glob
from PIL import Image
from matplotlib.pyplot import imshow


# helper function for Inception v3 CNN Model
def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    """
    Utility function to apply conv + BN for Inception V3.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 1
    x = Conv2D(nb_filter, (nb_row, nb_col),
                      strides=subsample,
                      activation='relu',
                      padding=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

# The structure of Inception v3 model
def inception_v3_model(img_rows, img_cols, channel=1, num_classes=None):
    """
    Inception-V3 Model for Keras
    Model Schema is based on 
    https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels.h5
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
    channel_axis = 1
    img_input = Input(shape=(channel, img_rows, img_cols))
    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], 
                        axis=channel_axis, name='mixed' + str(i))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, 
                    name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], 
                    axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], 
                        axis=channel_axis, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], 
                    axis=channel_axis, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis,
                    name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, 
                                name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], 
                                   axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], 
                        axis=channel_axis, name='mixed' + str(9 + i))

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x_fc = Flatten(name='flatten')(x_fc)
    x_fc = Dense(1000, activation='softmax', name='predictions')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data
    # Initially model is trained with ImageNet weights 
    # model.load_weights('inception_v3_weights_th_dim_ordering_th_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in 
    # layers but not in the model
    x_newfc = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x_newfc = Flatten(name='flatten')(x_newfc)
    # Then model is trained with VMMRdb dataset from http://vmmrdb.cecsresearch.org/
    # According to http://vmmrdb.cecsresearch.org/ 3040 classes are selected
    x_newfc = Dense(3040, activation='softmax', name='predictions')(x_newfc)
    
    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)
    # accuracy of predicting of test set of VMMRdb dataset was 36%
    model.load_weights('VMMRdb-incv3-27-0.36742.h5')
    
    # creating new Fully Connected layer for Cars dataset with 196 nodes
    x_newfc_new = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x_newfc_new = Flatten(name='flatten')(x_newfc_new)
    x_newfc_new = Dense(196, activation='softmax', name='predictions')(x_newfc_new)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc_new)
    model.load_weights('cars_incv3-276-0.81060.h5')
    # print(model.summary())
    
    # freeze all weights
    for layer in model.layers:
        layer.trainable = False

    # make trainable only later layers
    for layer in model.layers:
        if layer.name == "predictions":
            layer.trainable = True
        if layer.name == "flatten":
            layer.trainable = True
        if layer.name == "avg_pool":
            layer.trainable = True

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    return model 

if __name__ == '__main__':

    # image size for Inception v3
    img_width, img_height = 299, 299
    
    # directory for test images
    validation_data_dir = 'Cars/test/'    
    
    _batch_size = 40 # batch size
    num_classes = 196 # number of classes for Cars dataset
    channel = 3 # number of channels for input tensor (rgb)

    # check dimensionaly ordering for theano and tensorflow
    if K.image_data_format() == 'channels_first':
        input_size = (3, img_width, img_height)
    else:
        input_size = (img_width, img_height, 3)

    # input test 
    x_test = []
    # read image from specific location
    img = Image.open(validation_data_dir + '163/03819.jpg')
    
    # output an image
    img.show()   
    
    # resize image to proper size for Inception v3
    img = img.resize((299,299), Image.ANTIALIAS)
    # add to the list    
    x_test.append(np.array(img))
    # convert to numpy array    
    x_test = np.array(x_test)
    # change to the propoer input size
    # for theano it should be (1, 3, 299, 299)
    x_test= np.swapaxes(np.swapaxes(x_test, 2, 3), 1, 2)
    # load trained model
    model = inception_v3_model(img_height, img_width, channel, num_classes)
    
    folders = glob(validation_data_dir + "*/")    
    
    # car makes list will store all classes in Cars dataset from cars.txt file
    car_makes = []
    with open('cars.txt') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            tokens = lines[i].split(',')
            # first number is index and second number is car name
            # for car make, the last 2 character is \n, therefore we append
            # only car make name
            car_makes.append(tokens[1][:-2])
    
    # since we are testing only one image, then batch size = 1
    batch_size = 1
    # predict input image
    scores = model.predict(x_test, batch_size) 
    
    for i in range(scores.shape[0]):
        # get best 5 top scores
        best_5 = np.argsort(scores[i])[191:]
        top5 = []
        for j in range(best_5.shape[0]):
            top5.append(int(folders[best_5[j]].split("\\")[1]))
        # reverse the list, since the top 1 in the last position
        # top 2 in the pre last and so on
        top5 = top5[::-1]
        
        # get the class name in string representation 
        top5_res = [car_makes[x] for x in top5]
        print(top5_res)
    
