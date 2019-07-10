import matplotlib as mpl
mpl.use("Agg") #this line allows mpl to run with no display defined

'''importing all required modules
'''
import pandas as pd
import numpy as np
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.regularizers import L1L2



'''
importing Adversarial GANs Modules
'''
from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, fix_names
from keras.backend import backend as K
from cifar10_web import cifar10


'''
    defining generator that uses combinations of convolutions and 11 and 12
    In addition each Conv2D uses a LeakyReLU activation function and BatchNormalization
    
'''
def model_generator():
    model=Sequential()
    nch=256
    reg=lambda :L1L2()