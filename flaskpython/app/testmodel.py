from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers import Reshape, Lambda
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD , Adam
import tensorflow as tf

import random
import os
import cv2
import sys
import numpy as np
from keras.callbacks import TensorBoard
import itertools, time
from autocorrect import spell


drop_count = 0


CHAR_VECTOR = u' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

letters = [letter for letter in CHAR_VECTOR]

def ld(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, :], axis=-1))  # get max index -> len = 32
    #print(out_best)
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    #print(out_best)
    outstr = ''
    for i in out_best:
        if i < len(letters) and i > 0:
            outstr += letters[i]
    return outstr


def label_to_hangul(label):  # eng -> hangul
    region = label[0]
    two_num = label[1:3]
    hangul = label[3:5]
    four_num = label[5:]

    return region + two_num + hangul + four_num


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:

    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    global drop_count

    drop_count += 1
    input = Activation("relu")(input)
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    if drop_count % 2 == 0:
        norm = Dropout(0)(norm)
    return norm
    # norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    # return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual, is_first_block_of_first_layer, is_bypass_stride_required_2 = 0):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = 1
    #print(is_first_block_of_first_layer, is_bypass_stride_required_2)
    if input_shape[COL_AXIS] is not None:
        stride_width = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    elif is_bypass_stride_required_2 == 1:
        stride_width = 2
    #print(is_bypass_stride_required_2, is_first_block_of_first_layer, stride_width, input_shape[COL_AXIS])
    stride_height = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_height, stride_width),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, iteration, repetitions):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        is_first_layer = True
        #print(iteration, is_first_layer)
        for i in range(repetitions):
            init_strides = (1,1)
            if is_first_layer:
                if iteration == 1:
                    init_strides = (2, 2)
                elif iteration != 0:
                    init_strides = (2, 1)
            input = block_function(filters=filters, init_strides=init_strides, is_first_block_of_first_layer=is_first_layer and iteration == 0, is_bypass_stride_required_2=(iteration == 1 and i == 0))(input)
            is_first_layer = False
        return input

    return f


def basic_block(filters,kernel_size=(3,3), init_strides=(1, 1), is_first_block_of_first_layer=False, is_bypass_stride_required_2 = 0):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=kernel_size,
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual, is_first_block_of_first_layer, is_bypass_stride_required_2=is_bypass_stride_required_2)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False,is_bypass_stride_required_2 = 0):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual,is_first_block_of_first_layer,is_bypass_stride_required_2)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, embeding_size, absolute_max_string_len):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        # if K.image_dim_ordering() == 'tf':
        #     input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        # with tf.device('/job:ps/task:1/gpu:0'):
        input = Input(shape=input_shape, name='input')
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(1,1))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 1), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            # with tf.device('/job:ps/task:%d/gpu:0' % i):
            block = _residual_block(block_fn, filters=filters, iteration=i,repetitions=r)(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)
        block = Conv2D(filters=embeding_size, kernel_size=(2, 3), activation='softmax',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l2(1.e-4))(block)
        reshaped_block = Lambda(function=K.squeeze, arguments={'axis':1}, name='reshape')(block)#
        #print(absolute_max_string_len)
        labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([reshaped_block, labels, input_length, label_length])
        # Classifier block
        # block_shape = K.int_shape(block)
        # pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
        #                          strides=(1, 1))(block)
        # flatten1 = Flatten()(pool2)
        # dense = Dense(units=num_outputs, kernel_initializer="he_normal",
        #               activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=reshaped_block)
        return model

    #@staticmethod
    def build_resnet_18(input_shape, embeding_size, absolute_max_string_len, num_outputs = 10):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2,2,2,2], embeding_size, absolute_max_string_len)

    #@staticmethod
    def build_resnet_34(input_shape, embeding_size, absolute_max_string_len, num_outputs = 10):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3],embeding_size, absolute_max_string_len)

    #@staticmethod
    def build_resnet_50(input_shape,embeding_size, absolute_max_string_len, num_outputs = 10):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3],embeding_size, absolute_max_string_len)

    #@staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    #@staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


model = ResnetBuilder.build_resnet_18((32,128,3),80,14,62)
#rint(model.summary())
#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer="adadelta")

def get_model():
    return model
