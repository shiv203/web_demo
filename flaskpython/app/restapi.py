import random,time
from autocorrect import spell
# from __future__ import division
# from __future__ import print_function

import json
import sys
import argparse
import cv2,os
import editdistance
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from flask import jsonify 
from keras.models import Sequential,Model
from keras.layers import LSTM,Bidirectional,Dense,Activation,Lambda,Input
import keras.backend as K
from keras.optimizers import Adam, SGD, RMSprop,Adadelta
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
charset = u' !@#><~%&\$^*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
maxstrokeslen = 500
outputtextlen = len(charset)+1
input_layer = Input((maxstrokeslen, 3))
X = Bidirectional( LSTM(units = 512,return_sequences = True) ) (input_layer)
X = Bidirectional( LSTM(units = 512,return_sequences = True) ) (X)
X = Dense(outputtextlen)(X)
X = Activation('softmax', name='softmax')(X)
test_model = Model(input_layer,X)#.summary()
test_model.load_weights('Modell00000010.hdf5')
graph = tf.get_default_graph()

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(charset):  # CTC Blank
            ret.append("")
        else:
            ret.append(charset[c])
    return "".join(ret)

import itertools
#charset = u' !@#><~%&\$^*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, :], axis=-1))  # get max index -> len = 32
    #print(out_best)
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    #print(out_best)
    outstr = ''
    for i in out_best:
        if i < len(charset) and i > 0:
            outstr += charset[i]
    return outstr

def levenshteinDistance(s1, s2):
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


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(charset.find(char))
    return ret

import math

def cubicBezierPoint(a0, a1, a2, a3, t):
    return math.pow(1 - t, 3) * a0 + 3 * math.pow(1 - t, 2) * t * a1 + 3 * (1 - t) * math.pow(t, 2) * a2+ math.pow(t, 3) * a3

def bezier_curve(points):

    controlpoints = []
    renderpoints = []

    for i in range(1,len(points)-1,2):
        cp = ((points[i-1][0]+points[i][0])/2,(points[i-1][1]+points[i][1])/2,(points[i-1][2]+points[i][2])/2)
        controlpoints.append(cp)
        controlpoints.append(points[i])
        controlpoints.append(points[i+1])

        if (i + 2 ) < (len(points) - 1) :
            cp1 = ((points[i+1][0]+points[i+2][0])/2,(points[i+1][1]+points[i+2][1])/2,(points[i+1][2]+points[i+2][2])/2)
            controlpoints.append(cp1)

    for i in range(0, len(controlpoints) - 3,4) :
        a0 = controlpoints[i]
        a1 = controlpoints[i+1]
        a2 = controlpoints[i+2]
        a3 = controlpoints[i+3]
        op = (cubicBezierPoint(a0[0], a1[0], a2[0], a3[0], 0),cubicBezierPoint(a0[1], a1[1], a2[1], a3[1], 0),cubicBezierPoint(a0[2], a1[2], a2[2], a3[2], 0))
        renderpoints.append(op)
    #print(renderpoints)
    return renderpoints

def beizerprocess(data):
    builder = []
    for i in range(0,len(data)):
        x_cord = []
        y_cord = []
        for j in range(0,len(data[i][0])):
            x_cord.append(data[i][0][j])
            y_cord.append(data[i][1][j])
        minX = min(x_cord)
        maxX = max(x_cord)
        Xvalue = maxX - minX
        x_cord_mean = [(elt - minX)/Xvalue for elt in x_cord]

        minY = min(y_cord)
        maxY = max(y_cord)
        Yvalue = maxY - minY
        y_cord_mean = [(elt - minY)/Yvalue for elt in y_cord]

        temptuple = []
        for i in range(0,len(x_cord_mean)):
            temptuple.append( (x_cord_mean[i],y_cord_mean[i],0) )

        bezpoints = bezier_curve(temptuple)
        for l in range(0,len(bezpoints)):
            if (l == 0):
                templ = []
                templ.append(bezpoints[l][0])
                templ.append(bezpoints[l][1])
                templ.append(1)
            else:
                templ = []
                templ.append(bezpoints[l][0])
                templ.append(bezpoints[l][1])
                templ.append(0)
            builder.append(templ)
    return builder
def HWRmodel(text_data):

    data = text_data

    if len(data) != maxstrokeslen:
        c = len(data)
        for j in range(c, maxstrokeslen):
            data.append([-1,-1,-1])
            c+=1
    num = 71
    t = np.expand_dims(data,axis = 0)
    st = time.time()
    with graph.as_default():
        prediction = test_model.predict(t)
    et = time.time()
    # print(et - st)
    de = decode_label(prediction)
    sde = spell(de)
    print("prediction and spellprediction",de,sde)
    return sde

from flask import Flask, url_for, request
app = Flask(__name__)

# @app.route('/apitest')
# def apitest():
#     return 'API working'

#@app.route('/hwrrecog', methods=['POST'])
def hwrrecog(data):
   
    

    beizerdata = beizerprocess(data)
    text_out = HWRmodel(beizerdata)
    op1 = {'output':text_out}

    return  jsonify(op1)


