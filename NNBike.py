import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import DataMG as rd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#print(tf.config.experimental.list_physical_devices('CPU'))
def cons_model(arch):
    n = (len(arch) - 1)
    act = 'relu'
    model = keras.Sequential()

    for i in range(n):
        if i == n:
            act = 'tanh'
        else:
            act = 'softmax'
            pass
        input_node=arch[i+1]
        layer =  keras.layers.Dense(input_node,activation=act)
        model.add(layer)

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='squared_hinge',
                  metrics=['accuracy'])
    return model
def save_model(model,fp):
    model_json= model.to_json()
    with open(fp, "w") as js_f:
        js_f.write(model_json)
    model.save_weights('model22.h5')
def run_model(arch,dt_trn,dt_tst,epochs=10):
    model = cons_model(arch)
    model.fit(dt_trn, epochs=epochs)
    scores = model.evaluate(dt_tst, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    save_model(model,os.path.join('model', 'model22.json'))

def main():
    pth = r'2019-03-01 00_00_00-2019-03-31 00_00_00-citibike-01tripdata.csv'
    pth = os.path.normpath(path=pth)
    # the directory of the dataset
    data = rd.dataMG(pth)
    # make the data managing object.
    lrn = data.read2learn(frac=0.3)
    # run the proportion function to read data for learning
    # it has to dataframe object in it one for training the other for testing
    trin = lrn[0]
    tst = lrn[1]
    '''['end_st_cluster1', 'end_st_cluster2', 'end_st_cluster3', 'end_st_cluster4', 'end_st_cluster5']['area_Com', 'area',]['in_cls', 'out_cls']'''
    feature = trin[
        ['start_st_cluster1', 'start_st_cluster2', 'start_st_cluster3','start_st_cluster4', 'start_st_cluster5',
         'dayofweek1','dayofweek2','dayofweek3','dayofweek4','dayofweek5',
         'hours1','hours2','hours3','hours4','hours5']].copy()
    label = trin[['end_st_cluster1', 'end_st_cluster2', 'end_st_cluster3', 'end_st_cluster4', 'end_st_cluster5'
                  ]].copy()
    tst_feat = tst[
        ['start_st_cluster1', 'start_st_cluster2', 'start_st_cluster3','start_st_cluster4', 'start_st_cluster5',
         'dayofweek1','dayofweek2','dayofweek3','dayofweek4','dayofweek5',
         'hours1','hours2','hours3','hours4','hours5']].copy()
    tst_lbl = tst[['end_st_cluster1', 'end_st_cluster2', 'end_st_cluster3', 'end_st_cluster4', 'end_st_cluster5'
                   ]].copy()
    dt_trn = tf.data.Dataset.from_tensor_slices((feature.values, label.values)).shuffle(len(trin)).batch(1)
    dt_tst = tf.data.Dataset.from_tensor_slices((tst_feat.values, tst_lbl.values)).shuffle(len(tst)).batch(1)
    print(dt_trn)

    num_input = feature.shape[1]
    num_output = label.shape[1]
    arch = [num_input,12, num_output]
    print(arch)
    run_model(arch, dt_trn, dt_tst, epochs=10)


if __name__ == '__main__':
    main()