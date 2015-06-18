#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
from caffe.proto import caffe_pb2

#solver_file_name = "autoencoder_solver.prototxt"
proto_file_name = "autoencoder.prototxt"
proto_file_name_init = "autoencoder_init.prototxt"
train_dbname = "ilsvrc12_train_lmdb"
test_dbname = "ilsvrc12_val_lmdb"
solver_file_name = "autoencoder_solver.prototxt"
caffe_model_init = "_iter_64000.caffemodel"
caffe_model = "_iter_800.caffemodel"


#load database
def load_db(db_name):
    lmdb_env = lmdb.open(db_name)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        print data

#learning and testing
def learn_and_test(solver_file):
    caffe.set_mode_gpu()
#    caffe.set_device(0)
    solver = caffe.get_solver(solver_file)
    #start DNN
    solver.solve()

def get_blob_info(net):
    #return blob's name and it's size
#    print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
    return [(k, v.data.shape) for k, v in net.blobs.items()]

def get_layer_name(net):
    #return layers' name
    n = 0
    layers = []
    temp = get_blob_info(net)
    for t in temp:
#        print '{}:{}'.format(n, t[0])
        layers.append(t[0])
        n = n + 1
    return layers

def get_layer_weight(net, layer_name):
    #return layer's weight 
    weights = net.params[layer_name][0]
    #print '{}:{}'.format(layer_name[num], weights)
    return weights

def get_layer_data(net, layer_name):
    #batch * channel * height * width
    data = net.blobs[layer_name].data
    return data

#net = caffe.Net(proto_file_name, caffe_model,  caffe.TEST)
net_init = caffe.Net(proto_file_name_init, caffe_model_init, caffe.TEST)

print get_blob_info(net_init)

#print get_layer_weight(net_init, get_layer_name(net_init)[3])
#print get_layer_name(net_init)[3]
#params = []
#params.append(get_layer_name(net_init)[7])

#for layer, layer_init in zip(params, params):
#    net.params[layer][0].data[0] = get_layer_weight(net_init, layer_init)

#print (get_layer_weight(net, params[0]))
#net.save(caffe_model)
