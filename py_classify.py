# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import sys,os

caffe_root = '/home/lov1n/Downloads/caffe-master/'
sys.path.insert(0,caffe_root + 'python')

import caffe
os.chdir(caffe_root)

net_file = caffe_root + 'examples/myexamples/modelornot2/bvlc_reference_caffenet/deploy.prototxt'
caffe_model = caffe_root + 'examples/myexamples/modelornot2/_iter_1000.caffemodel'
mean_file = caffe_root + 'examples/myexamples/modelornot2/mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))    #将图像通道数设为outermost的维数
transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

# net.blobs['data'].reshape(50,3,227,227)


count =0
data_csv = pd.read_csv('examples/myexamples/modelornot2/image_100_to_verify.csv')
image_frame = data_csv['imageName']
model_frame = data_csv['modelID']
for i in range(len(data_csv)):
    im = caffe.io.load_image(caffe_root + 'examples/myexamples/modelornot2/verify/' + image_frame[i])
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    output = net.forward()
    out_prob = output['prob'][0]
    print image_frame[i],model_frame[i],'predicted class is：',out_prob.argmax()
    if model_frame[i] == out_prob.argmax():
        count += 1

print count