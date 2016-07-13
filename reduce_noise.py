#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import json
import math

import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers


if len(sys.argv) != 4:
    sys.exit("reduce_noise.py `path_to_model` `input_directory` `output_directory`")

gpu_device_id = 0
try:
    cuda.check_cuda_available()
    use_gpu = True
    cuda.get_device(gpu_device_id).use()
except:
    use_gpu = False
print("gpu_device_id: {0}".format(gpu_device_id))
print("use_gpu: {0}".format(use_gpu))

xp = cuda.cupy if use_gpu else np

# https://github.com/mrkn/chainer-waifu2x/blob/master/waifu2x.py
# MIT Lisence
# Author Kenta Murata

waifu2x_model_file = sys.argv[1]
with open(waifu2x_model_file) as fp:
    model_params = json.load(fp)

def make_Convolution2D(params):
    func = F.Convolution2D(
        params['nInputPlane'],
        params['nOutputPlane'],
        (params['kW'], params['kH'])
    )
    func.b = np.float32(params['bias'])
    func.W = np.float32(params['weight'])
    if use_gpu:
        func.b = cuda.to_gpu(func.b)
        func.W = cuda.to_gpu(func.W)
    return func

model = chainer.Chain()
for i, layer_params in enumerate(model_params):
    function = make_Convolution2D(layer_params)
    setattr(model, "conv{}".format(i + 1), function)

steps = len(model_params)

def forward(x):
    h = x
    for i in range(1, steps):
        key = 'conv{}'.format(i)
        h = F.leaky_relu(getattr(model, key)(h), 0.1)
    key = 'conv{}'.format(steps)
    y = getattr(model, key)(h)
    return y
    
def scale_image(image, block_offset, block_size=128):
    x_data = np.asarray(image).transpose(2, 0, 1).astype(np.float32)
    x_data /= 255

    output_size = block_size - block_offset * 2

    h_blocks = int(math.floor(x_data.shape[1] / output_size)) + (0 if x_data.shape[1] % output_size == 0 else 1)
    w_blocks = int(math.floor(x_data.shape[2] / output_size)) + (0 if x_data.shape[2] % output_size == 0 else 1)

    h = block_offset + h_blocks * output_size + block_offset
    w = block_offset + w_blocks * output_size + block_offset
    pad_h1 = block_offset
    pad_w1 = block_offset
    pad_h2 = (h - block_offset) - x_data.shape[1]
    pad_w2 = (w - block_offset) - x_data.shape[2]

    x_data = np.pad(x_data, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)), 'edge')
    result_data = np.zeros_like(x_data)

    for i in range(0, x_data.shape[1], output_size):
        if i + block_size > x_data.shape[1]:
            continue
        for j in range(0, x_data.shape[2], output_size):
            if j + block_size > x_data.shape[2]:
                continue
            block = x_data[:, i:(i + block_size), j:(j + block_size)]
            block = np.reshape(block, (1,) + block.shape)
            if use_gpu:
                block = cuda.to_gpu(block)
            x = chainer.Variable(block)
            y = forward(x)
            if use_gpu:
                y_data = cuda.to_cpu(y.data)[0]
            else:
                y_data = y.data[0]
                
            result_data[
                    :,
                    (i + block_offset):(i + block_offset + output_size),
                    (j + block_offset):(j + block_offset + output_size)
                ] = y_data
                
    result_data = result_data[
            :,
            (pad_h1 + 1):(result_data.shape[1] - pad_h2 - 1),
            (pad_w1 + 1):(result_data.shape[2] - pad_w2 - 1)]
            
    result_data[result_data < 0] = 0
    result_data[result_data > 1] = 1
    result_data *= 255

    result_image = Image.fromarray(np.uint8(result_data).transpose(1, 2, 0))
    return result_image

if __name__ == '__main__':
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    print("input dir: {0}".format(input_dir))
    print("output dir: {0}".format(output_dir))
    list = os.listdir(input_dir)
    for (i, file) in enumerate(sorted(list)):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, os.path.splitext(file)[0] + ".tiff")
        print("Processing [{0}/{1}] {2} -> {3}".format(i+1, len(list), input_file, output_file))
        image = Image.open(input_file).convert('RGB')
        scaled_image = scale_image(image, steps)
        scaled_image.save(output_file)
        print("Done.")
        