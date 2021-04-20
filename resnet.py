from __future__ import division

#Inspired from https://github.com/raghakot/keras-resnet
#and from https://github.com/suragnair/alpha-zero-general

import six
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    add,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2

import tensorflow_model_optimization as tfmot

import tflite_runtime.interpreter as tflite

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

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

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = input.shape
    residual_shape = residual.shape

    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def basic_residual_block(filters, init_strides=(1, 1)):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        conv1 = _conv_bn_relu(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)
        residual = _conv_bn_relu(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f

def bi_residual_block(filters, init_strides=(1,1)):
    """
    Insert description here.
    """
    def f(input1, input2):
        res1 = basic_residual_block(filters=filters, init_strides=init_strides)(input1)
        res2 = basic_residual_block(filters=filters, init_strides=init_strides)(input2)
        sum_res = add([res1, res2])
        #final_res = basic_residual_block(filters=filters, init_strides=init_strides)(sum_res)
        return sum_res

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    #if K.image_dim_ordering() == 'tf':
    #    ROW_AXIS = 1
    #    COL_AXIS = 2
    #    CHANNEL_AXIS = 3
    #else:
    #    CHANNEL_AXIS = 1
    #    ROW_AXIS = 2
    #    COL_AXIS = 3
    
    CHANNEL_AXIS = 3
    ROW_AXIS = 1
    COL_AXIS = 2

class ResNetBuilder():
    @staticmethod
    def build(input_shape, num_outputs, filters):
        """Builds a custom Tree ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final dense layer
            filters: The number of filters used in residual blocks
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()

        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        #if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

        in1 = Input(shape=input_shape)
        block1 = basic_residual_block(filters=filters, init_strides=(3,3))(in1)
        block2 = basic_residual_block(filters=filters)(block1)

        #Final part
        res_block = basic_residual_block(filters=filters)(block2)
        flatten1 = Flatten()(res_block)
        dense = Dense(units=256, activation="relu")(flatten1)
        out_value = Dense(units=num_outputs, activation="linear")(dense)

        model = Model(inputs=in1, outputs=out_value)
        
        return model

    @staticmethod
    def build_resnet_32(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, 32)

if __name__ == '__main__':
    model = ResNetBuilder.build_resnet_32((4,9,9), 1)
    #q_model = tfmot.quantization.keras.quantize_model(model)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.Mean()])

    a = np.zeros((4,9,9), dtype=np.float32)
    a = tf.reshape(a, (-1, 9, 9, 4))
    print(a.shape)


    start = time.time()
    print(model([a]))
    print("tf inference:{0} ms".format((time.time()-start)*1000))

    #model.save('tablut')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    quantized_tflite_model = converter.convert()

    with open("tablut.tflite", "wb") as f:
        f.write(quantized_tflite_model)

    interpreter = tflite.Interpreter("tablut.tflite", num_threads=2)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    n = 100
    sum = 0

    for i in range(n):

        startTime = time.time()
        interpreter.invoke()
        deltaTime = time.time()-startTime
        sum += deltaTime

    print("Inference time: {0} ms ({3} invokes), Input details: {1}, Output details: {2}".format((sum/n)*1000, input_details, output_details, n))