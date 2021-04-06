
import os
import sys
import time

import numpy as np

import tensorflow as tf

from treeresnet import TreeResNetBuilder

import json

#https://github.com/suragnair/alpha-zero-general

args = dotdict({
    'lr': 0.003,
    'training_steps': 10000,
    'epochs': 10,
    'batch_size': 64,
    'checkpoint_interval': 50,
    'num_filters': 32
})


class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, game):
        pass

    def train(self, data_function):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            data_function:  function that generate a dataset of examples,
                      each example is of form (board0, board1, v).
                      v is its value. The examples has
                      board in its canonical form.
        """
        pass

    def predict(self, board0, board1):
        """
        Input:
            board0: current board in its canonical form.
            board1: next board in its canonical form.

        Returns:
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass

    def tflite_optimization(self, filename):
        """
        Optimize current neural network in TfLite format.
        """
        pass


class TablutNNet(NeuralNet):
    def __init__(self, game, restoreFromCheckpoint = False):
        self.input_shape = game.input_shape()

        if restoreFromCheckpoint:
            self.load_checkpoint()
        else:

            self.nnet = TreeResNetBuilder.build(self.input_shape, 1, args.num_filters)

            self.nnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.Mean()])

            self.training_steps = 0

    def train(self, data_function):
        """
        data_function: function that generate a dataset of examples, each example is of form (board0, board1, v)
        """

        print("Training step: {0}".format(self.training_steps))
        history = self.nnet.fit(data_function(args.batch_size), epochs=args.epochs, initial_epoch=self.training_steps)

        self.training_steps += args.epochs

        if self.training_steps % args.checkpoint_interval == 0:
            self.save_checkpoint()
            self.tflite_optimization()

        return history

    def predict(self, board0, board1):
        """
        board0: np array with current board
        board1: np array with next board
        """

        #preparing input
        board0 = tf.reshape(board0, (-1, self.input_shape[1], self.input_shape[2], self.input_shape[0]))
        board1 = tf.reshape(board1, (-1, self.input_shape[1], self.input_shape[2], self.input_shape[0]))

        # run
        v = self.nnet([board0, board1])
        v = tf.reshape(v, (-1))
        
        return v[0]

    def predicts(self, boards0, boards1):
        """
        boards0: np array of a batch of current board
        boards1: np array of a batch of next board
        """

        if boards0.shape[0] != boardss1.shape[0]:
            raise Exception("Batch elements must be the same")


        #preparing input
        boards0 = tf.reshape(boards0, (boards0.shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[0]))
        boards1 = tf.reshape(boards1, (boards1.shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[0]))

        v = self.nnet.predict([boards0, boards1])

        return tf.reshape(v, (-1))


    def save_checkpoint(self, folder='checkpoint', filename='tablut'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        
        if self.nnet is not None:
            self.nnet.save(filepath)

            data = {'training_steps': self.training_steps, 'input_shape': self.input_shape}

            with open(os.path.join(folder, filename+".data"), "w") as f:
                json.dump(data, f)
            
            
    def load_checkpoint(self, folder='checkpoint', filename='tablut'):
        filepath = os.path.join(folder, filename)
        
        self.nnet = tf.keras.models.load_model(filepath)

        with open(os.path.join(folder, filename+".data"), "r") as f:
            data = json.load(f)
            self.training_steps = data['training_steps']
            self.input_shape = data['input_shape']


    def tflite_optimization(self, filename='tablut.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quantized_tflite_model = converter.convert()

        with open(filename, "wb") as f:
            f.write(quantized_tflite_model)