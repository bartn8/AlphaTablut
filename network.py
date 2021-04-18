import os
import sys
import time
import json

import numpy as np
import tensorflow as tf

from treeresnet import TreeResNetBuilder

from tablut import TablutConfig

# https://github.com/suragnair/alpha-zero-general


class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, config):
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


class TreeResNNet(NeuralNet):
    def __init__(self, config, restoreFromCheckpoint=False):
        self.config = config

        if restoreFromCheckpoint:
            self.load_checkpoint()
        else:
            self.nnet = TreeResNetBuilder.build(
                self.config.observation_shape, 1, self.config.num_filters)

            self.nnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr_init,
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.Mean()])

            self.training_steps=0

    def train(self, data_function):
        """
        data_function: function that generate a dataset of examples, each example is of form (board0, board1, v)
        """

        print("Training step: {0}".format(self.training_steps))
        self.history=self.nnet.fit(data_function(
            self.config.batch_size), epochs=self.config.epochs+self.training_steps, initial_epoch=self.training_steps)

        self.training_steps += self.config.epochs

        if self.training_steps % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
            self.tflite_optimization()

        return history

    def predict(self, board0, board1):
        """
        board0: np array with current board
        board1: np array with next board
        """

        # preparing input
        board0=tf.reshape(
            board0, (-1, self.config.observation_shape[1], self.config.observation_shape[2], self.config.observation_shape[0]))
        board1=tf.reshape(
            board1, (-1, self.config.observation_shape[1], self.config.observation_shape[2], self.config.observation_shape[0]))

        # run
        v=self.nnet([board0, board1])
        v=tf.reshape(v, (-1))

        return v[0]

    def predicts(self, boards0, boards1):
        """
        boards0: np array of a batch of current board
        boards1: np array of a batch of next board
        """

        if boards0.shape[0] != boardss1.shape[0]:
            raise Exception("Batch elements must be the same")


        # preparing input
        boards0=tf.reshape(boards0, (boards0.shape[0], self.config.observation_shape[1],
                           self.config.observation_shape[2], self.config.observation_shape[0]))
        boards1=tf.reshape(boards1, (boards1.shape[0], self.config.observation_shape[1],
                           self.config.observation_shape[2], self.config.observation_shape[0]))

        v=self.nnet.predict([boards0, boards1])

        return tf.reshape(v, (-1))


    def save_checkpoint(self):
        folder=self.config.folder
        filename=self.config.checkpoint_name
        filename_meta = self.config.checkpoint_metadata
        filepath=os.path.join(folder, filename)
        filepath_meta=os.path.join(folder, filename)

        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        if self.nnet is not None:
            self.nnet.save(filepath)

            data={'training_steps': self.training_steps,
                'history': self.history, 'config': self.config}

            with open(filepath_meta, "w") as f:
                json.dump(data, f)


    def load_checkpoint(self):
        folder=self.config.folder
        filename=self.config.checkpoint_name
        filename_meta = self.config.checkpoint_metadata
        filepath=os.path.join(folder, filename)
        filepath_meta=os.path.join(folder, filename)

        self.nnet=tf.keras.models.load_model(filepath)

        with open(filepath_meta, "r") as f:
            data=json.load(f)
            self.training_steps=data['training_steps']
            self.config=data['config']
            self.history=data['history']


    def tflite_optimization(self):
        folder=self.config.folder
        filename=self.config.tflite_model
        filepath=os.path.join(folder, filename)

        converter=tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types=[tf.float16]
        quantized_tflite_model=converter.convert()

        with open(filepath, "wb") as f:
            f.write(quantized_tflite_model)
