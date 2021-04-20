import os
import sys
import time
import pickle
import logging

import numpy as np
import tensorflow as tf

from resnet import ResNetBuilder
from tablut import TablutConfig
from actionbuffer import ActionBuffer

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


class ResNNet(NeuralNet):
    def __init__(self, config, restoreFromCheckpoint=False):
        self.config = config
        self.history = None

        if restoreFromCheckpoint:
            self.load_checkpoint()
        else:
            self.nnet = ResNetBuilder.build(
                self.config.network_input_shape, 1, self.config.num_filters)

            self.nnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr_init),
                              loss=tf.keras.losses.MeanSquaredError())

            self.training_steps = 0

    def train(self, data):
        """
        data: dataset of examples, each example is of form (board0, board1, v)
        """

        logging.info("Training step: {0}".format(self.training_steps))
        history_callback = self.nnet.fit(
            data, epochs=self.config.epochs+self.training_steps, initial_epoch=self.training_steps, verbose=1)

        self.history = history_callback.history["loss"]

        self.training_steps += self.config.epochs

        return self.history

    def predict(self, board0):
        """
        board0: np array with current board
        board1: np array with next board
        """
        # run
        v = self.nnet([board0])
        return v[0][0]

    def predicts(self, boards0):
        """
        boards0: np array of a batch of current board
        boards1: np array of a batch of next board
        """
        v = self.nnet.predict([boards0])
        return tf.reshape(v, (-1))

    def save_checkpoint(self):
        folder = self.config.folder
        filename = self.config.checkpoint_name
        filename_meta = self.config.checkpoint_metadata
        filepath = os.path.join(folder, filename)
        filepath_meta = os.path.join(folder, filename_meta)

        if not os.path.exists(folder):
            logging.info(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            logging.info("Checkpoint Directory exists! ")

        if self.nnet is not None:
            self.nnet.save(filepath)

            with open(filepath_meta, "wb") as f:
                pickle.dump([self.config, self.training_steps], f)

    def load_checkpoint(self):
        folder = self.config.folder
        filename = self.config.checkpoint_name
        filename_meta = self.config.checkpoint_metadata
        filepath = os.path.join(folder, filename)
        filepath_meta = os.path.join(folder, filename_meta)

        self.nnet = tf.keras.models.load_model(filepath)

        with open(filepath_meta, "rb") as f:
            config, self.training_steps = pickle.load(f)
        
        if config.observation_shape != self.config.observation_shape:
            raise Exception("Observation shape dismatch!")

    def tflite_optimization(self):
        folder = self.config.folder
        filename = self.config.tflite_model
        filepath = os.path.join(folder, filename)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.nnet)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quantized_tflite_model = converter.convert()

        with open(filepath, "wb") as f:
            f.write(quantized_tflite_model)


if __name__ == '__main__':
    config = TablutConfig()
    buf = ActionBuffer(config)
    buf.load_buffer()
    batch_size = config.batch_size
    batch_size = min(batch_size, buf.size())
    dataset = buf.generate_dataset(batch_size)

    net = ResNNet(config)
    net.train(dataset)

    board = np.zeros((1, 9, 9, 4), dtype=np.int8)

    board[0, :, :, 0] = np.array(
                            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 1, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

    board[0, :, :, 1] = np.array(
                            [[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 0, 0, 0, 0, 0, 1, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

    board[0, :, :, 2] = np.array(
                            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

    board[0, :, :, 3] = np.array(
                            [[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

    next_board = board.copy()
    next_board[0,3,4,0] = 0
    next_board[0,3,0,0] = 1

    print(net.predict(next_board))

