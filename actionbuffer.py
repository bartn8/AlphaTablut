import numpy as np
import pickle
import threading
import os
import random
from tqdm import tqdm
from hashlib import md5

from tablutconfig import TablutConfig

import tensorflow as tf
import tflite_runtime.interpreter as tflite

CURRENT_STATE = 'current_state'
VISIT_COUNT = 'visit_count'
REWARD = 'reward'
WHITE_WINS = 'white_wins'
BLACK_WINS = 'black_wins'
DRAWS = 'draws'


class ActionBuffer:

    def __init__(self, config):
        self.buffer = {}
        self.onetimebuffer = {}
        self.config = config
        self.game_counter = 0

    def increment_game_counter(self):
        self.game_counter += 1

    def size(self):
        return len(self.buffer)

    def store_action(self, board, reward, weight, utility):
        """
        board: current state of the game
        reward: 1 if white wins, -1 if black wins, 0 otherwise
        weight: [0,1] reward update weight
        """
        if (board.shape) != (self.config.observation_shape):
            raise Exception("Board shape mismatch")

        action_hash = md5(str(board.astype(np.int8).tolist()).encode('utf-8')).hexdigest()

        if action_hash in self.buffer:
            action = self.buffer[action_hash]
            action[VISIT_COUNT] += 1
            action[REWARD] = action[REWARD] * \
                (1 - weight*0.5) + float(reward) * weight * 0.5
            if utility == 1:
                action[WHITE_WINS] += 1
            elif utility == -1:
                action[BLACK_WINS] += 1
            else:
                action[DRAWS] += 1

            if action_hash in self.onetimebuffer:
                del self.onetimebuffer[action_hash]

        else:
            if self.size() > self.config.action_buffer_maxsize:
                # Ricerco uno stato da eliminare
                validkeys = self.onetimebuffer.keys()

                if len(validkeys) == 0:
                    return

                delkey = random.choice(list(validkeys))
                del self.buffer[delkey]
                del self.onetimebuffer[delkey]

            action = {CURRENT_STATE: board.copy(), VISIT_COUNT: 1, REWARD: float(
                reward), WHITE_WINS: 0, BLACK_WINS: 0, DRAWS: 0}

            if utility == 1:
                action[WHITE_WINS] += 1
            elif utility == -1:
                action[BLACK_WINS] += 1
            else:
                action[DRAWS] += 1

            self.buffer[action_hash] = action

            if -self.config.action_buffer_trim_th < reward and reward < self.config.action_buffer_trim_th:
                self.onetimebuffer[action_hash] = action

    def generate_dataset(self, batch_size):
        keys = len(self.buffer)
        batch_size = min(batch_size, keys)

        # X Data
        board1 = np.zeros(
            (batch_size, self.config.observation_shape[1], self.config.observation_shape[2], self.config.observation_shape[3]), dtype=np.float32)

        # Y Data
        values = np.zeros((batch_size, 1), dtype=np.float32)

        # Itero le chiavi del buffer
        # Ne seleziono a caso pari al batch_size
        k = 0
        selected_keys = []
        all_keys = list(self.buffer.keys())
        while k < batch_size:
            key = random.choice(all_keys)
            if key not in selected_keys:
                selected_keys.append(key)
                k += 1

        i = 0
        for key in selected_keys:
            # Imposto i dati nei data
            board1[i] = self.buffer[key][CURRENT_STATE]
            values[i] = self.buffer[key][REWARD]
            i += 1

        def generator():
            for s1, l in zip(board1, values):
                yield {"input_1": s1}, l

        dataset = tf.data.Dataset.from_generator(generator, output_types=(
            {"input_1": tf.float32}, tf.float32))

        #dataset = dataset.shuffle(keys)
        dataset = dataset.batch(batch_size)
        return dataset

    def save_buffer(self):
        folder = self.config.folder
        filename = self.config.action_buffer_name
        filepath = os.path.join(folder, filename)
        self.save_buffer_to(filepath)

    def save_buffer_to(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump([self.buffer, self.game_counter, self.config], f)

    def load_buffer(self):
        folder = self.config.folder
        filename = self.config.action_buffer_name
        filepath = os.path.join(folder, filename)
        self.load_buffer_from(filepath)

    def load_buffer_from(self, filepath):
        config = self.config

        with open(filepath, "rb") as f:
            self.buffer, self.game_counter, config = pickle.load(f)

        if config.observation_shape != self.config.observation_shape:
            raise Exception("Observation shape dismatch!")

        # onetimebuffer rebuilding.
        self.onetimebuffer = {}

        for key in self.buffer:
            if self.buffer[key][VISIT_COUNT] <= 1 and -self.config.action_buffer_trim_th < self.buffer[key][REWARD] and self.buffer[key][REWARD] < self.config.action_buffer_trim_th:
                self.onetimebuffer[key] = self.buffer[key]

    def remove_duplicate(self):
        keys = list(self.buffer.keys())
        newbuffer = {}

        for i in tqdm(range(len(keys))):
            key = keys[i]
            a = self.buffer[key]
            newkey = md5(str(a[CURRENT_STATE].astype(np.int8).tolist()).encode('utf-8')).hexdigest()

            if newkey not in newbuffer:
                newbuffer[newkey] = a
            else:
                b = newbuffer[newkey]
                action = {CURRENT_STATE: a[CURRENT_STATE].copy(), VISIT_COUNT: a[VISIT_COUNT] + b[VISIT_COUNT], REWARD: float(
                            0.5 * a[REWARD] + 0.5 * b[REWARD]), WHITE_WINS: a[WHITE_WINS] + b[WHITE_WINS], BLACK_WINS: a[BLACK_WINS] + b[BLACK_WINS], DRAWS: a[DRAWS] + b[DRAWS]}
                newbuffer[newkey] = action

        self.buffer = newbuffer

    def trim(self):
        for key in self.onetimebuffer:
            del self.buffer[key]

        self.onetimebuffer = {}


if __name__ == '__main__':
    config = TablutConfig()
    buf = ActionBuffer(config)
    buf.load_buffer()
    #buf.trim()
    #buf.remove_duplicate()
    #buf.save_buffer()
    batch_size = config.batch_size
    batch_size = min(batch_size, buf.size())
    #dataset = buf.generate_dataset(batch_size)

    print(buf.game_counter)

    maxVisit = 0
    maxVisitValue = None
    maxVisit2 = 0
    maxVisitValue2 = None
    for key in buf.buffer:
        if buf.buffer[key][VISIT_COUNT] > maxVisit:
            maxVisit = buf.buffer[key][VISIT_COUNT]
            maxVisitValue = buf.buffer[key]
        if buf.buffer[key][VISIT_COUNT] > maxVisit2:
            if buf.buffer[key][VISIT_COUNT] < maxVisit:
                maxVisit2 = buf.buffer[key][VISIT_COUNT]
                maxVisitValue2 = buf.buffer[key]

    #maxVisitValue = buf.buffer[random.choice(list(buf.buffer.keys()))]
    board = maxVisitValue[CURRENT_STATE]
    print(md5(str(board.astype(np.int8).tolist()).encode('utf-8')).hexdigest())
    #board[0,0,2,2] = 0
    #board[0,8,6,2] = 1

    print(np.moveaxis(maxVisitValue[CURRENT_STATE], -1, 0))
    print(maxVisitValue[REWARD])
    print(maxVisitValue[VISIT_COUNT])

    print(np.moveaxis(maxVisitValue2[CURRENT_STATE], -1, 0))
    print(maxVisitValue2[REWARD])
    print(maxVisitValue2[VISIT_COUNT])

    #folder = config.folder
    #filename = config.tflite_model
    #filepath = os.path.join(folder, filename)
    #interpreter = tflite.Interpreter(filepath, num_threads=2)
    # interpreter.allocate_tensors()

    # Get input and output tensors.
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()

    # Test the model on random input data.
    #input_shape = input_details[0]['shape']
    # interpreter.set_tensor(
    #    input_details[0]['index'], maxVisitValue[CURRENT_STATE])

    # interpreter.invoke()

    # print(interpreter.get_tensor(output_details[0]['index']))
