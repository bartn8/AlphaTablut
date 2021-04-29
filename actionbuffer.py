import numpy as np
import pickle
import threading
import os
import random

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

        action_hash = hash(board.tobytes())

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
        else:
            if self.size() > self.config.action_buffer_maxsize:
                #Ricerco uno stato da eliminare
                validkeys = []
                for key in self.buffer:
                    if self.buffer[key][VISIT_COUNT] == 1:
                        validkeys.append(key)

                delkey = random.choice(validkeys)
                del self.buffer[delkey]

            action = {CURRENT_STATE: board.copy(), VISIT_COUNT: 1, REWARD: float(
                reward), WHITE_WINS: 0, BLACK_WINS: 0, DRAWS: 0}

            if utility == 1:
                action[WHITE_WINS] += 1
            elif utility == -1:
                action[BLACK_WINS] += 1
            else:
                action[DRAWS] += 1

            self.buffer[action_hash] = action

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
                k+=1

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

        with open(filepath, "wb") as f:
            pickle.dump([self.buffer, self.game_counter, self.config], f)

    def load_buffer(self):
        folder = self.config.folder
        filename = self.config.action_buffer_name
        filepath = os.path.join(folder, filename)

        config = self.config

        with open(filepath, "rb") as f:
            self.buffer, self.game_counter, config = pickle.load(f)

        if config.observation_shape != self.config.observation_shape:
            raise Exception("Observation shape dismatch!")


if __name__ == '__main__':
    config = TablutConfig()
    buf = ActionBuffer(config)
    buf.load_buffer()
    batch_size = config.batch_size
    batch_size = min(batch_size, buf.size())
    dataset = buf.generate_dataset(batch_size)

    maxVisit = 0
    maxVisitValue = None
    #for key in buf.buffer:
    #    if buf.buffer[key][WHITE_WINS] > maxVisit:
    #        maxVisit = buf.buffer[key][WHITE_WINS]
    #        maxVisitValue = buf.buffer[key]

    maxVisitValue = buf.buffer[random.choice(list(buf.buffer.keys()))]
    board = maxVisitValue[CURRENT_STATE]
    #board[0,0,2,2] = 0
    #board[0,8,6,2] = 1

    print(np.moveaxis(maxVisitValue[CURRENT_STATE], -1, 0))
    print(maxVisitValue[REWARD])

    folder = config.folder
    filename = config.tflite_model
    filepath = os.path.join(folder, filename)
    interpreter = tflite.Interpreter(filepath, num_threads=2)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], maxVisitValue[CURRENT_STATE])


    interpreter.invoke()

    print(interpreter.get_tensor(output_details[0]['index']))
