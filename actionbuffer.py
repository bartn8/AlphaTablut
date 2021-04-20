import numpy as np
import pickle
import threading
import os

import tensorflow as tf

CURRENT_STATE = 'current_state'
VISIT_COUNT = 'visit_count'
REWARD = 'reward'
WHITE_WINS = 'white_wins'
BLACK_WINS = 'black_wins'
DRAWS = 'draws'

class ActionBuffer:

    def __init__(self, config):
        self.buffer = {}
        #self._lock = threading.Lock()
        self.config = config
        self.game_counter = 0

    def increment_game_counter(self):
        self.game_counter += 1

    def size(self):
        return len(self.buffer)

    def store_action(self, board, reward, weight):
        """
        board: current state of the game
        reward: 1 if white wins, -1 if black wins, 0 otherwise
        weight: [0,1] reward update weight
        """
        if (board.shape) != (self.config.observation_shape):
            raise Exception("Board shape mismatch")

        action_hash = hash(board.tobytes())

        # with self._lock:
        if action_hash in self.buffer:
            action = self.buffer[action_hash]
            action[VISIT_COUNT] += 1
            action[REWARD] = action[REWARD] * \
                (1 - float(weight)) + float(reward) * float(weight)
            if reward == 1:
                action[WHITE_WINS] += 1
            elif reward == -1:
                action[BLACK_WINS] += 1
            else:
                action[DRAWS] += 1
        else:
            action = {CURRENT_STATE: board.copy(), VISIT_COUNT: 1, REWARD: float(
                reward), WHITE_WINS: 0, BLACK_WINS: 0, DRAWS: 0}

            if reward == 1:
                action[WHITE_WINS] += 1
            elif reward == -1:
                action[BLACK_WINS] += 1
            else:
                action[DRAWS] += 1

            self.buffer[action_hash] = action

    def generate_dataset(self, batch_size):
        keys = len(self.buffer)
        batch_size = min(batch_size, keys)

        # X Data
        board1 = np.zeros(
            (keys, self.config.observation_shape[1], self.config.observation_shape[2], self.config.observation_shape[3]), dtype=np.float32)

        # Y Data
        values = np.zeros((keys, 1), dtype=np.float32)

        # Itero le chiavi del buffer
        i = 0
        for key in self.buffer:
            # Imposto i dati nei data
            board1[i] = self.buffer[key][CURRENT_STATE]
            values[i] = self.buffer[key][REWARD]
            i += 1

        def generator():
            for s1, l in zip(board1, values):
                yield {"input_1": s1}, l

        dataset = tf.data.Dataset.from_generator(generator, output_types=(
            {"input_1": tf.float32}, tf.float32))

        dataset = dataset.shuffle(keys)
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
