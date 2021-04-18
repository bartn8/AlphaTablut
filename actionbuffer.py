import numpy as np
import pickle
import threading

import tensorflow as tf


class ActionBuffer:

    CURRENT_STATE = 'current_state'
    NEXT_STATE = 'next_state'
    VISIT_COUNT = 'visit_count'
    REWARD = 'reward'
    WHITE_WINS = 'white_wins'
    BLACK_WINS = 'black_wins'
    DRAWS = 'draws'

    def __init__(self, config):
        self.buffer = {}
        self._lock = threading.Lock()
        self.config = config

    def store_action(self, board0, board1, reward, weight):
        """
        board0: current state of the game
        board1: next state of the game
        reward: 1 if white wins, -1 if black wins, 0 otherwise
        weight: [0,1] reward update weight
        """
        if (board0.shape, board1.shape) != (self.config.observation_shape, self.config.observation_shape):
            raise Exception("Board shape mismatch")

        if np.array_equal(board0, board1):
            pass

        action_hash = hash(board0.tobytes()+board1.tobytes())

        with self._lock:
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
                action = {CURRENT_STATE: board0.copy(), NEXT_STATE: board1.copy(
                ), VISIT_COUNT: 1, REWARD: float(reward), WHITE_WINS: 0, BLACK_WINS: 0, DRAWS: 0}

                if reward == 1:
                    action[WHITE_WINS] += 1
                elif reward == -1:
                    action[BLACK_WINS] += 1
                else:
                    action[DRAWS] += 1

                self.buffer[action_hash] = action

    def generate_dataset(self, batch_size):
        keys = len(self.buffer)

        # X Data
        board0 = np.zeros(
            (keys, self.config.observation_shape[1], self.config.observation_shape[2], self.config.observation_shape[3]), dtype=np.float32)
        board1 = np.zeros(
            (keys, self.config.observation_shape[1], self.config.observation_shape[2], self.config.observation_shape[3]), dtype=np.float32)

        # Y Data
        values = np.zeros((keys), dtype=np.float32)

        # Itero le chiavi del buffer
        i = 0
        for key in self.buffer:
            # Imposto i dati nei data
            board0[i] = self.buffer[key][CURRENT_STATE]
            board1[i] = self.buffer[key][NEXT_STATE]
            values[i] = self.buffer[key][REWARD]

        def generator():
            for s1, s2, l in zip(sent1, sent2, labels):
                yield {"input_1": s1, "input_2": s2}, l

        #tf.data.Dataset.from_tensor_slices((board0, board1, values))
        dataset = tf.data.Dataset.from_generator(generator, output_types=(
            {"input_1": tf.float32, "input_2": tf.float32}, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset

    def save_buffer(self):
        folder = self.config.folder
        filename = self.config.action_buffer_name
        filepath = os.path.join(folder, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump([self.buffer, self.input_shape], f)

    def load_buffer(self):
        folder = self.config.folder
        filename = self.config.action_buffer_name
        filepath = os.path.join(folder, filename)

        with open(filepath, "rb") as f:
            self.buffer, self.input_shape = pickle.load(f)
