#!python
# Strumento per l'estrazione di dati dal dataset Tablut Challange.

import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
from actionbuffer import ActionBuffer
from tablutconfig import TablutConfig

DTYPE = np.float32

# Lettura del dataset

argparser = argparse.ArgumentParser(
    description='TablutScraper')

argparser.add_argument(
    '-f', '--fromPath',
    metavar='F',
    type=str,
    help='Directory with dataset')

argparser.add_argument(
    '-t', '--toPath',
    metavar='T',
    type=str,
    help='Actionbuffer to store data')

args = argparser.parse_args()

dirPath = args.fromPath
actionBufferPath = args.toPath

datatxts = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

print("Found {0} dataset files".format(len(datatxts)))
print("Loading ActionBuffer...")

config = TablutConfig()

buffer = ActionBuffer(config)
buffer.load_buffer_from(actionBufferPath)

print("Done. Start scraping...")

whiteConstraints = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

for datatxt in datatxts:
    print("Start scraping {0}...".format(datatxt))

    try:
        with open(join(dirPath, datatxt), "r") as f:
            line = "_"
            prevTurn = "B"
            reward = 0
            gamehistory = []

            while line:
                while line != "FINE: Stato:":
                    line = f.readline().strip()
                    if not line:
                        print("EOF")
                        break

                state = np.zeros(config.observation_shape, dtype=DTYPE)

                for row in range(9):
                    line = f.readline().strip()
                    for i, char in enumerate(line):
                        if char == 'W':
                            state[0, row, i, 0] = 1
                        elif char == 'B':
                            state[0, row, i, 1] = 1
                        elif char == 'K':
                            state[0, row, i, 2] = 1

                state[0, :, :, 3] = whiteConstraints

                gamehistory.append(state)

                line = f.readline().strip()
                if line != '-':
                    raise Exception("{0} Unexcepted. Excepted: -".format(line))

                line = f.readline().strip()

                if line == 'WW':
                    reward = 1
                    break
                elif line == 'BW':
                    reward = -1
                    break
                elif line == 'D':
                    reward = 0
                    break
                elif line == 'W':
                    if prevTurn == line:
                        raise Exception("Same turn")
                    prevTurn = line
                elif line == 'B':
                    if prevTurn == line:
                        raise Exception("Same turn")
                    prevTurn = line
                else:
                    raise Exception(
                        "{0} Unexcepted. Excepted: WW, BW, D, W, B".format(line))

        print("Moves: {0}. Reward: {1}".format(len(gamehistory), reward))

        if reward != 0:

            k = 0
            i = len(gamehistory)-1
            while i >= 0:
                board1 = gamehistory[i]
                buffer.store_action(
                    board1, reward/(1+(9*k/config.max_moves)), 1, reward)
                i -= 1
                k += 1

            buffer.increment_game_counter()

        print("Action buffer updated. Total states: {0} Total games: {1}".format(
            buffer.size(), buffer.game_counter))

    except Exception as e:
        print(e)

print("Trimming actionbuffer...")
buffer.trim()
print("Action buffer updated. Total states: {0} Total games: {1}".format(
    buffer.size(), buffer.game_counter))
print("Saving actionbuffer...")
buffer.save_buffer_to(actionBufferPath)
print("Done.")
