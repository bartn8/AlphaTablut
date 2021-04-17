from network import TablutNNet
from actionbuffer import ActionBuffer
from selfplay import SelfPlay
from tablut import TablutConfig

import ray
import time
import os
import argparse


class AlphaTablut:

    def __init__(self):
        self.config = TablutConfig()
        self.action_buffer = ActionBuffer(config.observation_shape)
        self.nnet = TablutNNet()

    def load_actionbuffer(self):
        self.action_buffer.load_buffer(self.config.folder, self.config.action_buffer_name)

    def load_checkpoint(self):
        self.nnet.load_checkpoint(self.config.folder, self.config.checkpoint_name)

    def save_tflite(self):
        self.nnet.tflite_optimization(self.config.folder, self.config.checkpoint_name)

# Menu: Train, load pre-trained, play, self-play

# Ray Workers:
#   1) Selplayer
#   2) Trainer
#   3) ActionBuffer Explorer
# Start ray
ray.init()


@ray.remote
def self_play_worker(action_buffer):
    pass


@ray.remote
def trainer_worker(action_buffer):
    pass


def menu_selfplay():


def repl(args):

    #Istanzio l'oggetto che gestisce il tutto
    tablut = AlphaTablut()

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model",
            "Play against AlphaTablut",
            "AlphaTablut Self-Play",
            "Exit",
        ]
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)

        if choice == 0:
            menu_train()
        elif choice == 1:
            menu_load()
        elif choice == 2:
            menu_play()
        elif choice == 3:
            menu_selfplay()
        else:
            break

    print("\nDone")


def main():
    argparser = argparse.ArgumentParser(
        description='AlphaTablut')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    repl(args)


if __name__ == '__main__':
    main()
