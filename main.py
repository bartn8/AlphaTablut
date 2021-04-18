from network import TreeResNNet
from actionbuffer import ActionBuffer
from selfplay import SelfPlay
from tablut import AshtonTablut, TablutConfig, Search, OldSchoolHeuristicFunction, NeuralHeuristicFunction, MixedHeuristicFunction

import ray
import time
import os
import argparse
import logging


class AlphaTablut:

    def __init__(self):
        self.config = TablutConfig()
        self.action_buffer = ActionBuffer(self.config)
        self.nnet = TreeResNNet(self.config)
    
    def check_saving_folder(self):
        folder = self.config.folder

        if not os.path.exists(folder):
            os.mkdir(folder)

    def check_saved_actionbuffer(self):
        folder = self.config.folder
        filename = self.config.action_buffer_name
        filepath = os.path.join(folder, filename)

        return os.path.exists(filepath) and os.path.isfile(filepath)
    
    def check_saved_checkpoint(self):
        folder=self.config.folder
        filename=self.config.checkpoint_name
        filename_meta = self.config.checkpoint_metadata
        filepath=os.path.join(folder, filename)
        filepath_meta=os.path.join(folder, filename)

        return os.path.exists(folder) and os.path.isdir(folder) and os.path.exists(filepath) and os.path.isfile(filepath_meta)

    def check_saved_tflite_model(self):
        folder=self.config.folder
        filename=self.config.tflite_model
        filepath=os.path.join(folder, filename)

        return os.path.exists(filepath) and os.path.isfile(filepath)

    def get_neural_heuristic(self):
        if self.check_saved_tflite_model():
            return NeuralHeuristicFunction(self.config)

        return None

    def get_mixed_heuristic(self, alpha):
        if self.check_saved_tflite_model():
            return MixedHeuristicFunction(self.config, alpha)

        return None

# Menu: Train, load pre-trained, play, self-play

# Ray Workers:
#   1) Selplayer
#   2) Trainer
#   3) ActionBuffer Explorer
# Start ray
ray.init()


@ray.remote
def self_play_worker(tablut):
    pass


@ray.remote
def trainer_worker(tablut):
    pass

def menu_train(tablut):
    print("Not implemented yet.")


def menu_load(tablut):
    if tablut.check_saved_checkpoint():
        print("Saved checkpoint found.")
        print("Loading checkpoint...")
        tablut.nnet.load_checkpoint()
    else:
        print("No checkpoint found")
    
    if tablut.check_saved_actionbuffer():
        print("Saved checkpoint found.")
        print("Loading action buffer...")
        tablut.action_buffer.load_buffer()
    else:
        print("No actionbuffer found")

    print("Done.")

def menu_save_tflite(tablut):
    print("Saving tflite model")
    tablut.nnet.tflite_optimization()
    print("Done.")


def menu_play(tablut):
    time = input("Insert AlphaTablut Search time in seconds: ")
    time = int(time)

    player = input("Choose a player: W or B ").upper()[0]
    while player not in ('W', 'B'):
        player = input("Invalid input. Choose a player: W or B").upper()[0]

    #Inizializzo
    alpha_player = 'W' if player == 'B' else 'B'
    heuristic = tablut.get_neural_heuristic()
    
    if heuristic is None:
        print("Tflite model not found... Using OldSchoolHeuristic")
        heuristic = OldSchoolHeuristicFunction()

    search = Search()
    current_state = AshtonTablut.get_initial(heuristic)

    #Faccio partire il game loop
    while not current_state.terminal_test():
        current_player = current_state.to_move()

        print("Current Player: {0}".format(current_player))
        current_state.display()

        if current_player == player:
            input_valid = False
            
            while not input_valid:
                actions = [AshtonTablut.num_to_coords(x) for x in current_state.actions()]
                action = input("Choose an action from {0}:".format(actions))
                filtered_action = action
                for x in action:
                    if x not in "1234567890,":
                        filtered_action = filtered_action.replace(x, '')

                try:
                    action = tuple(int(x) for x in filtered_action.split(","))
                except ValueError as a:
                    print(a)
                    continue

                if action in actions:
                    input_valid = True

            print("You have chosen {0} -> {1}".format(action[:2], action[2:4]))

            action = AshtonTablut.coords_to_num(action[0], action[1], action[2], action[3])
            current_state = current_state.result(action)
        else:
            best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
                state=current_state, initial_cutoff_depth=2, cutoff_time=time)

            best_action = AshtonTablut.num_to_coords(best_action)
            print("AlphaTablut has chosen {0} -> {1}".format(best_action[:2], best_action[2:4]))
            current_state = best_next_state

    utility = current_state.utility(player)

    if utility >= 1:
        print("You Won!")
    elif utility <= -1:
        print("You Lost!")

    print("Done.")


def menu_selfplay(tablut):
    print("Not implemented yet.")


def repl(args):
    #Istanzio l'oggetto che gestisce il tutto
    tablut = AlphaTablut()

    print("Checking saving folder...")
    tablut.check_saving_folder()

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model and/or actionbuffer",
            "Save Tflite model",
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
            menu_train(tablut)
        elif choice == 1:
            menu_load(tablut)
        elif choice == 2:
            menu_save_tflite(tablut)
        elif choice == 3:
            menu_play(tablut)
        elif choice == 4:
            menu_selfplay(tablut)
        else:
            break

    print("\nDone")


def main():
    argparser = argparse.ArgumentParser(
        description='AlphaTablut')

    args = argparser.parse_args()

    #log_level = logging.DEBUG if args.debug else logging.INFO
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    repl(args)


if __name__ == '__main__':
    main()
