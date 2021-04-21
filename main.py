from network import ResNNet
from actionbuffer import ActionBuffer
from selfplay import SelfPlay
from tablut import AshtonTablut, Search, OldSchoolHeuristicFunction, NeuralHeuristicFunction, MixedHeuristicFunction
from tablutconfig import TablutConfig

import ray
import time
import os
import argparse
import logging
from tqdm import tqdm
import queue

ray.init(log_to_driver=False)


class AlphaTablut:

    def __init__(self):
        self.config = TablutConfig()
        self.action_buffer = ActionBuffer(self.config)
        self.nnet = ResNNet(self.config)

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
        folder = self.config.folder
        filename = self.config.checkpoint_name
        filename_meta = self.config.checkpoint_metadata
        filepath = os.path.join(folder, filename)
        filepath_meta = os.path.join(folder, filename_meta)

        return os.path.exists(folder) and os.path.isdir(folder) and os.path.exists(filepath) and os.path.isfile(filepath_meta)

    def check_saved_tflite_model(self):
        folder = self.config.folder
        filename = self.config.tflite_model
        filepath = os.path.join(folder, filename)

        return os.path.exists(filepath) and os.path.isfile(filepath)

    def get_neural_heuristic(self):
        if self.check_saved_tflite_model():
            return NeuralHeuristicFunction(self.config)

        return None

    def get_mixed_heuristic(self, alpha):
        if self.check_saved_tflite_model():
            return MixedHeuristicFunction(self.config, alpha)

        return None


class SelfPlayResult:
    def __init__(self, priority, winner, utility, history):
        self.priority = priority
        self.winner = winner
        self.utility = utility
        self.history = history

# Menu: Train, load pre-trained, play, self-play


# Ray Workers:
#   1) Selplayer
#   2) Trainer
#   3) ActionBuffer Explorer
@ray.remote
def self_play_worker(priority, heuristic_alpha, random=False):
    logging.basicConfig(
        format='%(levelname)s:%(asctime)s: %(message)s', level=logging.INFO)
    config = TablutConfig()

    folder = config.folder
    filename = config.tflite_model
    filepath = os.path.join(folder, filename)

    heuristic = OldSchoolHeuristicFunction()

    #if not random:
    #    if os.path.exists(filepath) and os.path.isfile(filepath):
    #        heuristic = MixedHeuristicFunction(config, heuristic_alpha)
    #        heuristic.init_tflite()
    #        if heuristic.initialized():
    #            logging.info("Tflite model loaded")
    #    else:
    #        logging.info("Tflite model not found... Using OldSchoolHeuristic")
    #        heuristic = OldSchoolHeuristicFunction()

    time_per_move = config.max_time / (2*priority+1)

    logging.info("Starting SelfPlay. Priority: {0}".format(priority))

    winner = 'D'
    selfplay = SelfPlay(config, heuristic, priority, time_per_move)

    while winner == 'D':
        winner, utility, history = selfplay.play(random)

    logging.info("Done. Result: {0}".format(winner))

    return SelfPlayResult(priority, winner, utility, history)


def menu_train(tablut):
    batch_size = tablut.config.batch_size
    min_batch_size = tablut.config.min_batch_size
    num_workers = tablut.config.num_workers
    steps_counter = tablut.nnet.training_steps
    idtable = {}
    next_idtable = {}
    played_games = 0

    # training loop
    pbar = tqdm(initial=steps_counter, desc='Game played {0}, Training steps'.format(
        tablut.action_buffer.game_counter), total=tablut.config.training_steps)

    while tablut.nnet.training_steps < tablut.config.training_steps:
        heuristic_alpha = min(
            1.0, (2*tablut.nnet.training_steps/tablut.config.training_steps))
        #workers_alpha = 0.7 + \
        #    min(0.2, tablut.nnet.training_steps/tablut.config.training_steps)

        #num_search_workers = int(num_workers * workers_alpha)
        #num_random_workers = num_workers - num_search_workers

        logging.info("Starting workers...")

        for priority in range(num_workers):
            if priority not in idtable:
                id = self_play_worker.remote(
                    priority, heuristic_alpha, priority == num_workers-1)
                idtable[priority] = id

        ready_ids, remaining_ids = ray.wait(
            list(idtable.values()), num_returns=1, timeout=1)

        for id in remaining_ids:
            for priority in idtable:
                if idtable[priority] == id:
                    next_idtable[priority] = id

        for id in ready_ids:
            result = ray.get(id)
            priority, winner, utility, history = result.priority, result.winner, result.utility, result.history
            played_games += 1

            logging.info("Worker done.".format(priority))

            # Aggiornamento dell'action buffer e riavvio
            logging.info("ActionBuffer updating...")

            k = 1
            i = len(history)-1
            while i >= 0:
                board1 = history[i]
                tablut.action_buffer.store_action(
                    board1, utility/k, 1/(priority+1))
                i -= 1
                k += 1

            tablut.action_buffer.increment_game_counter()

            logging.info("ActionBuffer updating done")

        if tablut.action_buffer.size() >= min_batch_size and played_games >= tablut.config.new_games_per_epoch:
            played_games = 0
            logging.debug("Dataset generating...")
            batch_size = min(tablut.config.batch_size, tablut.action_buffer.size())
            dataset = tablut.action_buffer.generate_dataset(batch_size)
            logging.debug("Done.")
            logging.debug("Network training...")
            loss_history = tablut.nnet.train(dataset)
            logging.info("Training step {0}, Loss: {1}".format(tablut.nnet.training_steps, loss_history[-1]))
            logging.debug("Done.")

            if tablut.nnet.training_steps % tablut.config.checkpoint_interval == 0:
                tablut.nnet.save_checkpoint()
                tablut.nnet.tflite_optimization()
                tablut.action_buffer.save_buffer()

        idtable = next_idtable
        next_idtable = {}

        pbar.set_description('Game played {0}, Training steps'.format(
            tablut.action_buffer.game_counter))
        pbar.update(tablut.nnet.training_steps-steps_counter)
        steps_counter = tablut.nnet.training_steps

    pbar.close()

    print("Done.")


def menu_load(tablut):
    if tablut.check_saved_checkpoint():
        print("Saved checkpoint found.")
        print("Loading checkpoint...")
        tablut.nnet.load_checkpoint()
    else:
        print("No checkpoint found")

    if tablut.check_saved_actionbuffer():
        print("Saved actionbuffer found.")
        print("Loading action buffer...")
        tablut.action_buffer.load_buffer()
    else:
        print("No actionbuffer found")

    print("Done.")


def menu_play(tablut):
    time = input("Insert AlphaTablut Search time in seconds: ")
    time = int(time)

    player = input("Choose a player: W or B ").upper()[0]
    while player not in ('W', 'B'):
        player = input("Invalid input. Choose a player: W or B").upper()[0]

    # Inizializzo
    alpha_player = 'W' if player == 'B' else 'B'
    heuristic = tablut.get_neural_heuristic()

    if heuristic is None:
        print("Tflite model not found... Using OldSchoolHeuristic")
        heuristic = OldSchoolHeuristicFunction()
    else:
        heuristic.init_tflite()
        if heuristic.initialized():
            print("Tflite model loaded")
        else:
            print("Model loading failed... Using OldSchoolHeuristic")
            heuristic = OldSchoolHeuristicFunction()

    search = Search()
    current_state = AshtonTablut.get_initial(heuristic)

    # Faccio partire il game loop
    i = 0
    while not current_state.terminal_test():
        current_player = current_state.to_move()

        print("Turn {0}".format(i+1))
        print("Current Player: {0}".format(current_player))
        print(current_state.display())

        if current_player == player:
            input_valid = False

            while not input_valid:
                actions = [AshtonTablut.num_to_coords(
                    x) for x in current_state.actions()]
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

            action = AshtonTablut.coords_to_num(
                action[0], action[1], action[2], action[3])
            current_state = current_state.result(action)
        else:
            best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
                state=current_state, initial_cutoff_depth=2, cutoff_time=time)

            best_action = AshtonTablut.num_to_coords(best_action)
            print(
                "AlphaTablut has chosen {0} -> {1}, Max depth: {2}, Nodes Explored: {3}, Best Score: {4}, Search Time: {5}".format(best_action[:2], best_action[2:4], max_depth, nodes_explored, best_score, search_time))
            current_state = best_next_state

        i += 1

    utility = current_state.utility(player)

    if utility >= 1:
        print("You Won!")
    elif utility <= -1:
        print("You Lost!")

    print("Done.")


def menu_selfplay(tablut):
    time = input("Insert AlphaTablut Search time in seconds: ")
    time = int(time)

    max_moves = input("Insert max moves: ")
    max_moves = int(max_moves)

    heuristic = tablut.get_neural_heuristic()

    if heuristic is None:
        print("Tflite model not found... Using OldSchoolHeuristic")
        heuristic = OldSchoolHeuristicFunction()
    else:
        heuristic.init_tflite()
        if heuristic.initialized():
            print("Tflite model loaded")
        else:
            print("Model loading failed... Using OldSchoolHeuristic")
            heuristic = OldSchoolHeuristicFunction()
        

    search = Search()
    current_state = AshtonTablut.get_initial(heuristic)

    # Faccio partire il game loop
    i = 0
    while not current_state.terminal_test() and i < max_moves:

        current_player = current_state.to_move()

        print("Turn {0}".format(i+1))
        print("Current Player: {0}".format(current_player))
        print(current_state.display())

        best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
            state=current_state, initial_cutoff_depth=2, cutoff_time=time)

        best_action = AshtonTablut.num_to_coords(best_action)
        print(
            "AlphaTablut has chosen {0} -> {1}, Max depth: {2}, Nodes Explored: {3}, Best Score: {4}, Search Time: {5}".format(best_action[:2], best_action[2:4], max_depth, nodes_explored, best_score, search_time))
        current_state = best_next_state
        i += 1

    print("Done.")


def repl(args):
    # Istanzio l'oggetto che gestisce il tutto
    tablut = AlphaTablut()

    print("Checking saving folder...")
    tablut.check_saving_folder()

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model and/or actionbuffer",
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
            try:
                menu_train(tablut)
            except KeyboardInterrupt:
                tablut.nnet.save_checkpoint()
                tablut.nnet.tflite_optimization()
                tablut.action_buffer.save_buffer()
        elif choice == 1:
            menu_load(tablut)
        elif choice == 2:
            menu_play(tablut)
        elif choice == 3:
            menu_selfplay(tablut)
        else:
            break

    print("\nDone")


def main():
    argparser = argparse.ArgumentParser(
        description='AlphaTablut')

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s:%(asctime)s: %(message)s',
                        filename='train.log', level=log_level)

    repl(args)


if __name__ == '__main__':
    main()
