import os
import time

from tablut import AshtonTablut, TablutConfig, IterativeDeepeningSearch, test

import tflite_runtime.interpreter as tflite

import numpy as np


class SelfPlay():

    def __init__(self, heuristic_weight, priority, time_per_move, model_path="tablut.tflite"):
        self.heuristic_weight = heuristic_weight
        self.priority = priority
        self.time_per_move = time_per_move
        self.model_path = model_path
        self.steps_without_capturing = 0
        self.draw_queue = []
        self.config = TablutConfig()
        self.game_history = []
        self.interpreter_initialized = False

    def init_tflite(self):
        if not os.path.isfile(self.model_path):
            return False

        self.interpreter = tflite.Interpreter(
            model_path=self.model_path, num_threads=self.config.threads_per_worker)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.tflite_input_details = interpreter.get_input_details()
        self.tflite_output_details = interpreter.get_output_details()

        self.interpreter_initialized = True

        return True

    def heuristic_eval(self, state, next_state, player):
        utility = self.game.utility(state, player)
        if utility != 0:
            return utility

        if self.interpreter_initialized:
            return self.tflite_eval(state, next_state, player) * self.heuristic_weight + self.hardcoded_eval(state, next_state, player) * (1-self.heuristic_weight)

        return self.hardcoded_eval(state, next_state, player)

    def tflite_eval(self, state, next_state, player):
        board0 = np.reshape(convert_board(state.board),
                            self.tflite_input_details[0]['shape'])
        board1 = np.reshape(convert_board(next_state.board),
                            self.tflite_input_details[1]['shape'])

        self.interpreter.set_tensor(
            self.tflite_input_details[0]['index'], board0)
        self.interpreter.set_tensor(
            self.tflite_input_details[1]['index'], board1)

        self.interpreter.invoke()

        v = np.reshape(self.interpreter.get_tensor(
            self.tflite_output_details[0]['index']), (-1))[0]

        return v if player == 'W' else -v

    def have_captured(self, state, next_state):
        board = state.board()
        next_board = next_state.board()
        a = np.sum(board[0]) - np.sum(next_board[0])
        b = np.sum(board[1]) - np.sum(next_board[1])
        return a+b

    def have_draw(self, board):
        if self.steps_without_capturing < self.config.moves_for_draw:
            return False
        # Controllo se ho un certo numero di stati ripetuti
        trovati = 0
        for boardCached in self.draw_queue:
            if np.array_equal(board, boardCached):
                trovati += 1

        if trovati > 0:
            return True

        return False

    def play(self):
        current_state = AshtonTablut.get_initial()
        player = current_state.to_move()
        max_moves = self.config.max_moves
        self.game_history = [current_state]

        print("Start new game. Player: {0}, Time per move: {1} s, Priority: {2}, Max Moves: {3}".format(
            player, self.time_per_move, self.priority, max_moves))
        start = time.time()

        have_draw = False

        while not current_state.terminal_test() and not have_draw and current_state.turn() < max_moves:
            search = IterativeDeepeningSearch()

            best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.search(
                state=current_state, cutoff_time=self.time_per_move)
            
            #best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = iterative_deepening_alpha_beta_search(
            #    state=current_state, game=self.game, t=self.time_per_move, eval_fn=self.heuristic_eval)

            #    st = time.time()
            #    best_action, best_score, max_depth, nodes_explored = alpha_beta_cutoff_search(current_state, self.game, d=2), 0, 0, 0
            #    best_next_state, search_time = self.game.result(current_state, best_action), time.time()-st

            # Random
            #st = time.time()
            #best_action, best_score, max_depth, nodes_explored = random_player(self.game, current_state), 0, 0, 0
            #best_next_state, search_time = self.game.result(current_state, best_action), time.time()-st

            captured = self.have_captured(current_state, best_next_state)
            if captured == 0:
                self.steps_without_capturing += 1
                self.draw_queue.append(current_state.board)
            else:
                self.steps_without_capturing = 0
                self.draw_queue = []

            best_action = AshtonTablut.num_to_coords(best_action)
            print("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}, Captured: {7}".format(current_state.to_move(), (best_action[0], best_action[1]), (best_action[2], best_action[3]), search_time, max_depth, nodes_explored, best_score, captured))

            current_state = best_next_state
            self.game_history.append(current_state)

            current_state.display()

            have_draw = self.have_draw(current_state.board)

        end = time.time()

        result = "DRAW"
        if not have_draw:
            if current_state.utility(player) == 1:
                result = "WON"
            elif current_state.utility(player) == -1:
                result = "LOST"

        print("Game ended: Player {0} {1}, Moves: {2}, Time: {3} s".format(
            player, result, current_state.turn(), end-start))

        return self.priority, player, have_draw, current_state.utility(player), self.game_history


if __name__ == '__main__':
    test()
    self_play = SelfPlay(0, 1, 5, None)
    self_play.play()
