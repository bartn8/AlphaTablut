import os
import time

from tablut import AshtonTablut, TablutConfig, convert_board, num_to_coords, coords_to_num
from games import alpha_beta_cutoff_search, iterative_deepening_alpha_beta_search, random_player, GameState
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
        self.game = AshtonTablut()
        self.turn = 0
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
        if utility == 1:
            return (1000 / (self.turn+1)) / 1000
        elif utility == -1:
            return -1

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

    def hardcoded_eval(self, state, next_state, player):
        # Spurio's evaluation function

        #board0 = state.board
        board1 = convert_board(next_state.board)

        #num_p_w_0 = np.sum(board0[0])
        num_p_w_1 = np.sum(board1[0])

        #num_p_b_0 = np.sum(board0[1])
        num_p_b_1 = np.sum(board1[1])

        if player == 'B':
            count = 0

            # Per ogni pedina bianca conto il numero di pedine nere intorno
            # Le citadels possono fare da spalla
            allies = board1[1] | board1[3]
            enemies = board1[0]

            # Seleziono i pedoni del giocatore
            pedoni = np.where(enemies == 1)

            # Ogni pedone vale 1
            for y, x in zip(pedoni[0], pedoni[1]):
                # Su
                for newY in reversed(range(y)):
                    if allies[newY, x] == 0:
                        count += 1
                        break
                # Giu
                for newY in range(y+1, 9):
                    if allies[newY, x] == 0:
                        count += 1
                        break
                # Sinistra
                for newX in reversed(range(x)):
                    if allies[y, newX] == 0:
                        count += 1
                        break
                # Destra
                for newX in range(x+1, 9):
                    if allies[y, newX] == 0:
                        count += 1
                        break

            # Re (Vale 5)
            y, x = np.where(board1[2] == 1)
            y, x = int(y), int(x)

            # Su
            for newY in reversed(range(y)):
                if allies[newY, x] == 0:
                    count += 5
                    break
            # Giu
            for newY in range(y+1, 9):
                if allies[newY, x] == 0:
                    count += 5
                    break
            # Sinistra
            for newX in reversed(range(x)):
                if allies[y, newX] == 0:
                    count += 5
                    break
            # Destra
            for newX in range(x+1, 9):
                if allies[y, newX] == 0:
                    count += 5
                    break

            if self.turn >= 4:
                return (num_p_b_1 * 0.7 - num_p_w_1 - count * 6) / 1000
            else:
                return (num_p_b_1 * 0.7 - num_p_w_1 - count / 176) / 1000

        else:
            # Re
            y, x = np.where(board1[2] == 1)
            y, x = int(y), int(x)
            king_edge_distance = min(x, y, 8-x, 8-y)

            if self.turn >= 4:
                return (num_p_w_1 * 0.7 - king_edge_distance * (17-num_p_b_1) * 0.07 - num_p_b_1) / 1000
            else:
                return (num_p_w_1 * 1.2 - king_edge_distance * (17-num_p_b_1) * 0.07 - num_p_b_1) / 1000

    def have_captured(self, state, next_state):
        a = np.sum(state.board[0]) - np.sum(next_state.board[0])
        b = np.sum(state.board[1]) - np.sum(next_state.board[1])
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
        current_state = self.game.initial
        player = self.game.to_move(current_state)
        max_moves = self.config.max_moves
        self.game_history = [current_state]

        print("Start new game. Player: {0}, Time per move: {1} s, Priority: {2}, Max Moves: {3}".format(
            player, self.time_per_move, self.priority, max_moves))
        start = time.time()

        have_draw = False

        self.turn = 0
        while not self.game.terminal_test(current_state) and not have_draw and self.turn < max_moves:
            best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = iterative_deepening_alpha_beta_search(
                state=current_state, game=self.game, t=self.time_per_move, eval_fn=self.heuristic_eval)

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

            best_action = num_to_coords(best_action)
            print("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}, Captured: {7}".format(self.game.to_move(
                current_state), (best_action[0], best_action[1]), (best_action[2], best_action[3]), search_time, max_depth, nodes_explored, best_score, captured))

            current_state = best_next_state
            self.game_history.append(current_state)

            self.game.display(current_state)

            have_draw = self.have_draw(current_state.board)

            self.turn += 1

        end = time.time()

        result = "DRAW"
        if not have_draw:
            if self.game.utility(current_state, player) == 1:
                result = "WON"
            elif self.game.utility(current_state, player) == -1:
                result = "LOST"

        print("Game ended: Player {0} {1}, Moves: {2}, Time: {3} s".format(
            player, result, self.turn, end-start))

        return self.priority, player, have_draw, self.game.utility(current_state, player), self.game_history


if __name__ == '__main__':
    self_play = SelfPlay(0, 1, 58, None)
    self_play.play()
