import os
import time

from tablut import AshtonTablut, TablutConfig, convert_board
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
        self.interpreter_initialized = False

    def init_tflite(self):
        if not os.path.isfile(self.model_path):
            return False

        self.interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=self.config.threads_per_worker)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors.
        self.tflite_input_details = interpreter.get_input_details()
        self.tflite_output_details = interpreter.get_output_details()

        self.interpreter_initialized = True

        return True

    def heuristic_eval(self, state, next_state, player):
        utility = self.game.utility(next_state, player)
        if utility != 0:
            return utility

        if self.interpreter_initialized:
            return self.tflite_eval(state, next_state, player) * self.heuristic_weight + self.hardcoded_eval(state, next_state, player) * (1-self.heuristic_weight)
        
        return self.hardcoded_eval(state, next_state, player)

    def tflite_eval(self, state, next_state, player):
        board0 = np.reshape(convert_board(state.board), self.tflite_input_details[0]['shape'])
        board1 = np.reshape(convert_board(next_state.board), self.tflite_input_details[1]['shape'])

        self.interpreter.set_tensor(self.tflite_input_details[0]['index'], board0)
        self.interpreter.set_tensor(self.tflite_input_details[1]['index'], board1)

        self.interpreter.invoke()

        v = np.reshape(self.interpreter.get_tensor(self.tflite_output_details[0]['index']), (-1))[0]

        return v if player == 'W' else -v

    def hardcoded_eval(self, state, next_state, player):
        #board0 = state.board
        board1 = convert_board(next_state.board)

        #num_p_w_0 = np.sum(board0[0])
        num_p_w_1 = np.sum(board1[0])

        #num_p_b_0 = np.sum(board0[1])
        num_p_b_1 = np.sum(board1[1])

        y,x = np.where(board1[2] == 1)
        y,x = int(y), int(x)
        
        king_edge_distance = min(x,y,8-x,8-y)
        king_throne_distance = abs(x-4)+abs(y-4)

        king_black_distance = 16
        mask = board1[1] | board1[3]

        #Su
        for newY in reversed(range(y)):
            if mask[newY,x] == 1:
                king_black_distance -= y-newY 
                break

        #Giu
        for newY in range(y+1,9):
            if mask[newY,x] == 1:
                king_black_distance -= newY-y
                break

        #Sinistra
        for newX in reversed(range(x)):
            if mask[y,newX] == 1:
                king_black_distance -= x-newX
                break

        #Destra
        for newX in range(x+1,9):
            if mask[y,newX] == 1:
                king_black_distance -= newX-x
                break

        #Normalizzazione [-1, 1]
        num_p_w_1 = (num_p_w_1 / 4) - 1
        num_p_b_1 = (num_p_b_1 / 8) - 1
        king_edge_distance = (king_edge_distance / 2) - 1
        king_throne_distance = (king_throne_distance / 4) - 1
        king_black_distance = (king_black_distance/8)-1

        if player == 'W':
            #White
            #Max Pedine white (0.1), Distanza Re-Trono (0.2), Distanza Re-Black (0.3)
            #Min Pedine black (0.1), Distanza Re-Bordo (0.3)
            score = num_p_w_1 * 0.1 + -num_p_b_1 * 0.1 + -king_edge_distance * 0.4 + king_throne_distance * 0.1 + king_black_distance * 0.3
        else:
            #Black
            #Min Pedine white (0.1), Distanza Re-Trono (0.2), Distanza Re-Black (0.3)
            #Max Pedine black (0.1), Distanza Re-Bordo (0.3)
            score = -num_p_w_1 * 0.1 + num_p_b_1 * 0.1 + king_edge_distance * 0.3 + -king_throne_distance * 0.1 + -king_black_distance * 0.4

        return score

    def have_captured(self, state, next_state):
        a = np.sum(state.board[0]) -  np.sum(next_state.board[0])
        b = np.sum(state.board[1]) - np.sum(next_state.board[1])
        return a+b

    def have_draw(self, board):
        if self.steps_without_capturing < self.config.moves_for_draw:
            return False
        #Controllo se ho un certo numero di stati ripetuti
        trovati = 0
        for boardCached in self.draw_queue:
            if np.array_equal(board, boardCached):
                trovati +=1
        
        if trovati > 0:
            return True

        return False

    def play(self):
        current_state = self.game.initial
        player = self.game.to_move(current_state)
        max_moves = self.config.max_moves        
        game_history = [current_state]

        print("Start new game. Player: {0}, Time per move: {1} s, Priority: {2}, Max Moves: {3}".format(player, self.time_per_move, self.priority, max_moves))
        start = time.time()

        have_draw = False

        i = 0
        while not self.game.terminal_test(current_state) and not have_draw and i < max_moves:
            #if i % 2 == 0:
            best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = iterative_deepening_alpha_beta_search(state=current_state, game=self.game, t=self.time_per_move, eval_fn=self.heuristic_eval)
            #else:
            #    st = time.time()
            #    best_action, best_score, max_depth, nodes_explored = alpha_beta_cutoff_search(current_state, self.game, d=2), 0, 0, 0
            #    best_next_state, search_time = self.game.result(current_state, best_action), time.time()-st
            #Random
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

            print("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}, Captured: {7}".format(self.game.to_move(current_state), best_action[0], best_action[1], search_time, max_depth, nodes_explored, best_score, captured))

            current_state = best_next_state
            game_history.append(current_state)

            self.game.display(current_state)
            
            have_draw = self.have_draw(current_state.board)

            i +=1

        end = time.time()

        result = "DRAW"
        if not have_draw:
            if self.game.utility(current_state, player) == 1:
                result = "WON"
            elif self.game.utility(current_state, player) == -1:
                result = "LOST"

        print("Game ended: Player {0} {1}, Moves: {2}, Time: {3} s".format(player, result, i, end-start))

        return self.priority, player, have_draw, self.game.utility(current_state, player), game_history

if __name__ == '__main__':
    self_play = SelfPlay(0, 1, 10, None)
    self_play.play()