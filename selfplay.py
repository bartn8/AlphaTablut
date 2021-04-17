import os
import time

from tablut import AshtonTablut, Search, OldSchoolHeuristicFunction

import numpy as np


class SelfPlay():

    def __init__(self, config, heuristic, priority, time_per_move):
        self.config = config
        self.heuristic = heuristic
        self.priority = priority
        self.time_per_move = time_per_move

        self.steps_without_capturing = 0
        self.draw_queue = []
        self.game_history = []

    def have_captured(self, state, next_state):
        board = state.board()
        next_board = next_state.board()
        a = np.sum(board[0, :, :, 0]) - np.sum(next_board[0, :, :, 0])
        b = np.sum(board[0, :, :, 1]) - np.sum(next_board[0, :, :, 1])
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
            search = Search()

            best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
                state=current_state, initial_cutoff_depth=2, cutoff_time=self.time_per_move)

            captured = self.have_captured(current_state, best_next_state)
            if captured == 0:
                self.steps_without_capturing += 1
                self.draw_queue.append(current_state.board())
            else:
                self.steps_without_capturing = 0
                self.draw_queue = []

            best_action = AshtonTablut.num_to_coords(best_action)
            print("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}, Captured: {7}".format(current_state.to_move(
            ), (best_action[0], best_action[1]), (best_action[2], best_action[3]), search_time, max_depth, nodes_explored, best_score, captured))

            current_state = best_next_state
            self.game_history.append(current_state)

            current_state.display()

            have_draw = self.have_draw(current_state.board())

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
    # test()
    self_play = SelfPlay(TablutConfig(), OldSchoolHeuristicFunction(), 1, 10)
    self_play.play()
