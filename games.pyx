#cython: language_level=3
"""Games or Adversarial Search (Chapter 5)"""

import time
import random
from collections import namedtuple

import numpy as np

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

# ______________________________________________________________________________
# MinMax Search


def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


# ______________________________________________________________________________

def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

def key_function(x):
    return x[1]

nodes_explored = 0
max_depth = 0

def iterative_deepening_alpha_beta_search(state, game, initial_depth=2, t=55, cutoff_test = None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search using time and uses an evaluation function."""

    player = game.to_move(state)
    current_depth = initial_depth
    
    #Metrics
    global nodes_explored
    global max_depth

    nodes_explored = 0
    max_depth = 0

    # Functions used by alpha_beta
    def max_value(parent_state, state, player, alpha, beta, depth):
        global nodes_explored
        global max_depth
        nodes_explored += 1
        max_depth = max(max_depth, depth)

        if cutoff_test(state, depth):
            return eval_fn(parent_state, state, player)

        v = -np.inf
        
        for a in game.actions(state):  
            next_state = game.result(state, a)            
            v = max(min_value(state, next_state, player, alpha, beta, depth + 1),v)
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(parent_state, state, player, alpha, beta, depth):
        global nodes_explored
        global max_depth
        nodes_explored += 1
        max_depth = max(max_depth, depth)

        if cutoff_test(state, depth):
            return eval_fn(parent_state, state, player)

        v = np.inf

        for a in game.actions(state):  
            next_state = game.result(state, a)          
            v = min(max_value(state, next_state, player, alpha, beta, depth + 1), v)
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    #Start time now
    start_time = time.time()

    # Body of iterative_deepening_alpha_beta_search starts here:
    # The default test cuts off after a time treshold t or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > current_depth or (time.time()-start_time) > t or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda parent_state, state, player: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    best_next_state = None

    children = []

    for a in game.actions(state):
        next_state = game.result(state, a)
        children.append((next_state, eval_fn(state, next_state, player), a))

    sorted(children, key=key_function)

    while (time.time()-start_time) <= t:
        
        for next_state, evaluation, a in children:
            v = min_value(state, next_state, player, best_score, beta, 1)
            if v > best_score:
                best_next_state = next_state
                best_score = v
                best_action = a

        current_depth +=1

    return best_next_state, best_action, best_score, max_depth, nodes_explored, (time.time()-start_time)

# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# ______________________________________________________________________________
# Some Sample Games


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))