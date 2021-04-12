#cython: language_level=3

import datetime
import time
import os
import random

cimport cython

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

import array
from cpython cimport array

# tag: numpy
import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays.
DTYPE = np.int8
ctypedef signed char DTYPE_t

#------------------------------ Tablut Config -------------------------------------------------------

class TablutConfig:

    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (4, 9, 9)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(6561))
        # List of players. You should only edit the length
        self.players = list(range(2))
        self.moves_for_draw = 10

        # Network
        self.num_filters = 32

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 8
        self.threads_per_worker = 2
        self.max_moves = 50  # Maximum number of moves if game is not finished before
        self.max_depth = 2

        # Exploration noise
        self.enable_noise_on_training = True
        self.noise_mean = 0.0
        self.noise_deviation = 0.1

        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/results", os.path.basename(__file__)[
                                         :-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 300000
        self.batch_size = 512  # Number of parts of games to train on at each training step
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 100
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.epochs = 10

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate



#------------------------------ Tablut Game -------------------------------------------------------

# ______________________________________________________________________________
## Constants

# Celle non calpestabili: citadels, trono 1 calpestabili 0
# Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels

cdef np.ndarray constraints = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

cdef np.ndarray initialBoard = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1,-1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                            
                            [[0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0]]], dtype=DTYPE)

# ______________________________________________________________________________
## Optimized Cython functions

cdef inline int coords_to_number(int l, int k, int j, int i):
    return l*729 + k*81 + j*9 + i

cdef inline (int, int, int, int) number_to_coords(int number):
    cdef int i, j, k, l

    i = number % 9
    number = number // 9
    j = number % 9
    number = number // 9
    k = number % 9
    number = number // 9
    l = number % 9
    number = number // 9

    return l, k, j, i

cdef class AshtonTablut:
    cdef np.ndarray _board
    cdef unicode _to_move
    cdef int _utility
    cdef array.array _moves
    
    def __init__(self, board, to_move, utility, moves):
        self._board = board
        self._to_move = to_move
        self._utility = utility
        self._moves = moves

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    def get_initial():
        return AshtonTablut(initialBoard.copy(), 'W', 0, AshtonTablut.legal_actions(initialBoard, 'W'))

    @staticmethod
    def coords_to_num(l,k,j,i):
        return coords_to_number(l,k,j,i)

    @staticmethod
    def num_to_coords(number):
        return number_to_coords(number)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef array.array legal_actions(np.ndarray[DTYPE_t, ndim=3] board, unicode to_move):
        if to_move == 'W':
            return AshtonTablut.legal_actions_white(board)
        else:
            return AshtonTablut.legal_actions_black(board)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef array.array legal_actions_black(np.ndarray[DTYPE_t, ndim=3] board):
        cdef array.array legal = array.array('i')

        # Creo una maschera: pedoni, re, cittadelle
        cdef DTYPE_t[:,:] mask = board[0] | board[1] | constraints

        # Seleziono i pedoni del giocatore
        cdef long y, x, newY, newX

        for y in range(9):
            for x in range(9):
                if board[1,y,x] != 1:
                    continue

                # Seleziono le celle adiacenti (no diagonali)
                # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

                # Casi specifici per la maschera delle citadels
                # Sfrutto la simmetria
                if (y == 0 or y == 8) and (x==3 or x==5):
                    legal.append(coords_to_number(y, x, 8-y, x))
                elif (x == 0 or x == 8) and (y==3 or y==5):
                    legal.append(coords_to_number(y, x, y, 8-x))
                elif x==4 and (y == 0 or y == 7):
                    legal.append(coords_to_number(y, x, y+1, x))
                elif x==4 and (y == 1 or y == 8):
                    legal.append(coords_to_number(y, x, y-1, x))
                elif y==4 and (x == 0 or x == 7):
                    legal.append(coords_to_number(y, x, y, x+1))
                elif y==4 and (x == 1 or x == 8):
                    legal.append(coords_to_number(y, x, y, x-1))
                    
                
                # Su
                newY = y-1
                while newY >= 0:
                    if mask[newY, x] == 0:
                        legal.append(coords_to_number(y, x, newY, x))
                    else:
                        break
                    newY -=1

                # Giu
                newY = y+1
                while newY < 9:
                    if mask[newY, x] == 0:
                        legal.append(coords_to_number(y, x, newY, x))
                    else:
                        break
                    newY +=1

                # Sinistra
                newX = x-1
                while newX >= 0:
                    if mask[y, newX] == 0:
                        legal.append(coords_to_number(y, x, y, newX))
                    else:
                        break
                    newX -=1

                # Destra
                newX = x+1
                while newX < 9:
                    if mask[y, newX] == 0:
                        legal.append(coords_to_number(y, x, y, newX))
                    else:
                        break
                    newX +=1

        return legal


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef array.array legal_actions_white(np.ndarray[DTYPE_t, ndim=3] board):
        cdef array.array legal = array.array('i')

        # Creo una maschera: pedoni, re, cittadelle
        cdef DTYPE_t[:,:] mask = board[0] | board[1] | constraints

        cdef long y, x, newY, newX, kingX, kingY
        kingX = 4
        kingY = 4

        for y in range(9):
            for x in range(9):
                if board[0,y,x] == -1:
                    kingY = y
                    kingX = x
                    continue
                elif board[0,y,x] != 1:
                    continue

                # Seleziono le celle adiacenti (no diagonali)
                # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

                # Su
                newY = y-1
                while newY >= 0:
                    if mask[newY, x] == 0:
                        legal.append(coords_to_number(y, x, newY, x))
                    else:
                        break
                    newY -=1

                # Giu
                newY = y+1
                while newY < 9:
                    if mask[newY, x] == 0:
                        legal.append(coords_to_number(y, x, newY, x))
                    else:
                        break
                    newY +=1

                # Sinistra
                newX = x-1
                while newX >= 0:
                    if mask[y, newX] == 0:
                        legal.append(coords_to_number(y, x, y, newX))
                    else:
                        break
                    newX -=1

                # Destra
                newX = x+1
                while newX < 9:
                    if mask[y, newX] == 0:
                        legal.append(coords_to_number(y, x, y, newX))
                    else:
                        break
                    newX +=1

        # Mosse del Re    
    # Mosse del Re    
        # Mosse del Re    
    # Mosse del Re    
        # Mosse del Re    
    # Mosse del Re    
        # Mosse del Re    
        y, x = kingY, kingX
        # Seleziono le celle adiacenti (no diagonali)
        # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

        # Su
        newY = y-1
        while newY >= 0:
            if mask[newY, x] == 0:
                legal.append(coords_to_number(y, x, newY, x))
            else:
                break
            newY -=1

        # Giu
        newY = y+1
        while newY < 9:
            if mask[newY, x] == 0:
                legal.append(coords_to_number(y, x, newY, x))
            else:
                break
            newY +=1

        # Sinistra
        newX = x-1
        while newX >= 0:
            if mask[y, newX] == 0:
                legal.append(coords_to_number(y, x, y, newX))
            else:
                break
            newX -=1

        # Destra
        newX = x+1
        while newX < 9:
            if mask[y, newX] == 0:
                legal.append(coords_to_number(y, x, y, newX))
            else:
                break
            newX +=1

        return legal

    @cython.boundscheck(False) 
    @cython.wraparound(False)
    @staticmethod
    cdef int check_eat(DTYPE_t[:,:,:] board, unicode to_move, int move):
        cdef int eaten = 0
        if to_move == 'W':
            eaten = AshtonTablut.check_white_eat(board, move)
        else:
            eaten = AshtonTablut.check_black_eat(board, move)

        return eaten

    # La board viene modificata!
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    # Controllo se il bianco mangia dei pedoni neri
    cdef int check_white_eat(DTYPE_t[:,:,:] board, int move):
        # Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
        cdef (int,int,int,int) c = number_to_coords(move)
        cdef int y = c[2]
        cdef int x = c[3]
        cdef int captured = 0

        cdef DTYPE_t[:,:] allies = board[0] | constraints
        cdef DTYPE_t[:,:] enemies = board[1]

        # Controlli U,D,L,R
        cdef bint lookUp = allies[y-2, x] != 0 and allies[y-1, x] == 0 and enemies[y-2, x] == 0 and enemies[y-1, x] == 1
        cdef bint lookDown = allies[y+1, x] == 0 and allies[y+2, x] != 0 and enemies[y+1, x] == 1 and enemies[y+2, x] == 0
        cdef bint lookLeft = allies[y, x-2] != 0 and allies[y, x-1] == 0 and enemies[y, x-2] == 0 and enemies[y, x-1] == 1
        cdef bint lookRight = allies[y, x+1] == 0 and allies[y, x+2] != 0 and enemies[y, x+1] == 1 and enemies[y, x+2] == 0

        if lookUp:
            board[1, y-1, x] = 0
            captured += 1
        if lookDown:
            board[1, y+1, x] = 0
            captured += 1
        if lookLeft:
            board[1, y, x-1] = 0
            captured += 1
        if lookRight:
            board[1, y, x+1] = 0
            captured += 1

        return captured

    # La board viene modificata!
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    # Controllo se il nero mangia dei pedoni bianchi
    cdef int check_black_eat(DTYPE_t[:,:,:] board, int move):
        # Dove è finita la pedina nera che dovrà catturare uno o più pedoni bianchi?
        cdef (int,int,int,int) c = number_to_coords(move)
        cdef int y = c[2]
        cdef int x = c[3]
        cdef int captured = 0

        cdef DTYPE_t[:,:] allies = board[1] | constraints
        cdef DTYPE_t[:,:] enemies = board[0]

        # Controlli U,D,L,R
        cdef bint lookUp = allies[y-2, x] != 0 and allies[y-1, x] == 0 and enemies[y-2, x] == 0 and enemies[y-1, x] != 0
        cdef bint lookDown = allies[y+1, x] == 0 and allies[y+2, x] != 0 and enemies[y+1, x] != 0 and enemies[y+2, x] == 0
        cdef bint lookLeft = allies[y, x-2] != 0 and allies[y, x-1] == 0 and enemies[y, x-2] == 0 and enemies[y, x-1] != 0
        cdef bint lookRight = allies[y, x+1] == 0 and allies[y, x+2] != 0 and enemies[y, x+1] != 0 and enemies[y, x+2] == 0

        if lookUp:
            board[0, y-1, x] = 0
            captured += 1
        if lookDown:
            board[0, y+1, x] = 0
            captured += 1
        if lookLeft:
            board[0, y, x-1] = 0
            captured += 1
        if lookRight:
            board[0, y, x+1] = 0
            captured += 1

        return captured

    @cython.boundscheck(False) 
    @cython.wraparound(False)
    @staticmethod
    cdef bint have_winner(DTYPE_t[:,:,:] board, unicode to_move):
        if to_move == 'W':
            return AshtonTablut.white_win_check(board)
        else:
            return AshtonTablut.black_win_check(board)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef bint white_win_check(DTYPE_t[:,:,:] board):
        # Controllo che il Re sia in un bordo della board
        cdef long y,x

        for y in range(9):
            if board[0,y,0] == -1:
                return True
            if board[0,y,8] == -1:
                return True
        
        for x in range(9):
            if board[0,0,x] == -1:
                return True
            if board[0,8,x] == -1:
                return True

        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef bint black_win_check(DTYPE_t[:,:,:] board):
        # Controllo se il nero ha catturato il re

        # Se il re è sul trono allora 4
        # Se il re è adiacente al trono allora 3 pedoni che lo circondano
        # Altrimenti catturo come pedone normale (citadels possono fare da nemico)

        cdef DTYPE_t[:,:] enemies = board[1] | constraints
        cdef long y,x

        for y in range(9):
            for x in range(9):
                # Re sul trono. Controllo i bordi (3,4), (4,3), (4,5), (5,4)
                if y==4 and x==4:
                    return board[1, 3, 4] == 1 and board[1, 4, 3] == 1 and board[1, 4, 5] == 1 and board[1, 5, 4] == 1
                # Re adiacente al trono: controllo se sono presenti nemici intorno
                elif y == 3 and x == 4 or y == 4 and x == 3 or y == 4 and x == 5 or y == 5 and x == 4:
                    return enemies[(y-1),x] == 1 and enemies[y+1, x] == 1 and enemies[y, x-1] == 1 and enemies[y, x+1] == 1
                # Check cattura normale.
                else:  
                    return enemies[y-1, x] == 1 and enemies[y+1, x] == 1 or enemies[y, x-1] == 1 and enemies[y, x+1] == 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def convert_board(self):
        cdef np.ndarray[DTYPE_t, ndim = 3] newBoard = np.zeros((4, 9, 9), dtype=DTYPE)
        newBoard[3] = constraints
        newBoard[1] = self._board[1]

        for y in range(9):
            for x in range(9):
                if self._board[0,y, x] == -1:
                    newBoard[2, y, x] = 1
                if self._board[0, y, x] == 1:
                    newBoard[0, y, x] = 1

        return newBoard

    cpdef actions(self):
        """Legal moves are any square not yet taken."""
        return self._moves

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef result(self, int move):
        """Return the state that results from making a move from a state."""
        cdef unicode next_to_move

        # Copio la board
        cdef np.ndarray[DTYPE_t, ndim = 3] board = self._board.copy()

        # Board da modificare
        cdef DTYPE_t[:,:] move_board = board[0 if self._to_move == 'W' else 1]

        cdef (int,int,int,int) c = number_to_coords(move)
        cdef int y0 = c[0]
        cdef int x0 = c[1]
        cdef int y1 = c[2]
        cdef int x1 = c[3]

        cdef int utility = 0
        cdef int eaten = 0
        cdef array.array moves
        cdef bint winCheck

        tmp = move_board[y0, x0]
        move_board[y0, x0] = 0
        move_board[y1, x1] = tmp

        # Controllo se mangio pedine
        eaten = AshtonTablut.check_eat(board, self._to_move, move)

        next_to_move = 'W' if self._to_move == 'B' else 'B'
        moves = AshtonTablut.legal_actions(board, next_to_move)

        winCheck = AshtonTablut.have_winner(board, self._to_move) or len(moves) == 0

        if winCheck:
            utility = 1 if self._to_move == 'W' else -1

        return AshtonTablut(board, next_to_move, utility, moves)

    cpdef int utility(self, unicode player):
        """A state is terminal if it is won or there are no empty squares."""
        return self._utility if player == 'W' else -self._utility

    cpdef int terminal_test(self):
        """A state is terminal if it is won or there are no empty squares."""
        return self._utility == -1 or self._utility == 1

    cpdef to_move(self):
        """Return the player whose move it is in this state."""
        return self._to_move

    cpdef board(self):
        return self._board

    cdef eval_fn(self, parent_state, player):
        return float(self.utility(player))

    def display(self):
        """Print or otherwise display the state."""
        cdef np.ndarray[DTYPE_t, ndim = 3] board = self.convert_board()
        print(-board[0]+board[1]-20*board[2]+10*board[3])

# ______________________________________________________________________________
### Players for Games

def query_player(AshtonTablut game):
    """Make a move by querying standard input."""
    print("current state:")
    game.display()
    print("available moves: {}".format(game.actions()))
    print("")
    move = None
    if game.actions():
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(AshtonTablut game):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions()) if game.actions() else None

#------------------------------ Search -------------------------------------------------------

cdef long nodes_explored = 0
cdef long max_depth = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cutoff_test(AshtonTablut state, depth, current_cutoff_depth, start_time, cutoff_time):
    return depth > current_cutoff_depth or (get_time()-start_time) > cutoff_time or state.terminal_test()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float max_value(AshtonTablut parent_state, AshtonTablut state, unicode player, float alpha, float beta, long depth, long current_cutoff_depth, double start_time, double cutoff_time):
    cdef float v
    cdef int[:] moves = state.actions()
    cdef int a

    global nodes_explored
    global max_depth

    nodes_explored += 1
    max_depth = max(max_depth, depth)

    if cutoff_test(state, depth, current_cutoff_depth, start_time, cutoff_time):
        return state.eval_fn(parent_state, player)
    
    v = -np.inf
        
    for a in moves:  
        next_state = state.result(a)             
        v = max(min_value(state, next_state, player, alpha, beta, depth + 1, current_cutoff_depth, start_time, cutoff_time),v)
        if v >= beta:
            return v
        alpha = max(alpha, v)

    return v

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float min_value(AshtonTablut parent_state, AshtonTablut state, unicode player, float alpha, float beta, long depth, long current_cutoff_depth, double start_time, double cutoff_time):
    cdef float v
    cdef int[:] moves = state.actions()
    cdef int a

    global nodes_explored
    global max_depth

    nodes_explored += 1
    max_depth = max(max_depth, depth)

    if cutoff_test(state, depth, current_cutoff_depth, start_time, cutoff_time):
        return state.eval_fn(parent_state, player)
    
    v = np.inf
        
    for a in moves:  
        next_state = state.result(a)           
        v = min(max_value(state, next_state, player, alpha, beta, depth + 1, current_cutoff_depth, start_time, cutoff_time),v)
        if v <= alpha:
            return v
        beta = min(beta, v)

    return v 

def iterative_deepening_alpha_beta_search(AshtonTablut state, long initial_cutoff_depth=2, double cutoff_time=55.0):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search using time and uses an evaluation function."""

    cdef unicode player = state.to_move()
    cdef long current_cutoff_depth = initial_cutoff_depth
    cdef double start_time = get_time()
    cdef float best_score = -np.inf
    cdef float beta = np.inf
    cdef int best_action = 0

    cdef float v = 0
    cdef int[:] moves = state.actions()
    cdef int a = 0

    best_next_state = None

    global nodes_explored
    global max_depth

    nodes_explored = 0
    max_depth = 0
    
    while (get_time()-start_time) <= cutoff_time:
        for a in moves:
            next_state = state.result(a) 
            v = min_value(state, next_state, player, best_score, beta, 1, current_cutoff_depth, start_time, cutoff_time)
            if v > best_score:
                best_next_state = next_state
                best_score = v
                best_action = a
        
        current_cutoff_depth += 1
    
    return best_next_state, best_action, best_score, max_depth, nodes_explored, (get_time()-start_time)

#------------------------------ Utils ---------------------------------------------------------
cdef inline double get_time():
    cdef timespec ts
    cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    return current

#------------------------------ Test ---------------------------------------------------------
def test():
    g = AshtonTablut.get_initial()

    st = get_time()
    AshtonTablut.legal_actions_white(g.board())
    print("White legal actions: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    AshtonTablut.legal_actions_black(g.board())
    print("Black legal actions: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    AshtonTablut.check_black_eat(g.board(), 0)
    print("Black eat check: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    AshtonTablut.check_white_eat(g.board(), 0)
    print("White eat check: {0} ms".format(1000*(get_time()-st)))
    fake_board = np.zeros((2, 9, 9), dtype=DTYPE)

    fake_board[0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, -1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

    fake_board[1] = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 1, 0, 0, 0, 0, 0, 1, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=DTYPE)

    fake_state = AshtonTablut(
        to_move='W', utility=0, board=fake_board, moves=AshtonTablut.legal_actions(fake_board, 'W'))

    st = get_time()
    get_time()
    print("Time: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    time.time()
    print("Time: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    fake_state.result(fake_state._moves[0])
    print("Result: {0} ms".format(1000*(get_time()-st)))