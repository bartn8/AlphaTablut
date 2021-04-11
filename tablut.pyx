#cython: language_level=3

import datetime
import time
import os
from games import Game, GameState

cimport cython

import array
from cpython cimport array


# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.


# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int8

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef signed char DTYPE_t


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


# Celle non calpestabili: citadels, trono 1 calpestabili 0
# Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels

whiteContraints = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints = np.zeros((9, 9, 9, 9), dtype=DTYPE)

# Celle non calpestabili: citadels, trono 1 calpestabili 0
# Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels
# Maschere speciali per la verifica delle mosse attuabili dal nero
blackContraints[:, :] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints[0, 3:6] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints[1,  4] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints[8, 3:6] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

blackContraints[7,  4] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

blackContraints[3:6, 0] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints[4,  1] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints[3:6, 8] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackContraints[4,  7] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)


initialBoard = np.zeros((2, 9, 9), dtype=DTYPE)

# Board[0]: Bianco altro 0
initialBoard[0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, -1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

# Board[1]: Nero altro 0
initialBoard[1] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=DTYPE)

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

def coords_to_num(l,k,j,i):
    return l*729 + k*81 + j*9 + i

def num_to_coords(number):
    i = number % 9
    number = number // 9
    j = number % 9
    number = number // 9
    k = number % 9
    number = number // 9
    l = number % 9
    number = number // 9

    return l, k, j, i

@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
cdef array.array legal_actions(np.ndarray[DTYPE_t, ndim=3] board, to_move):
    if to_move == 'W':
        return legal_actions_white(board)
    else:
        return legal_actions_black(board)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
cdef array.array legal_actions_black(np.ndarray[DTYPE_t, ndim=3] board):
    cdef array.array legal = array.array('i')

    # Creo una maschera: pedoni, re, cittadelle
    cdef DTYPE_t[:,:] preMask = board[0] | board[1]
    cdef DTYPE_t[:,:] mask

    # Seleziono i pedoni del giocatore
    cdef long[:,:] pedoni = np.argwhere(board[1] == 1)
    cdef long y, x, newY, newX

    for y, x in pedoni:
        # Seleziono le celle adiacenti (no diagonali)
        # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

        # Casi specifici per la maschera delle citadels
        mask = preMask | blackContraints[y, x]

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


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
cdef array.array legal_actions_white(np.ndarray[DTYPE_t, ndim=3] board):
    cdef array.array legal = array.array('i')

    # Creo una maschera: pedoni, re, cittadelle
    cdef DTYPE_t[:,:] mask = board[0] | board[1] | whiteContraints

    # Seleziono i pedoni del giocatore
    cdef long[:,:] pedoni = np.argwhere(board[0] == 1)

    # Seleziono il Re
    cdef long[:,:] king = np.argwhere(board[0] == -1)

    cdef long y, x, newY, newX

    for y, x in pedoni:
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
    y, x = king[0]
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

# La board viene modificata!
@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
# Controllo se il bianco mangia dei pedoni neri
cdef int check_white_eat(np.ndarray[DTYPE_t, ndim=3] board, int move):
    # Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
    cdef (int,int,int,int) c = number_to_coords(move)
    cdef int y = c[2]
    cdef int x = c[3]
    cdef int captured = 0

    cdef DTYPE_t[:,:] allies = np.where(board[0] != 0, 1, 0).astype(DTYPE) | whiteContraints
    cdef DTYPE_t[:,:] enemies = board[1]

    # Controlli U,D,L,R
    cdef bint lookUp = allies[y-2, x] == 1 and allies[y-1, x] == 0 and enemies[y-2, x] == 0 and enemies[y-1, x] == 1
    cdef bint lookDown = allies[y+1, x] == 0 and allies[y+2, x] == 1 and enemies[y+1, x] == 1 and enemies[y+2, x] == 0
    cdef bint lookLeft = allies[y, x-2] == 1 and allies[y, x-1] == 0 and enemies[y, x-2] == 0 and enemies[y, x-1] == 1
    cdef bint lookRight = allies[y, x+1] == 0 and allies[y, x+2] == 1 and enemies[y, x+1] == 1 and enemies[y, x+2] == 0

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
@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
# Controllo se il bianco mangia dei pedoni neri
cdef int check_black_eat(np.ndarray[DTYPE_t, ndim=3] board, int move):
    # Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
    cdef (int,int,int,int) c = number_to_coords(move)
    cdef int y = c[2]
    cdef int x = c[3]
    cdef int captured = 0

    cdef DTYPE_t[:,:] allies = board[1] | whiteContraints
    cdef DTYPE_t[:,:] enemies = board[0]

    # Controlli U,D,L,R
    cdef bint lookUp = allies[y-2, x] == 1 and allies[y-1, x] == 0 and enemies[y-2, x] == 0 and enemies[y-1, x] == 1
    cdef bint lookDown = allies[y+1, x] == 0 and allies[y+2, x] == 1 and enemies[y+1, x] == 1 and enemies[y+2, x] == 0
    cdef bint lookLeft = allies[y, x-2] == 1 and allies[y, x-1] == 0 and enemies[y, x-2] == 0 and enemies[y, x-1] == 1
    cdef bint lookRight = allies[y, x+1] == 0 and allies[y, x+2] == 1 and enemies[y, x+1] == 1 and enemies[y, x+2] == 0

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

@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
cdef bint have_winner(np.ndarray[DTYPE_t, ndim=3] board, to_move):
    if to_move == 'W':
        return white_win_check(board)
    else:
        return black_win_check(board)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
cdef bint white_win_check(np.ndarray[DTYPE_t, ndim=3] board):
    # Controllo che il Re sia in un bordo della board
    cdef long[:,:] king = np.argwhere(board[0] == -1)
    cdef long y = king[0][0]
    cdef long x = king[0][1]
    return x == 0 or x == 8 or y == 0 or y == 8


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
cdef bint black_win_check(np.ndarray[DTYPE_t, ndim=3] board):
    # Controllo se il nero ha catturato il re

    # Se il re è sul trono allora 4
    # Se il re è adiacente al trono allora 3 pedoni che lo circondano
    # Altrimenti catturo come pedone normale (citadels possono fare da nemico)

    cdef DTYPE_t[:,:] enemies = board[1] | whiteContraints
    cdef long[:,:] king = np.argwhere(board[0] == -1)
    cdef long y = king[0][0]
    cdef long x = king[0][1]

    # Re sul trono. Controllo i bordi (3,4), (4,3), (4,5), (5,4)
    if y == 4 and x == 4:
        return board[1, 3, 4] == 1 and board[1, 4, 3] == 1 and board[1, 4, 5] == 1 and board[1, 5, 4] == 1
    # Re adiacente al trono: controllo se sono presenti nemici intorno
    elif y == 3 and x == 4 or y == 4 and x == 3 or y == 4 and x == 5 or y == 5 and x == 4:
        return enemies[(y-1),x] == 1 and enemies[y+1, x] == 1 and enemies[y, x-1] == 1 and enemies[y, x+1] == 1
    # Check cattura normale.
    else:  
        return enemies[y-1, x] == 1 and enemies[y+1, x] == 1 or enemies[y, x-1] == 1 and enemies[y, x+1] == 1


def convert_board(board):
    cdef np.ndarray[DTYPE_t, ndim = 3] newBoard = np.zeros((4, 9, 9), dtype=DTYPE)
    newBoard[3] = whiteContraints
    newBoard[2] = np.where(board[0] == -1, 1, 0)
    newBoard[1] = board[1]
    newBoard[0] = np.where(board[0] == 1, 1, 0)
    return newBoard


class AshtonTablut(Game):
    def __init__(self):
        self.initial = GameState(to_move='W', utility=0, board=initialBoard.copy(), moves=legal_actions(initialBoard, 'W'))

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    # turn off negative index wrapping for entire function
    @cython.wraparound(False)
    def result(self, state, int move):
        """Return the state that results from making a move from a state."""

        # Copio la board
        cdef np.ndarray[DTYPE_t, ndim = 3] board = state.board.copy()

        # Controllo se ho mosso il re
        cdef np.ndarray[DTYPE_t, ndim = 2] move_board = board[0 if state.to_move == 'W' else 1]

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
        if state.to_move == 'W':
            eaten = check_white_eat(board, move)
        else:
            eaten = check_black_eat(board, move)

        to_move = 'W' if state.to_move == 'B' else 'B'
        moves = legal_actions(board, to_move)

        winCheck = have_winner(board, state.to_move) or len(moves) == 0

        if winCheck:
            utility = 1 if state.to_move == 'W' else -1

        return GameState(to_move=to_move, utility=utility, board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'W' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility == -1 or state.utility == 1 or len(state.moves) == 0

    def display(self, state):
        """Print or otherwise display the state."""
        board = state.board
        white = np.where(board[0] == 1, 1, 0)
        king = np.where(board[0] == -1, 1, 0)
        print(-white+board[1]-20*king+10*whiteContraints)

def test():
    g = AshtonTablut()

    st = time.time()
    legal_actions_white(g.initial.board)
    print("White legal actions: {0} ms".format(1000*(time.time()-st)))

    st = time.time()
    legal_actions_black(g.initial.board)
    print("Black legal actions: {0} ms".format(1000*(time.time()-st)))

    st = time.time()
    g.result(g.initial, g.initial.moves[0])
    print("Result: {0} ms".format(1000*(time.time()-st)))

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

    fake_state = GameState(
        to_move='W', utility=0, board=fake_board, moves=legal_actions(fake_board, 'W'))

    print(fake_state.moves)
    #l*729 + k*81 + j*9 + i
