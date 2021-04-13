#cython: language_level=3

import datetime
import time as ptime
import os
import random

cimport cython

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdlib cimport srand, rand, RAND_MAX
from libc.time cimport time

import array
from cpython cimport array

# tag: numpy
import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays.
DTYPE = np.int8
ctypedef signed char DTYPE_t

srand(time(NULL))

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

cdef np.ndarray whiteConstraints = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

# Celle non calpestabili: citadels, trono 1 calpestabili 0
# Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels
# Maschere speciali per la verifica delle mosse attuabili dal nero
cdef np.ndarray blackConstraints = np.zeros((9, 9, 9, 9), dtype=DTYPE)

blackConstraints[:, :] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackConstraints[0, 3:6] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackConstraints[1,  4] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackConstraints[8, 3:6] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

blackConstraints[7,  4] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

blackConstraints[3:6, 0] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackConstraints[4,  1] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackConstraints[3:6, 8] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

blackConstraints[4,  7] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
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
    cdef np.ndarray _moves
    cdef long _turn
    
    def __init__(self, board, to_move, utility, moves, turn = 0):
        self._board = board
        self._to_move = to_move
        self._utility = utility
        self._moves = moves
        self._turn = turn

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
    cdef np.ndarray legal_actions(DTYPE_t[:,:,:] board, unicode to_move):
        if to_move == 'W':
            return AshtonTablut.legal_actions_white(board)
        else:
            return AshtonTablut.legal_actions_black(board)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef np.ndarray legal_actions_black(DTYPE_t[:,:,:] board):
        cdef np.ndarray actions = np.zeros((256, ), dtype=np.int32)
        cdef int[:] legal = actions
        cdef int i = 0
        cdef int tmp, r = 0

        # Creo una maschera: pedoni, re, cittadelle
        cdef DTYPE_t[:,:] board0 = board[0]
        cdef DTYPE_t[:,:] board1 = board[1]
        cdef DTYPE_t[:,:,:,:] constraints = blackConstraints

        # Seleziono i pedoni del giocatore
        cdef int y, x, newY, newX

        for y in range(9):
            for x in range(9):
                if board1[y,x] != 1:
                    continue

                # Seleziono le celle adiacenti (no diagonali)
                # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)
                
                # Su
                newY = y-1
                while newY >= 0:
                    if board0[newY, x] == 0 and board1[newY, x] == 0 and constraints[y, x, newY, x] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY -=1

                # Giu
                newY = y+1
                while newY < 9:
                    if board0[newY, x] == 0 and board1[newY, x] == 0 and constraints[y, x, newY, x] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY +=1

                # Sinistra
                newX = x-1
                while newX >= 0:
                    if board0[y, newX] == 0 and board1[y, newX] == 0 and constraints[y, x, y, newX] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX -=1

                # Destra
                newX = x+1
                while newX < 9:
                    if board0[y, newX] == 0 and board1[y, newX] == 0 and constraints[y, x, y, newX] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX +=1

        actions = actions[:i]

        #Shuffle
        for k in range(i//2):
            r = int((rand()/RAND_MAX))*(i-1)
            tmp = legal[i]
            legal[i] = legal[r]
            legal[r] = tmp

        return actions

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef np.ndarray legal_actions_white(DTYPE_t[:,:,:] board):
        cdef np.ndarray actions = np.zeros((256, ), dtype=np.int32)
        cdef int[:] legal = actions
        cdef int i = 0
        cdef int tmp, r = 0

        # Creo una maschera: pedoni, re, cittadelle
        cdef DTYPE_t[:,:] board0 = board[0]
        cdef DTYPE_t[:,:] board1 = board[1]
        cdef DTYPE_t[:,:] constraints = whiteConstraints

        cdef int y, x, newY, newX, kingX, kingY
        kingX = 4
        kingY = 4

        for y in range(9):
            for x in range(9):
                if board0[y,x] == -1:
                    kingY = y
                    kingX = x
                    continue
                elif board0[y,x] != 1:
                    continue

                # Seleziono le celle adiacenti (no diagonali)
                # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

                # Su
                newY = y-1
                while newY >= 0:
                    if board0[newY, x] == 0 and board1[newY, x] == 0 and constraints[newY, x] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY -=1

                # Giu
                newY = y+1
                while newY < 9:
                    if board0[newY, x] == 0 and board1[newY, x] == 0 and constraints[newY, x] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY +=1

                # Sinistra
                newX = x-1
                while newX >= 0:
                    if board0[y, newX] == 0 and board1[y, newX] == 0 and constraints[y, newX] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX -=1

                # Destra
                newX = x+1
                while newX < 9:
                    if board0[y, newX] == 0 and board1[y, newX] == 0 and constraints[y, newX] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX +=1

        # Mosse del Re      
        y, x = kingY, kingX
        # Seleziono le celle adiacenti (no diagonali)
        # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

        # Su
        newY = y-1
        while newY >= 0:
            if board0[newY, x] == 0 and board1[newY, x] == 0 and constraints[newY, x] == 0:
                legal[i] = coords_to_number(y, x, newY, x)
                i+=1
            else:
                break
            newY -=1

        # Giu
        newY = y+1
        while newY < 9:
            if board0[newY, x] == 0 and board1[newY, x] == 0 and constraints[newY, x] == 0:
                legal[i] = coords_to_number(y, x, newY, x)
                i+=1
            else:
                break
            newY +=1

        # Sinistra
        newX = x-1
        while newX >= 0:
            if board0[y, newX] == 0 and board1[y, newX] == 0 and constraints[y, newX] == 0:
                legal[i] = coords_to_number(y, x, y, newX)
                i+=1
            else:
                break
            newX -=1

        # Destra
        newX = x+1
        while newX < 9:
            if board0[y, newX] == 0 and board1[y, newX] == 0 and constraints[y, newX] == 0:
                legal[i] = coords_to_number(y, x, y, newX)
                i+=1
            else:
                break
            newX +=1

        actions = actions[:i]

        #Shuffle
        for k in range(i//2):
            r = int((rand()/RAND_MAX))*(i-1)
            tmp = legal[i]
            legal[i] = legal[r]
            legal[r] = tmp

        return actions

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
        cdef DTYPE_t[:,:] allies = board[0]
        cdef DTYPE_t[:,:] constraints = whiteConstraints
        cdef DTYPE_t[:,:] enemies = board[1]

        # Controlli U,D,L,R
        cdef bint lookUp = y-2 >= 0 and allies[y-2, x] + constraints[y-2, x] > 0 and allies[y-1, x] + constraints[y-1, x] == 0 and enemies[y-2, x] == 0 and enemies[y-1, x] == 1
        cdef bint lookDown = y+2 < 9 and allies[y+1, x] + constraints[y+1, x] == 0 and allies[y+2, x] + constraints[y+2, x] > 0 and enemies[y+1, x] == 1 and enemies[y+2, x] == 0
        cdef bint lookLeft = x-2 >= 0 and allies[y, x-2] + constraints[y, x-2] > 0 and allies[y, x-1] + constraints[y, x-1] == 0 and enemies[y, x-2] == 0 and enemies[y, x-1] == 1
        cdef bint lookRight = x+2 < 9 and allies[y, x+1] + constraints[y, x+1] == 0 and allies[y, x+2] + constraints[y, x+2] > 0 and enemies[y, x+1] == 1 and enemies[y, x+2] == 0

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
        cdef DTYPE_t[:,:] allies = board[1]
        cdef DTYPE_t[:,:] constraints = whiteConstraints
        cdef DTYPE_t[:,:] enemies = board[0]

        # Controlli U,D,L,R
        cdef bint lookUp = y-2 >= 0 and allies[y-2, x] + constraints[y-2, x] > 0 and allies[y-1, x] + constraints[y-1, x] == 0 and enemies[y-2, x] == 0 and enemies[y-1, x] == 1
        cdef bint lookDown = y+2 < 9 and allies[y+1, x] + constraints[y+1, x] == 0 and allies[y+2, x] + constraints[y+2, x] > 0 and enemies[y+1, x] == 1 and enemies[y+2, x] == 0
        cdef bint lookLeft = x-2 >= 0 and allies[y, x-2] + constraints[y, x-2] > 0 and allies[y, x-1] + constraints[y, x-1] == 0 and enemies[y, x-2] == 0 and enemies[y, x-1] == 1
        cdef bint lookRight = x+2 < 9 and allies[y, x+1] + constraints[y, x+1] == 0 and allies[y, x+2] + constraints[y, x+2] > 0 and enemies[y, x+1] == 1 and enemies[y, x+2] == 0

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
        cdef DTYPE_t[:,:] board0 = board[0]
        cdef DTYPE_t[:,:] board1 = board[1]
        cdef DTYPE_t[:,:] constraints = whiteConstraints
        cdef long y,x

        for y in range(9):
            for x in range(9):
                if board0[y, x] == -1:
                    # Re sul trono. Controllo i bordi (3,4), (4,3), (4,5), (5,4)
                    if y==4 and x==4:
                        return board1[3, 4] == 1 and board1[4, 3] == 1 and board1[4, 5] == 1 and board1[5, 4] == 1
                    # Re adiacente al trono: controllo se sono presenti nemici intorno
                    elif y == 3 and x == 4 or y == 4 and x == 3 or y == 4 and x == 5 or y == 5 and x == 4:
                        return board1[(y-1),x] == 1 and board1[y+1, x] == 1 and board1[y, x-1] == 1 and board1[y, x+1] == 1
                    # Check cattura normale.
                    else:  
                        return (board1[y-1, x] + constraints[y-1, x] > 0) and (board1[y+1, x] + constraints[y+1, x] > 0) or (board1[y, x-1] + constraints[y, x-1] > 0) and (board1[y, x+1] + constraints[y, x+1] > 0)

    @staticmethod
    cdef (float, float) get_utility_bounds():
        return -1.0, 1.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float hardcoded_eval(self, AshtonTablut parent_state, unicode player):
        # Spurio's evaluation function

        cdef DTYPE_t[:,:,:] board = self._board
        cdef DTYPE_t[:,:] allies = board[1]
        cdef DTYPE_t[:,:] constraints = whiteConstraints

        cdef float count = 0.0
        cdef float countKing = 0.0
        cdef int kingX = 0, kingY = 0, y, x, newX, newY
        cdef int numpw = 0, numpb = 0
        cdef float score = 0.0

        for y in range(9):
            for x in range(9):
                if board[0,y,x] == -1:
                    kingY, kingX = y,x

                    # Su
                    newY = y-1
                    while newY >= 0:
                        if allies[newY, x]+constraints[newY, x] > 0:
                            countKing +=1 / (y-newY)
                            break
                        newY -=1

                    # Giu
                    newY = y+1
                    while newY < 9:
                        if allies[newY, x]+constraints[newY, x] > 0:
                            countKing +=1 / (newY-y)
                            break
                        newY +=1
                    # Sinistra
                    newX = x-1
                    while newX >= 0:
                        if allies[y, newX]+constraints[y, newX] > 0:
                            countKing +=1 / (x-newX)
                            break
                        newX -=1

                    # Destra
                    newX = x+1
                    while newX < 9:
                        if allies[y, newX]+constraints[y, newX] > 0:
                            countKing +=1 / (newX-x)
                            break
                        newX +=1

                elif board[0,y,x] == 1:
                    numpw += 1

                    # Su
                    newY = y-1
                    while newY >= 0:
                        if allies[newY, x]+constraints[newY, x] > 0:
                            count +=1 / (y-newY)
                            break
                        newY -=1

                    # Giu
                    newY = y+1
                    while newY < 9:
                        if allies[newY, x]+constraints[newY, x] > 0:
                            count +=1 / (newY-y)
                            break
                        newY +=1
                    # Sinistra
                    newX = x-1
                    while newX >= 0:
                        if allies[y, newX]+constraints[y, newX] > 0:
                            count +=1 / (x-newX)
                            break
                        newX -=1

                    # Destra
                    newX = x+1
                    while newX < 9:
                        if allies[y, newX]+constraints[y, newX] > 0:
                            count +=1 / (newX-x)
                            break
                        newX +=1

                elif board[1,y,x] == 1:
                    numpb += 1

        king_edge_distance = min(kingX,kingY,8-kingX,8-kingY)

        if self._to_move == 'W':
            if self._turn >= 4:
                score = (numpw / 4 -1) * 0.05 - (king_edge_distance / 2 -1) * 0.6 - (numpb / 8 -1) * 0.05 - ((countKing) / 2 -1) * 0.3
            else:
                score = (numpw / 4 -1) * 0.5 - (king_edge_distance / 2 -1) * 0.1 - (numpb / 8 -1) * 0.3 - ((countKing) / 2 -1) * 0.1
        else:
            if self._turn >= 4:
                score = (numpb / 4 -1) * 0.05 + ((count+countKing*5) / 26 -1) * 0.6 - (numpw / 4 -1) * 0.15 + (king_edge_distance / 2 -1) * 0.2
            else:
                score = (numpb / 4 -1) * 0.3 + ((count+countKing*5) / 26 -1) * 0.1 - (numpw / 4 -1) * 0.5 + (king_edge_distance / 2 -1) * 0.1

        return score if player == 'W' else -score

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def convert_board(self):
        cdef np.ndarray[DTYPE_t, ndim = 3] newBoard = np.zeros((4, 9, 9), dtype=DTYPE)
        newBoard[3] = whiteConstraints
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
        cdef np.ndarray moves
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

        return AshtonTablut(board, next_to_move, utility, moves, self._turn+1)

    cpdef int utility(self, unicode player):
        """A state is terminal if it is won or there are no empty squares."""
        return self._utility if player == 'W' else -self._utility

    cpdef bint terminal_test(self):
        """A state is terminal if it is won or there are no empty squares."""
        return self._utility == -1 or self._utility == 1

    cpdef to_move(self):
        """Return the player whose move it is in this state."""
        return self._to_move

    cpdef board(self):
        return self._board

    cpdef turn(self):
        return self._turn

    cpdef eval_fn(self, parent_state, player):
        return self.hardcoded_eval(parent_state, player)

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
cdef class Search:

    cdef long nodes_explored
    cdef long max_depth
    cdef double start_time
    cdef double cutoff_time 
    cdef long current_cutoff_depth 
    cdef bint heuristic_used

    def __init__(self):
        self.nodes_explored = 0
        self.max_depth = 0
        self.start_time = 0.0
        self.cutoff_time = 0.0
        self.current_cutoff_depth = 0
        self.heuristic_used = False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cutoff_test(self, AshtonTablut state, long depth):
        return depth > self.current_cutoff_depth or (get_time()-self.start_time) > self.cutoff_time or state.terminal_test()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float max_value(self, AshtonTablut parent_state, AshtonTablut state, unicode player, float alpha, float beta, long depth):
        cdef float v = -np.inf
        cdef int[:] moves = state.actions()
        cdef int a
        cdef float utility = 0.0

        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, depth)

        if self.cutoff_test(state, depth):
            utility = state.utility(player)
            if utility == 0.0:
                self.heuristic_used = True
                return state.eval_fn(parent_state, player)
            return utility
                
            
        for a in moves:
            next_state = state.result(a)             
            v = max(self.min_value(state, next_state, player, alpha, beta, depth + 1),v)
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float min_value(self, AshtonTablut parent_state, AshtonTablut state, unicode player, float alpha, float beta, long depth):
        cdef float v = np.inf
        cdef int[:] moves = state.actions()
        cdef int a

        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, depth)

        if self.cutoff_test(state, depth):
            utility = state.utility(player)
            if utility == 0.0:
                self.heuristic_used = True
                return state.eval_fn(parent_state, player)
            return utility
        
        for a in moves:
            next_state = state.result(a)           
            v = min(self.max_value(state, next_state, player, alpha, beta, depth + 1),v)

            if v <= alpha:
                return v
            beta = min(beta, v)

        return v 

    def cutoff_search(self, AshtonTablut state, long cutoff_depth=5, double cutoff_time=55.0):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search using time and uses an evaluation function."""

        cdef unicode player = state.to_move()
        
        cdef float best_score = -np.inf
        cdef float beta = np.inf
        cdef int best_action = 0

        cdef float v = np.inf
        cdef int[:] moves = state.actions()
        cdef int a = 0

        cdef AshtonTablut best_next_state

        self.cutoff_time = cutoff_time
        self.current_cutoff_depth = cutoff_depth
        self.start_time = get_time()
        self.nodes_explored = 0
        self.max_depth = 0

        for a in moves:
            next_state = state.result(a)

            if next_state.utility(player) >= 1.0:
                return next_state, a, 1.0, 1, 1, (get_time()-self.start_time)

            v = self.min_value(state, next_state, player, best_score, beta, 1)

            if v >= 1.0:
                return next_state, a, 1.0, self.max_depth, self.nodes_explored, (get_time()-self.start_time)

            if v > best_score:
                best_next_state = next_state
                best_score = v
                best_action = a

            print("Action: {0}, score: {1}".format(number_to_coords(a), v))
        
        return best_next_state, best_action, best_score, self.max_depth, self.nodes_explored, (get_time()-self.start_time)


    def iterative_deepening_search(self, AshtonTablut state, long initial_cutoff_depth=5, double cutoff_time=55.0):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search using time and uses an evaluation function."""

        cdef unicode player = state.to_move()
        
        cdef float best_score = -np.inf
        cdef float beta = np.inf
        cdef int best_action = 0

        cdef float v = np.inf
        cdef int[:] moves = state.actions()
        cdef int a = 0

        cdef AshtonTablut best_next_state

        self.cutoff_time = cutoff_time
        self.current_cutoff_depth = initial_cutoff_depth
        self.start_time = get_time()
        self.nodes_explored = 0
        self.max_depth = 0
        self.heuristic_used = True

        #Pre-check vittoria e ordinamento seguendo la depth precedente
        def key_function(x):
            return x[2]

        next_states = []

        for a in moves:
            next_state = state.result(a)

            if next_state.utility(player) >= 1.0:
                return next_state, a, 1.0, 1, 1, (get_time()-self.start_time)

            next_states.append((next_state, a, np.inf))

        while (get_time()-self.start_time) <= self.cutoff_time and self.heuristic_used:
            best_score = -np.inf
            self.heuristic_used = False
            tmp_states = []

            for next_state, a, v_prec in next_states:
                v = min(self.min_value(state, next_state, player, best_score, beta, 1), v_prec)
                tmp_states.append((next_state, a, v))

                if v >= 1.0:
                    return next_state, a, 1.0, self.max_depth, self.nodes_explored, (get_time()-self.start_time)

                if v > best_score:
                    best_next_state = next_state
                    best_score = v
                    best_action = a
            
            tmp_states = sorted(tmp_states, key=key_function, reverse=True)
            next_states = tmp_states
            self.current_cutoff_depth += 1

            print("New cut-off depth: {0}, best action: {1} ({2}), nodes: {3}".format(self.current_cutoff_depth, number_to_coords(best_action), best_score, self.nodes_explored))
            #for a in moves:
            #    print("Action: {0}, score: {1}".format(number_to_coords(a), v_history[a]))

        return best_next_state, best_action, best_score, self.max_depth, self.nodes_explored, (get_time()-self.start_time)

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
                              [0, 0, 0, 1, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0,-1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

    fake_board[1] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 1, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

    fake_state = AshtonTablut(
        to_move='B', utility=0, board=fake_board, moves=AshtonTablut.legal_actions(fake_board, 'B'))

    print(fake_state.actions())

    st = get_time()
    get_time()
    print("Time: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    ptime.time()
    print("Time: {0} ms".format(1000*(get_time()-st)))

    st = get_time()
    fake_state = fake_state.result(4839)
    print("Result: {0} ms".format(1000*(get_time()-st)))
    print(fake_state.utility('B'))
    fake_state.display()
    