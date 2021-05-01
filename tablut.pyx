#!python 
#cython: embedsignature=True, binding=True, language_level=3, boundscheck=False, wraparound=False, cdivision=False 
#distutils: extra_compile_args = -march=native
#tag: numpy

import datetime
import time as ptime
import os
import random

import tflite_runtime.interpreter as tflite

from tablutconfig import TablutConfig

cimport cython

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdlib cimport srand, rand, RAND_MAX
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cpython cimport array
import array

import numpy as np
cimport numpy as np

# We now need to fix a datatype for our arrays.
DTYPE = np.float32
ctypedef float DTYPE_t

def init_rand():
    cdef long int seed = <long int>get_time()
    srand(<int>seed)

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

blackConstraints[1, 4] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
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

blackConstraints[7, 4] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
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

blackConstraints[4, 1] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
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

blackConstraints[4, 7] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

#Effettuo reshaping: Così ho la rete neurale con ingresso diretto
cdef np.ndarray initialBoard = np.zeros((1, 9, 9, 4), dtype=DTYPE)

initialBoard[0, :, :, 0] = np.array(
                           [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

initialBoard[0, :, :, 1] = np.array(
                           [[0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=DTYPE)

initialBoard[0, :, :, 2] = np.array(
                           [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

initialBoard[0, :, :, 3] = np.array(
                           [[0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=DTYPE)

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
    cdef int* _moves
    cdef int _moves_length
    cdef long _turn

    cdef HeuristicFunction _heuristic
    
    def __init__(self, np.ndarray board, unicode to_move, long turn = 0, HeuristicFunction heuristic=None):
        cdef bint winCheck
        cdef unicode prev_to_move = 'W' if to_move == 'B' else 'B'
        
        self._board = board
        self._to_move = to_move
        self._turn = turn
        self._utility = 0

        if heuristic is None:
            heuristic = HeuristicFunction()

        self._heuristic = heuristic

        winCheck = AshtonTablut.have_winner(board, prev_to_move)

        if not winCheck:
            self._moves, self._moves_length = AshtonTablut.legal_actions(board, to_move)
            winCheck = winCheck or self._moves_length == 0
        else:
            self._moves, self._moves_length = NULL, 0

        if winCheck:
            self._utility = 1 if prev_to_move == 'W' else -1
            
    def __dealloc__(self):
        if self._moves != NULL:
            PyMem_Free(self._moves)

    @staticmethod
    def get_initial(heuristic=None):
        return AshtonTablut(initialBoard, 'W', 0, heuristic)

    @staticmethod
    def get_initial_board():
        return initialBoard.copy()

    @staticmethod
    def parse_board(board, player, turn, heuristic):
        if board.shape == (1, 9, 9, 4):
            return AshtonTablut(board, player[0], turn, heuristic)
        return None

    @staticmethod
    def coords_to_num(l,k,j,i):
        return coords_to_number(l,k,j,i)

    @staticmethod
    def num_to_coords(number):
        return number_to_coords(number)

    @staticmethod
    cdef (int*, int) legal_actions(DTYPE_t[:,:,:,:] board, unicode to_move):
        if to_move == 'W':
            return AshtonTablut.legal_actions_white(board)
        else:
            return AshtonTablut.legal_actions_black(board)

    @staticmethod
    cdef (int*, int) legal_actions_black(DTYPE_t[:,:,:,:] board):
        cdef int* legal = <int*> PyMem_Malloc(256 * sizeof(int))
        cdef int i = 0
        cdef int tmp, r = 0

        cdef DTYPE_t[:,:,:,:] constraints = blackConstraints

        # Seleziono i pedoni del giocatore
        cdef int y, x, newY, newX

        for y in range(9):
            for x in range(9):
                #Seleziono solo i neri
                if board[0, y, x, 1] != 1:
                    continue

                # Seleziono le celle adiacenti (no diagonali)
                # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)
                
                # Su
                newY = y-1
                while newY >= 0:
                    if board[0, newY, x, 0] + board[0, newY, x, 1] + board[0, newY, x, 2] + constraints[y, x, newY, x] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY -=1

                # Giu
                newY = y+1
                while newY < 9:
                    if board[0, newY, x, 0] + board[0, newY, x, 1] + board[0, newY, x, 2] + constraints[y, x, newY, x] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY +=1

                # Sinistra
                newX = x-1
                while newX >= 0:
                    if board[0, y, newX, 0] + board[0, y, newX, 1] + board[0, y, newX, 2] + constraints[y, x, y, newX] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX -=1

                # Destra
                newX = x+1
                while newX < 9:
                    if board[0, y, newX, 0] + board[0, y, newX, 1] + board[0, y, newX, 2] + constraints[y, x, y, newX] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX +=1

        #Shuffle
        for k in range(i//4):
            r = <int> rand() % i
            tmp = legal[k]
            legal[k] = legal[r]
            legal[r] = tmp

        return legal, i

    @staticmethod
    cdef (int*, int) legal_actions_white(DTYPE_t[:,:,:,:] board):
        cdef int* legal = <int*> PyMem_Malloc(256 * sizeof(int))
        cdef int i = 0
        cdef int tmp, r = 0

        cdef int y, x, newY, newX

        for y in range(9):
            for x in range(9):
                #Seleziono solo le pedine bianche o il re
                if board[0, y, x, 0] + board[0, y, x, 2] < 1:
                    continue

                # Seleziono le celle adiacenti (no diagonali)
                # Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

                # Su
                newY = y-1
                while newY >= 0:
                    if board[0, newY, x, 0] + board[0, newY, x, 1] + board[0, newY, x, 2] + board[0, newY, x, 3] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY -=1

                # Giu
                newY = y+1
                while newY < 9:
                    if board[0, newY, x, 0] + board[0, newY, x, 1] + board[0, newY, x, 2] + board[0, newY, x, 3] == 0:
                        legal[i] = coords_to_number(y, x, newY, x)
                        i+=1
                    else:
                        break
                    newY +=1

                # Sinistra
                newX = x-1
                while newX >= 0:
                    if board[0, y, newX, 0] + board[0, y, newX, 1] + board[0, y, newX, 2] + board[0, y, newX, 3] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX -=1

                # Destra
                newX = x+1
                while newX < 9:
                    if board[0, y, newX, 0] + board[0, y, newX, 1] + board[0, y, newX, 2] + board[0, y, newX, 3] == 0:
                        legal[i] = coords_to_number(y, x, y, newX)
                        i+=1
                    else:
                        break
                    newX +=1

        #Shuffle
        for k in range(i//4):
            r = <int> rand() % i
            tmp = legal[k]
            legal[k] = legal[r]
            legal[r] = tmp

        return legal, i

    # La board viene modificata!
    @staticmethod
    cdef int check_eat(DTYPE_t[:,:,:,:] board, unicode to_move, int move):
        cdef int eaten = 0
        if to_move == 'W':
            eaten = AshtonTablut.check_white_eat(board, move)
        else:
            eaten = AshtonTablut.check_black_eat(board, move)

        return eaten

    # La board viene modificata!
    @staticmethod
    # Controllo se il bianco mangia dei pedoni neri
    cdef int check_white_eat(DTYPE_t[:,:,:,:] board, int move):
        # Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
        cdef (int,int,int,int) c = number_to_coords(move)
        cdef int y = c[2]
        cdef int x = c[3]
        cdef int captured = 0

        # Controlli U,D,L,R
        cdef bint lookUp = y-2 >= 0 and board[0, y-2, x, 0] + board[0, y-2, x, 2] + board[0, y-2, x, 3] > 0 and board[0, y-1, x, 1] == 1
        cdef bint lookDown = y+2 < 9 and board[0, y+2, x, 0] + board[0, y+2, x, 2] + board[0, y+2, x, 3] > 0 and board[0, y+1, x, 1] == 1
        cdef bint lookLeft = x-2 >= 0 and board[0, y, x-2, 0] + board[0, y, x-2, 2] + board[0, y, x-2, 3] > 0 and board[0, y, x-1, 1] == 1
        cdef bint lookRight = x+2 < 9 and board[0, y, x+2, 0] + board[0, y, x+2, 2] + board[0, y, x+2, 3] > 0 and board[0, y, x+1, 1] == 1

        if lookUp:
            board[0, y-1, x, 1] = 0
            captured += 1
        if lookDown:
            board[0, y+1, x, 1] = 0
            captured += 1
        if lookLeft:
            board[0, y, x-1, 1] = 0
            captured += 1
        if lookRight:
            board[0, y, x+1, 1] = 0
            captured += 1

        return captured

    # La board viene modificata!
    @staticmethod
    # Controllo se il nero mangia dei pedoni bianchi
    cdef int check_black_eat(DTYPE_t[:,:,:,:] board, int move):
        # Dove è finita la pedina nera che dovrà catturare uno o più pedoni bianchi?
        cdef (int,int,int,int) c = number_to_coords(move)
        cdef int y = c[2]
        cdef int x = c[3]
        cdef int captured = 0

        # Controlli U,D,L,R
        cdef bint lookUp = y-2 >= 0 and board[0, y-2, x, 1] + board[0, y-2, x, 3] > 0 and board[0, y-1, x, 0] == 1
        cdef bint lookDown = y+2 < 9 and board[0, y+2, x, 1] + board[0, y+2, x, 3] > 0 and board[0, y+1, x, 0] == 1
        cdef bint lookLeft = x-2 >= 0 and board[0, y, x-2, 1] + board[0, y, x-2, 3] > 0 and board[0, y, x-1, 0] == 1
        cdef bint lookRight = x+2 < 9 and board[0, y, x+2, 1] + board[0, y, x+2, 3] > 0 and board[0, y, x+1, 0] == 1

        if lookUp:
            board[0, y-1, x, 0] = 0
            captured += 1
        if lookDown:
            board[0, y+1, x, 0] = 0
            captured += 1
        if lookLeft:
            board[0, y, x-1, 0] = 0
            captured += 1
        if lookRight:
            board[0, y, x+1, 0] = 0
            captured += 1

        return captured

    @staticmethod
    cdef bint have_winner(DTYPE_t[:,:,:,:] board, unicode to_move):
        if to_move == 'W':
            return AshtonTablut.white_win_check(board)
        else:
            return AshtonTablut.black_win_check(board)

    @staticmethod
    cdef bint white_win_check(DTYPE_t[:,:,:,:] board):
        # Controllo che il Re sia in un bordo della board
        cdef long k

        for k in range(9):
            if board[0, 0, k, 2] == 1:
                return True
            if board[0, 8, k, 2] == 1:
                return True
            if board[0, k, 0, 2] == 1:
                return True
            if board[0, k, 8, 2] == 1:
                return True

        return False

    @staticmethod
    cdef bint black_win_check(DTYPE_t[:,:,:,:] board):
        # Controllo se il nero ha catturato il re

        # Se il re è sul trono allora 4
        # Se il re è adiacente al trono allora 3 pedoni che lo circondano
        # Altrimenti catturo come pedone normale (citadels possono fare da nemico)
        cdef long y,x

        for y in range(9):
            for x in range(9):
                if board[0, y, x, 2] == 1:
                    # Re sul trono. Controllo i bordi (3,4), (4,3), (4,5), (5,4)
                    if y==4 and x==4:
                        return board[0, 3, 4, 1] == 1 and board[0, 4, 3, 1] == 1 and board[0, 4, 5, 1] == 1 and board[0, 5, 4, 1] == 1
                    # Re adiacente al trono: controllo se sono presenti nemici intorno
                    elif y == 3 and x == 4 or y == 4 and x == 3 or y == 4 and x == 5 or y == 5 and x == 4:
                        return (board[0, y-1, x, 1] + board[0, y-1, x, 3] > 0) and (board[0, y+1, x, 1] + board[0, y+1, x, 3] > 0) and (board[0, y, x-1, 1] + board[0, y, x-1, 3] > 0) and (board[0, y, x+1, 1] + board[0, y, x+1, 3] > 0)
                    # Check cattura normale.
                    else:  
                        return (board[0, y-1, x, 1] + board[0, y-1, x, 3] > 0) and (board[0, y+1, x, 1] + board[0, y+1, x, 3] > 0) or (board[0, y, x-1, 1] + board[0, y, x-1, 3] > 0) and (board[0, y, x+1, 1] + board[0, y, x+1, 3] > 0)

    @staticmethod
    cdef (float, float) get_utility_bounds():
        return -1.0, 1.0

    def actions(self):
        cdef np.ndarray[int, ndim=1] actions = np.zeros((self._moves_length, ), dtype=np.int32)

        for i in range(self._moves_length):
            actions[i] = self._moves[i]

        return actions
    
    cdef actions_length(self):
        return self._moves_length

    cpdef result(self, int move):
        """Return the state that results from making a move from a state."""
        cdef unicode next_to_move

        # Copio la board
        cdef np.ndarray[DTYPE_t, ndim = 4] board = self._board.copy()

        cdef (int,int,int,int) c = number_to_coords(move)
        cdef int y0 = c[0]
        cdef int x0 = c[1]
        cdef int y1 = c[2]
        cdef int x1 = c[3]

        cdef DTYPE_t tmp

        if self._to_move == 'B':
            tmp = board[0, y0, x0, 1]
            board[0, y0, x0, 1] = 0
            board[0, y1, x1, 1] = tmp
        else:
            if board[0, y0, x0, 2] == 1:
                board[0, y0, x0, 2] = 0
                board[0, y1, x1, 2] = 1
            else:
                tmp = board[0, y0, x0, 0]
                board[0, y0, x0, 0] = 0
                board[0, y1, x1, 0] = tmp

        # Controllo se mangio pedine
        AshtonTablut.check_eat(board, self._to_move, move)

        next_to_move = 'W' if self._to_move == 'B' else 'B'

        return AshtonTablut(board, next_to_move, self._turn+1, self._heuristic)

    cpdef int utility(self, unicode player):
        """A state is terminal if it is won or there are no empty squares."""
        return self._utility if player == 'W' else -self._utility

    cpdef bint terminal_test(self):
        """A state is terminal if it is won or there are no empty squares."""
        return self._utility <= -1 or self._utility >= 1

    cpdef to_move(self):
        """Return the player whose move it is in this state."""
        return self._to_move

    cpdef board(self):
        return self._board

    cpdef turn(self):
        return self._turn

    cdef eval_fn(self, player):
        return self._heuristic.evalutate(self, player)

    def display(self):
        """Print or otherwise display the state."""
        cdef np.ndarray[DTYPE_t, ndim = 3] board = np.moveaxis(self._board[0], -1, 0)
        return str(-board[0]+board[1]-20*board[2]+10*board[3])

# ______________________________________________________________________________
### Players for Games

def random_player(AshtonTablut game):
    """A player that chooses a legal move at random."""
    cdef int action
    cdef int index = 0
    cdef int actions_length = game.actions_length()
    cdef int* actions = game._moves
    if actions:
        index = <int> rand() % actions_length
        action = actions[index]
        return action
    return None

#------------------------------ Search -------------------------------------------------------
cdef class Search:

    cdef long nodes_explored
    cdef long max_depth
    cdef double start_time
    cdef double cutoff_time 
    cdef long current_cutoff_depth

    def __init__(self):
        self.nodes_explored = 0
        self.max_depth = 0
        self.start_time = 0.0
        self.cutoff_time = 0.0
        self.current_cutoff_depth = 0

    cdef float max_value(self, AshtonTablut state, unicode player, float alpha, float beta, long depth):
        cdef int* moves = state._moves
        cdef int moves_length = state._moves_length
        cdef float v = -np.inf
        cdef int a
        cdef bint terminal = state.terminal_test()

        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, depth)

        if depth > self.current_cutoff_depth or terminal or (get_time()-self.start_time) > self.cutoff_time:
            if terminal:
                return state.utility(player)
            
            return state.eval_fn(player)
                
        for i in range(moves_length):
            a = moves[i]
            next_state = state.result(a)             

            v = max(self.min_value(next_state, player, alpha, beta, depth + 1),v)
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    cdef float min_value(self, AshtonTablut state, unicode player, float alpha, float beta, long depth):
        cdef int* moves = state._moves
        cdef int moves_length = state._moves_length
        cdef float v = np.inf
        cdef int a
        cdef bint terminal = state.terminal_test()

        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, depth)

        if depth > self.current_cutoff_depth or terminal or (get_time()-self.start_time) > self.cutoff_time:
            if terminal:
                return state.utility(player)
            
            return state.eval_fn(player)

        for i in range(moves_length):
            a = moves[i]
            next_state = state.result(a)           

            v = min(self.max_value(next_state, player, alpha, beta, depth + 1),v)
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def cutoff_search(self, AshtonTablut state, long cutoff_depth=5, double cutoff_time=55.0):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search using time and uses an evaluation function."""

        cdef unicode player = state._to_move
        
        cdef float best_score = -np.inf
        cdef float beta = np.inf
        cdef int best_action = 0
        cdef AshtonTablut best_next_state = None
        cdef AshtonTablut next_state = None
        
        cdef int* moves = state._moves
        cdef int moves_length = state._moves_length
        cdef float v
        cdef int a

        self.cutoff_time = cutoff_time
        self.current_cutoff_depth = cutoff_depth
        self.start_time = get_time()
        self.nodes_explored = 0
        self.max_depth = 0

        for i in range(moves_length):
            a = moves[i]
            next_state = state.result(a)

            if next_state.utility(player) >= 1:
                return next_state, a, 1.0, 1, 1, (get_time()-self.start_time)

            v = self.min_value(next_state, player, best_score, beta, 1)

            if v >= 1:
                return next_state, a, 1.0, self.max_depth, self.nodes_explored, (get_time()-self.start_time)

            if v > best_score:
                best_next_state = next_state
                best_score = v
                best_action = a

        return best_next_state, best_action, best_score, self.max_depth, self.nodes_explored, (get_time()-self.start_time)

    def iterative_deepening_search(self, AshtonTablut state, long initial_cutoff_depth=2, double cutoff_time=55.0):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search using time and uses an evaluation function."""

        cdef unicode player = state._to_move
        
        cdef float best_score = -np.inf
        cdef float beta = np.inf
        cdef int best_action = 0

        cdef int* moves = state._moves
        cdef int moves_length = state._moves_length
        cdef float v 
        cdef int a    

        cdef ActionStore store = ActionStore()
        cdef ActionStore next_store

        cdef AshtonTablut tmp_best_next_state = None
        cdef AshtonTablut best_next_state = None
        cdef AshtonTablut next_state = None

        self.cutoff_time = cutoff_time
        self.current_cutoff_depth = initial_cutoff_depth
        self.start_time = get_time()
        self.nodes_explored = 0
        self.max_depth = 0

        for i in range(moves_length):
            a = moves[i]
            next_state = state.result(a)
            store.add(a, state.eval_fn(player))
        
        while (get_time()-self.start_time) <= self.cutoff_time:
            next_store = ActionStore()
            
            for i in range(store.size()):
                a = store.actions[i]
                next_state = state.result(a)

                v = self.min_value(next_state, player, best_score, beta, 1)

                if (get_time()-self.start_time) > self.cutoff_time:
                    break

                next_store.add(a, v)

            if next_store.size() > 0:
                store = next_store
                if (get_time()-self.start_time) <= self.cutoff_time:
                    a = store.actions[0]
                    v = store.utils[0]
                    if v >= 1.0:
                        return state.result(a), a, v, self.max_depth, self.nodes_explored, (get_time()-self.start_time)


            self.current_cutoff_depth += 1

            #print("New cut-off depth: {0}, best action: {1} ({2}), nodes: {3}".format(self.current_cutoff_depth, number_to_coords(store.actions[0]), store.utils[0], self.nodes_explored))

        if store.size() > 0:
            best_action = store.actions[0]
            best_score = store.utils[0]
            best_next_state = state.result(best_action)

        return best_next_state, best_action, best_score, self.max_depth, self.nodes_explored, (get_time()-self.start_time)


cdef class ActionStore:
    cdef array.array actions
    cdef array.array utils

    def __init__(self):
        self.actions = array.array('i')
        self.utils = array.array('f')
    
    cpdef add(self, int action, float value):
        cdef int idx = 0
        
        while idx < len(self.actions) and value <= self.utils[idx]:
            idx +=1
        
        self.actions.insert(idx, action)
        self.utils.insert(idx, value)

    cpdef size(self):
        return len(self.actions)


#------------------------------ Heuristic function --------------------------------------------

cdef class HeuristicFunction:

    cdef float evalutate(self, AshtonTablut state, unicode player):
        return state.utility(player)

cdef class OldSchoolHeuristicFunction(HeuristicFunction):

    cdef float evalutate(self, AshtonTablut state, unicode player):
        # Spurio's evaluation function ( Modificata :) )

        cdef DTYPE_t[:,:,:,:] board = state.board()

        cdef float count = 0.0
        cdef float countKing = 0.0
        cdef int kingX = 0, kingY = 0, y, x, newX, newY
        cdef int numpw = 0, numpb = 0
        cdef float score = 0.0

        for y in range(9):
            for x in range(9):
                if board[0, y, x, 2] == 1:
                    kingY, kingX = y,x

                    # Su
                    newY = y-1
                    while newY >= 0:
                        if board[0, newY, x, 1]+board[0, newY, x, 3] > 0:
                            countKing +=1 / (y-newY)
                            break
                        newY -=1

                    # Giu
                    newY = y+1
                    while newY < 9:
                        if board[0, newY, x, 1]+board[0, newY, x, 3] > 0:
                            countKing +=1 / (newY-y)
                            break
                        newY +=1
                    # Sinistra
                    newX = x-1
                    while newX >= 0:
                        if board[0, y, newX, 1]+board[0, y, newX, 3] > 0:
                            countKing +=1 / (x-newX)
                            break
                        newX -=1

                    # Destra
                    newX = x+1
                    while newX < 9:
                        if board[0, y, newX, 1]+board[0, y, newX, 3] > 0:
                            countKing +=1 / (newX-x)
                            break
                        newX +=1

                elif board[0, y, x, 0] == 1:
                    numpw += 1

                    # Su
                    newY = y-1
                    while newY >= 0:
                        if board[0, newY, x, 1]+board[0, newY, x, 3] > 0:
                            count +=1 / (y-newY)
                            break
                        newY -=1

                    # Giu
                    newY = y+1
                    while newY < 9:
                        if board[0, newY, x, 1]+board[0, newY, x, 3] > 0:
                            count +=1 / (newY-y)
                            break
                        newY +=1
                    # Sinistra
                    newX = x-1
                    while newX >= 0:
                        if board[0, y, newX, 1]+board[0, y, newX, 3] > 0:
                            count +=1 / (x-newX)
                            break
                        newX -=1

                    # Destra
                    newX = x+1
                    while newX < 9:
                        if board[0, y, newX, 1]+board[0, y, newX, 3] > 0:
                            count +=1 / (newX-x)
                            break
                        newX +=1

                elif board[0, y, x, 1] == 1:
                    numpb += 1

        king_edge_distance = min(kingX,kingY,8-kingX,8-kingY)

        if state.to_move() == 'W':
            if state.turn() >= 4:
                score = (numpw / 4 -1) * 0.1 - (king_edge_distance / 2 -1) * 0.5 - (numpb / 8 -1) * 0.1 - ((countKing) / 2 -1) * 0.3
            else:
                score = (numpw / 4 -1) * 0.5 - (king_edge_distance / 2 -1) * 0.1 - (numpb / 8 -1) * 0.3 - ((countKing) / 2 -1) * 0.1
        else:
            if state.turn() >= 4:
                score = (numpb / 4 -1) * 0.05 + ((count+countKing*5) / 26 -1) * 0.5 - (numpw / 4 -1) * 0.15 + (king_edge_distance / 2 -1) * 0.3
            else:
                score = (numpb / 4 -1) * 0.3 + ((count+countKing*5) / 26 -1) * 0.1 - (numpw / 4 -1) * 0.5 + (king_edge_distance / 2 -1) * 0.1

        return score if player == 'W' else -score

cdef class NeuralHeuristicFunction(HeuristicFunction):

    cdef object config
    cdef str model_path
    cdef bint interpreter_initialized

    cdef object interpreter
    cdef np.ndarray input_shape
    cdef int index_in_0, index_out_0

    def __init__(self, config):
        self.config = config
        self.interpreter_initialized = False

        folder = self.config.folder
        filename = self.config.tflite_model
        self.model_path = os.path.join(folder, filename)

    def init_tflite(self):
        if not os.path.isfile(self.model_path):
            return False

        self.interpreter = tflite.Interpreter(
            model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        tflite_input_details = self.interpreter.get_input_details()
        tflite_output_details = self.interpreter.get_output_details()

        self.input_shape = tflite_input_details[0]['shape']
        self.index_in_0 = tflite_input_details[0]['index']
        #self.index_in_1 = tflite_input_details[1]['index']
        self.index_out_0 = tflite_output_details[0]['index']

        self.interpreter_initialized = True

        return True

    def set_model_path(self, model_path):
        self.model_path = model_path

    def initialized(self):
        return self.interpreter_initialized

    cdef float tflite_eval(self, AshtonTablut state, unicode player):
        cdef np.ndarray board0 = state.board()
        #cdef np.ndarray board1 = next_state.board()
        cdef float v

        self.interpreter.set_tensor(self.index_in_0, board0)
        #self.interpreter.set_tensor(self.index_in_1, board1)

        self.interpreter.invoke()

        v = self.interpreter.get_tensor(self.index_out_0)[0][0]
        v = min(max(v, -1.0), 1.0)

        return v if player == 'W' else -v

    cdef float evalutate(self, AshtonTablut state, unicode player):
        if self.interpreter_initialized:
            return self.tflite_eval(state, player)
        return state.utility(player)

cdef class MixedHeuristicFunction(HeuristicFunction):
    cdef OldSchoolHeuristicFunction old_eval
    cdef NeuralHeuristicFunction neural_eval

    cdef public float alpha, cutoff

    def __init__(self, config, alpha = 1.0, cutoff = 0.0):
        self.alpha = max(min(alpha, 1.0), 0.0)
        self.cutoff = max(min(cutoff, 1.0), 0.0)
        self.old_eval = OldSchoolHeuristicFunction()
        self.neural_eval = NeuralHeuristicFunction(config)

    def init_tflite(self):
        return self.neural_eval.init_tflite()

    def initialized(self):
        return self.neural_eval.initialized()

    def set_model_path(self, model_path):
        self.neural_eval.set_model_path(model_path)

    def set_alpha(self, alpha):
        self.alpha = max(min(alpha, 1.0), 0.0)

    def set_cutoff(self, cutoff):
        self.cutoff = max(min(cutoff, 1.0), 0.0)

    cdef float evalutate(self, AshtonTablut state, unicode player):
        if self.neural_eval.interpreter_initialized and self.alpha >= self.cutoff:
            return self.neural_eval.tflite_eval(state, player) * self.alpha + self.old_eval.evalutate(state, player) * (1-self.alpha)

        return self.old_eval.evalutate(state, player)

#------------------------------ Utils ---------------------------------------------------------
cdef inline double get_time():
    cdef timespec ts
    cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    return current

#------------------------------ Test ---------------------------------------------------------
def test():
    cdef AshtonTablut g, fake_state

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
    
    fake_board = np.zeros((1, 9, 9, 4), dtype=DTYPE)

    fake_board[0,:,:,0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

    fake_board[0,:,:,1] = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 1, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 1, 0, 1, 0, 1, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=DTYPE)

    fake_board[0,:,:,2] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=DTYPE)

    fake_board[0,:,:,3] = whiteConstraints

    fake_state = AshtonTablut.parse_board(fake_board, 'W', 21, OldSchoolHeuristicFunction())

    print(fake_state.actions())

    st = get_time()
    get_time()
    print("Time: {0} ms".format(1000*(get_time()-st)))

    #st = get_time()
    #a = time(NULL)
    #print("Time2: {0} ms ({1})".format(1000*(get_time()-st), a))

    #st = get_time()
    #ptime.time()
    #print("Time: {0} ms".format(1000*(get_time()-st)))

    search = Search()
    best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
                    state=fake_state, initial_cutoff_depth=3, cutoff_time=5)
    best_action = AshtonTablut.num_to_coords(best_action)
    print("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}, Captured: {7}".format(fake_state.to_move(), (best_action[0], best_action[1]), (best_action[2], best_action[3]), search_time, max_depth, nodes_explored, best_score, 0))

    print(fake_state.display())
    