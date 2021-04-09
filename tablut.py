from games import Game, GameState
import numpy as np
import os
import time
import datetime

class TablutConfig:

    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = (4, 9, 9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(6561))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.moves_for_draw = 10

        #Network
        self.num_filters = 32

        ### Self-Play
        self.num_workers = 8  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.threads_per_worker = 2
        self.max_moves = 50  # Maximum number of moves if game is not finished before
        self.max_depth = 2

        # Exploration noise
        self.enable_noise_on_training = True
        self.noise_mean = 0.0
        self.noise_deviation = 0.1

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 300000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.epochs = 10

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate


#Celle non calpestabili: citadels, trono 1 calpestabili 0
#Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels
        
whiteContraints = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

blackContraints = np.zeros((9, 9, 9, 9), dtype=np.int8)

#Celle non calpestabili: citadels, trono 1 calpestabili 0
#Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels
#Maschere speciali per la verifica delle mosse attuabili dal nero
blackContraints[:,:] = np.array([ [0, 0, 0, 1, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[0,3:6] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[1,  4] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)    

blackContraints[8,3:6] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

blackContraints[7,  4] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)     

blackContraints[3:6,0] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 1, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[4,  1] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 1, 0, 0, 1, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)  

blackContraints[3:6,8] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[4,  7] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 1, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)
                                     

initialBoard = np.zeros((2, 9, 9), dtype=np.int8)

#Board[0]: Bianco altro 0
initialBoard[0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 1,-1, 1, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

#Board[1]: Nero altro 0
initialBoard[1] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0], 
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 0, 0, 0, 0, 0, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

def convert_board(board):
    newBoard = np.zeros((4,9,9), dtype=np.int8)
    newBoard[3] = whiteContraints
    newBoard[2] = np.where(board[0] == -1, 1, 0)
    newBoard[1] = board[1]
    newBoard[0] = np.where(board[0] == 1, 1, 0)
    return newBoard

class AshtonTablut(Game):
    def __init__(self):
        self.initial = GameState(to_move='W', utility=0, board=initialBoard.copy(), moves=[])
        moves = self.legal_actions(self.initial.to_move, self.initial.board)
        self.initial = GameState(to_move='W', utility=0, board=initialBoard.copy(), moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        if move not in state.moves:
            return state  # Illegal move has no effect
        
        #Copio la board
        board = state.board.copy()

        fromYX = move[0]
        toYX = move[1]

        #Controllo se ho mosso il re
        move_board = board[0 if state.to_move == 'W' else 1]

        tmp = move_board[fromYX]
        move_board[fromYX] = 0
        move_board[toYX] = tmp

        eaten = 0

        #Controllo se mangio pedine
        if state.to_move == 'W':
            eaten = self.check_white_eat(board, move)
        else:
            eaten = self.check_black_eat(board, move)

        to_move = 'W' if state.to_move == 'B' else 'B'
        moves = self.legal_actions(board, to_move)
        
        winCheck = self.have_winner(board, state.to_move) or len(moves) == 0
        utility = 0
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

    def legal_actions(self, board, to_move):
        if to_move == 'W':
            return self.legal_actions_white(board)
        else:
            return self.legal_actions_black(board)

    def legal_actions_black(self, board):
        legal = []

        #Creo una maschera: pedoni, re, cittadelle
        preMask = board[0] | board[1]
        
        #Seleziono i pedoni del giocatore
        pedoni = np.where(board[1] == 1)
        
        for y,x in zip(pedoni[0], pedoni[1]):
            #Seleziono le celle adiacenti (no diagonali)
            #Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

            #Casi specifici per la maschera delle citadels
            mask = preMask | blackContraints[y,x]
                
            #Su
            for newY in reversed(range(y)):
                if mask[newY,x] == 0:
                    legal.append(((y,x), (newY,x)))
                else:
                    break

            #Giu
            for newY in range(y+1,9):
                if mask[newY,x] == 0:
                    legal.append(((y,x), (newY,x)))
                else:
                    break

            #Sinistra
            for newX in reversed(range(x)):
                if mask[y,newX] == 0:
                    legal.append(((y,x), (y,newX)))
                else:
                    break

            #Destra
            for newX in range(x+1,9):
                if mask[y,newX] == 0:
                    legal.append(((y,x), (y,newX)))
                else:
                    break

        return legal

    def legal_actions_white(self, board):
        legal = []

        #Creo una maschera: pedoni, re, cittadelle
        mask = board[0] | board[1] | whiteContraints
        
        #Seleziono i pedoni del giocatore
        pedoni = np.where(board[0] == 1)
        
        for y,x in zip(pedoni[0], pedoni[1]):
            #Seleziono le celle adiacenti (no diagonali)
            #Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

            #Su
            for newY in reversed(range(y)):
                if mask[newY,x] == 0:
                    legal.append(((y,x), (newY,x)))
                else:
                    break

            #Giu
            for newY in range(y+1,9):
                if mask[newY,x] == 0:
                    legal.append(((y,x), (newY,x)))
                else:
                    break

            #Sinistra
            for newX in reversed(range(x)):
                if mask[y,newX] == 0:
                    legal.append(((y,x), (y,newX)))
                else:
                    break

            #Destra
            for newX in range(x+1,9):
                if mask[y,newX] == 0:
                    legal.append(((y,x), (y,newX)))
                else:
                    break

        #Mosse del Re
        y,x = np.where(board[0] == -1)
        y,x = int(y), int(x)
        #Seleziono le celle adiacenti (no diagonali)
        #Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

        #Su
        for newY in reversed(range(y)):
            if mask[newY,x] == 0:
                legal.append(((y,x), (newY,x)))
            else:
                break

        #Giu
        for newY in range(y+1,9):
            if mask[newY,x] == 0:
                legal.append(((y,x), (newY,x)))
            else:
                break

        #Sinistra
        for newX in reversed(range(x)):
            if mask[y,newX] == 0:
                legal.append(((y,x), (y,newX)))
            else:
                break

        #Destra
        for newX in range(x+1,9):
            if mask[y,newX] == 0:
                legal.append(((y,x), (y,newX)))
            else:
                break
        

        return legal

    #La board viene modificata!
    def check_black_eat(self, board, move):#Controllo se il nero mangia dei pedoni bianchi
        y,x = move[1]#Dove è finita la pedina nera che dovrà catturare uno o più pedoni bianchi?
        captured = 0

        #Le citadels possono fare da spalla
        allies = board[1] | whiteContraints
        enemies = board[0]

        #Seleziono le quattro terne di controllo
        lookUp = np.array([allies[y-2:y+1,x], enemies[y-2:y+1,x]])
        lookDown = np.array([allies[y:y+3,x], enemies[y:y+3,x]])
        lookLeft = np.array([allies[y,x-2:x+1], enemies[y,x-2:x+1]])
        lookRight = np.array([allies[y,x:x+3], enemies[y,x:x+3]])

        #print("LU: {0}, LD: {1}, LL: {2}, LR: {3}".format(lookUp, lookDown, lookLeft, lookRight))

        captureCheck = np.array([[1,0,1], [0,1,0]])

        if np.array_equal(lookUp, captureCheck):
            #print("captured white: {0}".format((y-1,x)))
            board[0, y-1, x] = 0
            captured +=1
        if np.array_equal(lookDown, captureCheck):
            #print("captured white: {0}".format((y+1,x)))
            board[0, y+1, x] = 0
            captured +=1
        if np.array_equal(lookLeft, captureCheck):
            #print("captured white: {0}".format((y,x-1)))
            board[0, y, x-1] = 0
            captured +=1
        if np.array_equal(lookRight, captureCheck):
            #print("captured white: {0}".format((y,x+1)))
            board[0, y, x+1] = 0
            captured +=1

        return captured

    #La board viene modificata!
    def check_white_eat(self, board, move):#Controllo se il bianco mangia dei pedoni neri
        y,x = move[1]#Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
        captured = 0

        #Il re può fare da spalla
        #Le citadels possono fare da spalla
        allies = np.where(board[0] != 0, 1, 0) | whiteContraints
        enemies = board[1]

        #Seleziono le quattro terne di controllo
        lookUp = np.array([allies[y-2:y+1,x], enemies[y-2:y+1,x]])
        lookDown = np.array([allies[y:y+3,x], enemies[y:y+3,x]])
        lookLeft = np.array([allies[y,x-2:x+1], enemies[y,x-2:x+1]])
        lookRight = np.array([allies[y,x:x+3], enemies[y,x:x+3]])

        captureCheck1 = np.array([[1,0,1], [0,1,0]])

        if np.array_equal(lookUp, captureCheck1):
            #print("captured black: {0}".format((y-1,x)))
            board[1, y-1, x] = 0
            captured +=1
        if np.array_equal(lookDown, captureCheck1):
            #print("captured black: {0}".format((y+1,x)))
            board[1, y+1, x] = 0
            captured +=1
        if np.array_equal(lookLeft, captureCheck1):
            #print("captured black: {0}".format((y,x-1)))
            board[1, y, x-1] = 0
            captured +=1
        if np.array_equal(lookRight, captureCheck1):
            #print("captured black: {0}".format((y,x+1)))
            board[1, y, x+1] = 0
            captured +=1

        return captured

    def have_winner(self, board, to_move):
        if to_move == 'W':
            return self.white_win_check(board) 
        else:
            return self.black_win_check(board)

    def white_win_check(self, board):
        #Controllo che il Re sia in un bordo della board
        y,x = np.where(board[0] == -1)
        y,x = int(y), int(x)

        return x == 0 or x == 8 or y == 0 or y == 8

    def black_win_check(self, board):
        #Controllo se il nero ha catturato il re

        #Se il re è sul trono allora 4
        #Se il re è adiacente al trono allora 3 pedoni che lo circondano
        #Altrimenti catturo come pedone normale (citadels possono fare da nemico)

        king = np.where(board[0] == -1)
        
        if king == (4,4):#Re sul trono. Controllo i bordi (3,4), (4,3), (4,5), (5,4)
            if board[1, 3, 4] == 1 and board[1, 4, 3] == 1 and board[1, 4, 5] == 1 and board[1, 5, 4] == 1:
                return True
        
        elif king in ((3,4), (4,3), (4,5), (5,4)):#Re adiacente al trono: controllo se sono presenti nemici intorno
            #Aggiungo il trono alle pedine nemiche (in realtà aggiungo anche le citadels ma non influenzano)
            enemies = board[1] | whiteContraints
            y,x = king
            if enemies[y-1, x] == 1 and enemies[y+1, x] == 1 and enemies[y, x-1] == 1 and enemies[y, x+1] == 1:
                return True

        else:#Check cattura normale.
            #Aggiungo i contraints
            enemies = board[1] | whiteContraints
            y,x = king
            if enemies[y-1, x] == 1 and enemies[y+1, x] == 1 or enemies[y, x-1] == 1 and enemies[y, x+1] == 1:
                return True

        return False
        