
from games import Game, GameState
import numpy as np


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
blackContraints[:,:] = np.array([ [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [1, 1, 0, 0, 1, 0, 0, 1, 1],
                                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[0,3:6] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [1, 1, 0, 0, 1, 0, 0, 1, 1],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[1,  4] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [1, 1, 0, 0, 1, 0, 0, 1, 1],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)    

blackContraints[8,3:6] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [1, 1, 0, 0, 1, 0, 0, 1, 1],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

blackContraints[7,  4] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [1, 1, 0, 0, 1, 0, 0, 1, 1],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)     

blackContraints[3:6,0] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 1, 0, 0, 1, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[4,  1] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 1, 0, 0, 1, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)  

blackContraints[3:6,8] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 1, 0, 0, 1, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

blackContraints[4,  7] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [1, 1, 0, 0, 1, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)
                                     

initialBoard = np.zeros((4, 9, 9), dtype=np.int8)

#Board[0]: Bianco altro 0
initialBoard[0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 0, 1, 1, 0, 0],
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

#Board[2]: Re 1 altro 0
initialBoard[2] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
        

class AshtonTablut(Game):
    def __init__(self):
        self.initial = GameState(to_move='W', utility=0, board=initialBoard.copy(), moves=[])
        moves = self.legal_actions(self.initial)
        self.initial = GameState(to_move='W', utility=0, board=initialBoard.copy(), moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        if move not in state.moves:
            return state  # Illegal move has no effect
        
        playerBoard = board[0 if state.to_move == 'W' else 1]

        fromYX = move[0]
        toYX = move[1]

        #Controllo se ho mosso il re
        if state.to-move == 'W':
            if board[2][fromYX] == 1:
                playerBoard = board[2]

        playerBoard = playerBoard.copy()

        tmp = playerBoard[fromYX]
        playerBoard[fromYX] = 0
        playerBoard[toYX] = tmp

        eaten = 0

        #Controllo se mangio pedine
        if state.to_move == 'W':
            playerBoard, eaten = self.check_white_eat(playerBoard)
        else:
            playerBoard, eaten = self.check_black_eat(playerBoard)

        to_move = 'W' if state.to_move == 'B' else 'B'
        moves = self.legal_actions(to_move, playerBoard)

        winCheck = self.have_winner(playerBoard) or len(moves) == 0

        utility = 0

        if winCheck:
            utility = 1 if to_move == 'W' else -1

        return GameState(to_move=to_move, utility=utility, board=playerBoard, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'W' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        """Print or otherwise display the state."""
        board = state.board
        print(-board[0]+board[1]-20*board[2]+10*whiteContraints)

    def legal_actions(self, to_move, board):
        if to_move == 'W':
            return self.legal_actions_white(board)
        else:
            return self.legal_actions_black(board)

    def legal_actions_black(self, board):
        legal = []

        #Creo una maschera: pedoni, re, cittadelle
        preMask = board[0] | board[1] | board[2]
        
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
        mask = board[0] | board[1] | board[2] | whiteContraints
        
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
        y,x = np.where(board[2] == 1)
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

        return board, captured

    #La board viene modificata!
    def check_white_eat(self, board, move):#Controllo se il bianco mangia dei pedoni neri
        y,x = move[1]#Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
        captured = 0

        #Il re può fare da spalla
        #Le citadels possono fare da spalla
        allies = board[0] | board[2] | whiteContraints
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

        return board, captured

    def have_draw(self, board):
        if self.stepsWithoutCapturing < 10:
            return False
        #Controllo se ho un certo numero di stati ripetuti
        trovati = 0
        for boardCached in self.drawQueue:
            if np.array_equal(board, boardCached):
                trovati +=1
        
        if trovati > 0:
            return True

        return False

    def have_winner(self, board):
        #White Check
        if self.white_win_check(board):
            return True

        #Black Check
        if self.black_win_check(board):
            return True

        return False

    def white_win_check(self, board):
        #Controllo che il Re sia in un bordo della board
        top = np.sum(board[2, 0])
        down = np.sum(board[2, 8])
        left = np.sum(board[2, :, 0])
        right = np.sum(board[2, :, 8])

        return top == 1 or down == 1 or left == 1 or right == 1

    def black_win_check(self, board):
        #Controllo se il nero ha catturato il re

        #Se il re è sul trono allora 4
        #Se il re è adiacente al trono allora 3 pedoni che lo circondano
        #Altrimenti catturo come pedone normale (citadels possono fare da nemico)

        king = np.where(board[2] == 1)
        
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
