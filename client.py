import socket
import argparse
import logging
import json
import numpy as np

from tablut import AshtonTablut, Search, NeuralHeuristicFunction, OldSchoolHeuristicFunction
from tablutconfig import TablutConfig

# Modified from: https://github.com/Jippiter/TablutGo/blob/main/src/TablutGoClient.py

# Connection Handler------------------------------------------------------------------


class ConnectionException(Exception):
    pass


class ConnectionClosedException(ConnectionException):
    pass


class ConnectionHandler:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((host, port))
        except OSError as e:
            raise ConnectionException(e)

    def close(self):
        self.socket.close()

    def send(self, message):
        msg = message + '\r\n'
        length = len(msg)
        try:
            self.socket.sendall(length.to_bytes(
                4, 'big') + bytes(msg, 'utf-8'))
        except OSError as e:
            print(e)
            exit()

    def recv(self):
        length = b''
        while len(length) < 4:
            try:
                data = self.socket.recv(4 - len(length))
            except OSError as e:
                raise ConnectionException(e)
            if data:
                length += data
            else:
                raise ConnectionClosedException("Connection aborted!")
        length = int.from_bytes(length, 'big')
        message = b''
        while len(message) < length:
            try:
                data = self.socket.recv(length - len(message))
            except OSError as e:
                raise ConnectionException(e)
            if data:
                message += data
            else:
                raise ConnectionClosedException("Connection aborted!")
        message = message.decode('utf-8')
        return length, message

# ---------------------------------------------------------------------------------------


class CommandLineException(Exception):
    pass


def send_move(connHandle, action, player):
    y0, x0, y1, x1 = AshtonTablut.num_to_coords(action)

    col = "abcdefghi"
    row = "123456789"

    move = {
        "from": col[x0]+row[y0],
        "to": col[x1]+row[y1],
        "turn": "WHITE" if player == 'W' else 'BLACK'
    }

    jsonData = json.dumps(move)
    connHandle.send(jsonData)
    logging.debug("Sent Data JSON: {0}".format(jsonData))

def JSON_to_local_state(data, turn, heuristic):
    logging.debug("Received Data JSON: {0}".format(data))

    raw_board = data['board']
    player = data['turn']

    board = AshtonTablut.get_initial_board()
    board[0, :, :, 0] = np.zeros((9, 9), dtype=np.int8)
    board[0, :, :, 1] = np.zeros((9, 9), dtype=np.int8)
    board[0, :, :, 2] = np.zeros((9, 9), dtype=np.int8)

    for i in range(9):
        for j in range(9):
            if raw_board[i][j][0] == 'W':
                board[0, i, j, 0] = 1
            elif raw_board[i][j][0] == 'B':
                board[0, i, j, 1] = 1
            elif raw_board[i][j][0] == 'K':
                board[0, i, j, 2] = 1
    
    return AshtonTablut.parse_board(board, player, turn, heuristic), player

def game_loop(args):
    # Args
    player = args.player.upper()[0]
    playing_player = 'W'
    timeout = args.timeout
    host = args.host
    cores = args.cores

    port = 5800 if player == 'W' else 5801

    # Network loading
    config = TablutConfig()
    #Al sesto turno l'euristica hard coded diventa dominante
    heuristic = NeuralHeuristicFunction(config)

    if heuristic.init_tflite():
        logging.info("Netowrk loaded successfully")
    else:
        logging.info("Netowrk loading error")
        heuristic = OldSchoolHeuristicFunction()

    turn = 0

    # Start connection
    connHandle = ConnectionHandler(host, port)
    connHandle.send('AlphaTablut')

    # Game loop
    while True:
        try:
            length, message = connHandle.recv()

            logging.debug("Received message of length {}".format(length))

            if message:
                # Sync local state with server state
                data = json.loads(message)
                state, playing_player = JSON_to_local_state(
                    data, turn, heuristic)

                logging.info("Turn {0}: {1} is playing.".format(
                    turn, playing_player))

                logging.info("\n"+state.display())

                if playing_player == 'WHITEWIN':
                    logging.info("We {} GG WP!".format(
                        'WON' if playing_player[0] == player else 'LOST'))
                    break
                elif playing_player == 'BLACKWIN':
                    logging.info("We {} GG WP!".format(
                        'WON' if playing_player[0] == player else 'LOST'))
                    break
                elif playing_player == 'DRAW':
                    logging.info("We {} GG WP!".format('DREW'))
                    break
                elif playing_player[0] == player:
                    logging.info("Computing and sending action.")

                    search = Search()
                    best_next_state, best_action, best_score, max_depth, nodes_explored, search_time = search.iterative_deepening_search(
                        state=state, initial_cutoff_depth=2, cutoff_time=timeout)

                    send_move(connHandle, best_action, player)
                    logging.debug("Action sent!")

                    best_action = AshtonTablut.num_to_coords(best_action)
                    logging.info("Game move ({0}): {1} -> {2}, Search time: {3}, Max Depth: {4}, Nodes explored: {5}, Score: {6}".format(
                        player,
                        (best_action[0], best_action[1]),
                        (best_action[2], best_action[3]),
                        search_time,
                        max_depth,
                        nodes_explored,
                        best_score))
                else:
                    logging.info("Waiting...")

                turn += 1
                #heuristic.set_alpha(2 / turn)

        except ConnectionException as e:
            logging.debug(e)
            logging.info("Coonection lost: {}".format(playing_player))
            break

    connHandle.close()


def main():
    # Il client è stato lanciato.
    # Faccio un parsing degli argomenti
    argparser = argparse.ArgumentParser(
        description='AlphaTablut Client')

    argparser.add_argument(
        '-p', '--player',
        metavar='P',
        default='White',
        type=str,
        help='Player Black/White (default: White)')

    argparser.add_argument(
        '-t', '--timeout',
        metavar='T',
        default=59,
        type=int,
        help='Timeout (default: 59)')

    argparser.add_argument(
        '-i', '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the server (default: 127.0.0.1)')

    argparser.add_argument(
        '-c', '--cores',
        metavar='C',
        default=4,
        type=int,
        help='Cores to use during Search (Default: 4)')

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
