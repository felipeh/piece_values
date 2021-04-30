import chess
import numpy as np
import numpy.linalg
import chess.pgn

def get_piece_imbalance(board):
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
              chess.ROOK, chess.QUEEN]
    return {piece: len(board.pieces(piece,chess.WHITE))-\
                    len(board.pieces(piece,chess.BLACK)) \
            for piece in pieces}

def get_unbalanced_nodes(pgn):
    data = []
    knt = 0
    while True:
        knt += 1
        game = chess.pgn.read_game(pgn)

        if game is None:
            break

        if knt % 300 == 0:
            print(knt)

        if knt > 20000:
            break

        if len(game.variations) == 0:
            continue

        if game.variations[0].eval() is None:
            continue

        node = game.variations[0]
        old_imbalance_set = set([0])
        while len(node.variations) > 0:
            newnode = node.variations[0]
            piece_imbalance = get_piece_imbalance(node.board())
            imbalance_set = set(piece_imbalance.values())
            if imbalance_set != set([0]) and \
                    imbalance_set == old_imbalance_set:
                if node.eval() is not None:
                    data.append((node.eval().white().score(), piece_imbalance))
            old_imbalance_set = imbalance_set
            node = newnode
    return data

def perform_least_squares(data):
    yvals = []
    xvals = []
    for pt in data:
        if pt[0] is None:
            continue
        xtuple = [pt[1][chess.PAWN],
                    pt[1][chess.KNIGHT],
                    pt[1][chess.BISHOP],
                    pt[1][chess.ROOK],
                    pt[1][chess.QUEEN]]
        xvals.append(xtuple)
        yvals.append(pt[0])
    X = np.array(xvals, dtype=np.float64)
    Y = np.array(yvals, dtype=np.float64)

    vals_fit = np.linalg.lstsq(X,Y,rcond=None)
    vals = vals_fit[0]
    values = {'pawn':vals[0],
              'knight':vals[1],
              'bishop':vals[2],
              'rook':vals[3],
              'queen':vals[4]}
    return values

if __name__=="__main__":
    dbfile = open('small_db.pgn','r')
    data = get_unbalanced_nodes(dbfile)
    values = perform_least_squares(data)
    print(values)
