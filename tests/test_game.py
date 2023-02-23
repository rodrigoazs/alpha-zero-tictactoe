import numpy as np

from game import TicTacToe


def test_is_winner():
    game = TicTacToe()
    game._board[0] = [1, 1, 1]
    assert game.is_winner(1)
    game = TicTacToe()
    game._board[0][2], game._board[1][2], game._board[2][2] = -1, -1, -1
    assert game.is_winner(-1)
    game = TicTacToe()
    game._board[0][0], game._board[1][1], game._board[2][2] = -1, -1, -1
    assert game.is_winner(-1)


def test_play_next_state():
    game = TicTacToe()
    game.play_next_state(7)
    assert game._next_player == -1
    np.testing.assert_array_equal(game.get_board(), [[0, 0, 0], [0, 0, 0], [0, 1, 0]])


def test_has_legal_moves():
    board = np.array([[-1, -1, 1], [-1, 1, 1], [1, 1, -1]])
    game = TicTacToe(board=board)
    legal_moves = game.has_legal_moves()
    assert not legal_moves
    board = np.array([[-1, -1, 1], [0, 1, 1], [1, 1, -1]])
    game = TicTacToe(board=board)
    legal_moves = game.has_legal_moves()
    assert legal_moves


def test_get_valid_moves():
    board = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0]])
    game = TicTacToe(board=board)
    valid_moves = game.get_valid_moves()
    np.testing.assert_array_equal(valid_moves, [1, 0, 1, 0, 1, 1, 1, 0, 1])


def test_get_reward_for_player():
    board = np.array([[-1, 1, -1], [-1, 1, 1], [1, -1, -1]])
    game = TicTacToe(board=board)
    assert game.get_reward_for_player() == 0
