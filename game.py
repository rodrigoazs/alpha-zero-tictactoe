import numpy as np


class TicTacToe:
    def __init__(self, player=1, board=None):
        self._board = np.zeros((3, 3), dtype=int) if board is None else board
        self._next_player = player

    def set_board(self, board):
        self._board = board

    def get_board(self):
        return self._board

    def set_next_player(self, player):
        self._next_player = player

    def get_next_player(self):
        return self._next_player

    def is_winner(self, player):
        # if won in cross
        if (
            self._board[0][0] == player
            and self._board[1][1] == player
            and self._board[2][2] == player
        ):
            return True
        if (
            self._board[0][2] == player
            and self._board[1][1] == player
            and self._board[2][0] == player
        ):
            return True
        # if won in columns and rows
        for axis in range(2):
            for value in np.sum(self._board, axis=axis):
                if value == 3 * player:
                    return True
        return False

    def play_next_state(self, action):
        b = np.copy(self._board)
        b.flat[action] = self._next_player
        self.set_board(b)
        self.set_next_player(-self._next_player)

    def has_legal_moves(self):
        # look for zeros
        for index in range(9):
            if self._board.flat[index] == 0:
                return True
        return False

    def get_valid_moves(self):
        # All moves are invalid by default
        valid_moves = np.zeros((3, 3), dtype=int)
        for index in range(9):
            if self._board.flat[index] == 0:
                valid_moves.flat[index] = 1
        return valid_moves.flatten()

    def get_masked_action_probs(self, action_probs):
        valid_moves = self.get_valid_moves()
        masked_action_probs = action_probs * valid_moves  # mask invalid moves
        masked_action_probs /= np.sum(masked_action_probs)
        return masked_action_probs

    def get_reward_for_player(self):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost
        # 0 if draw
        if self.is_winner(1):
            return 1
        if self.is_winner(-1):
            return -1
        if self.has_legal_moves():
            return None
        return 0

    def get_canonical_board(self):
        # get the board from the perspective of the other player
        return -1 * self._board
