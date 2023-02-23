import math

import numpy as np


def ucb_score(parent, child):
    """
    The score for an action that would transite between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposite player
        value_score = -child.value()
    else:
        value_score = 0
    return value_score + prior_score


class Node:
    def __init__(self, prior, player):
        self.visit_count = 0
        self.player = player
        self.prior = prior
        self.value_sum = 0
        self.children = {}  # enumerated by position in array flat
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def set_state(self, state):
        self.state = state

    def set_player(self, player):
        self.player = player

    def expand(self, state, player, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.set_state(state)
        self.set_player(player)
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, player=-1 * self.player)

    def __repr__(self):
        prior = "{0:.2f}".format(self.prior)
        return "Node(State: {} Prior: {} Count: {} Value: {})".format(
            str(self.state), prior, self.visit_count, self.value()
        )


class MCTS:
    def __init__(self, game, model, num_simulations=100):
        self.game = game
        self.model = model
        self._num_simulations = num_simulations

    def run(self, model, state, player):
        root = Node(0, player)

        # EXPAND root
        action_probs, value = model.predict(state)
        action_probs = self.game.get_masked_action_probs(action_probs)
        root.expand(state, player, action_probs)

        for _ in range(self._num_simulations):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            self.game.set_board(state)
            self.game.set_next_player(parent.player * -1)

            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            # play next state
            self.game.play_next_state(action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board()

            # The value of the new state
            value = self.game.get_reward_for_player()
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = model.predict(next_state)
                action_probs = self.game.get_masked_action_probs(action_probs)
                node.expand(next_state, parent.player * -1, action_probs)

            self.backpropagate(search_path, value, parent.player * -1)

        return root

    def backpropagate(self, search_path, value, player):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1
