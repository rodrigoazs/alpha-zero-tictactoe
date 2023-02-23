import os
from random import shuffle

import numpy as np
import torch
import torch.optim as optim

from mcts import MCTS


class Trainer:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def execute_episode(self):

        game = self.game()
        train_examples = []
        current_player = 1
        state = game.get_board()

        while True:
            canonical_board = game.get_canonical_board()

            self.mcts = MCTS(
                game, self.model, num_simulations=self.args["num_simulations"]
            )
            root = self.mcts.run(self.model, canonical_board, player=1)

            action_probs = [0 for _ in range(9)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append(
                (canonical_board.flatten(), current_player, action_probs)
            )

            action = root.select_action(temperature=self.args["temperature"])
            game.set_board(state)
            game.set_next_player(current_player)
            game.play_next_state(action)
            state = game.get_board()
            current_player = game.get_next_player()
            reward = game.get_reward_for_player()

            if reward is not None:
                ret = []
                for (
                    hist_state,
                    hist_current_player,
                    hist_action_probs,
                ) in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append(
                        (
                            hist_state,
                            hist_action_probs,
                            reward * ((-1) ** (hist_current_player != current_player)),
                        )
                    )
                return ret

    def learn(self):
        for i in range(1, self.args["numIters"] + 1):

            print("{}/{}".format(i, self.args["numIters"]))

            train_examples = []

            for eps in range(self.args["numEps"]):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args["checkpoint_path"]
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args["epochs"]):
            self.model.train()
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args["batch_size"]):
                sample_ids = np.random.randint(
                    len(examples), size=self.args["batch_size"]
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous()  # .cuda()
                target_pis = target_pis.contiguous()  # .cuda()
                target_vs = target_vs.contiguous()  # .cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
            },
            filepath,
        )
