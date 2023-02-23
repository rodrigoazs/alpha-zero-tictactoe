import torch

from game import TicTacToe
from model import Model
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    "batch_size": 64,
    "numIters": 500,  # Total number of training iterations
    "num_simulations": 100,  # Total number of MCTS simulations to run when deciding on a move to play
    "numEps": 100,  # Number of full games (episodes) to run during each iteration
    "epochs": 2,  # Number of epochs of training per iteration
    "temperature": 0,  # float("inf") for random
    "checkpoint_path": "latest.pth",  # location to save latest set of weights
}

game = TicTacToe
model = Model(device)

trainer = Trainer(game, model, args)
trainer.learn()
