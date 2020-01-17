import random
import numpy as np
import pandas as pd
from operator import add

# Q-Learning implementation
class QLAgent(object):
    def __init__(self):
        self.gamma = 0.9  # discounting factor [0,1]
        self.alpha = 0.1  # Learning rate [0,1]
        self.qTable = np.zeros((2**12, 3), float)
        self.available_actions = [[1, 0, 0],  # Straight ahead
                                  [0, 1, 0],  # Right
                                  [0, 0, 1]]  # Left
        self.epsilon = 0 # epsilon-greedy

    def getQT(self, state):
        state = state_to_index(state)
        return self.qTable[state]

    def bestAction(self, state):
        q = self.getQT(state)

        action_chosen = self.available_actions[0]
        max_val = q[action_to_index(action_chosen)]
        zero_actions = []

        for action in self.available_actions:
            ind = action_to_index(action)
            act = q[ind]
            if act == 0:
                zero_actions.append(action)
            elif act > max_val:
                max_val = q[ind]
                action_chosen = action

        if max_val == 0:
            choose = random.randint(0, len(zero_actions) - 1)
            action_chosen = zero_actions[choose]

        return action_chosen

    def updateQT(self, state, next_state, reward, action):
        q0 = self.getQT(state)
        q1 = self.getQT(next_state)

        actindex = action_to_index(action)
        statindex = state_to_index(state)
        new_val = reward + (self.gamma * np.amax(q1)) - q0[actindex]
        self.qTable[statindex][actindex] = q0[actindex] + (self.alpha * new_val)

    # This function is taken from maurock's DQN impl
    def get_state(self, game, player, food):
        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
             add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
            player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))), # danger left


            player.x_change == -20, # move left
            player.x_change == 20,  # move right
            player.y_change == -20, # move up
            player.y_change == 20,  # move down
            food.x_food < player.x, # food left
            food.x_food > player.x, # food right
            food.y_food < player.y, # food up
            food.y_food > player.y  # food down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def train_short_memory(self, state, action, reward, next_state, done):
        self.updateQT(state, next_state, reward, action)
    
# Utility functions

def action_to_index(action):
    try:
        # action's type may be ndarray
        temp = action.tolist()
        action = temp
    except:
        pass
    if action == [1,0,0]:
        return 0 # Straight
    elif action == [0,1,0]:
        return 1 # Right
    elif action == [0,0,1]:
        return 2 # Left
    else:
        raise "Unvalid action"

def state_to_index(state):
    sum = 0
    count = 0
    for s in state:
        sum += ((2**count) * s)
        count += 1
    return sum
