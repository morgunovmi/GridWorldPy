import numpy as np

BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (0, 4)
LOSE_STATE = (1, 4)
START = (2, 0)
OBSTACLES = [(1, 1), (2, 2), (1, 2)]
DETERMINISTIC = True

class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.obstacles = OBSTACLES
        for obs in self.obstacles:
            self.board[obs[0], obs[1]] = -1
        self.action_dict = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        self.state = state
        self.is_end = False
        self.determine = DETERMINISTIC

    def give_reward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def update_end(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.is_end = True
    
    def is_within_bounds(self, state):
        return state[0] >= 0 and state[0] < self.board.shape[0] and \
               state[1] >= 0 and state[1] < self.board.shape[1]

    def next_position(self, action):
        if self.determine:
            next_state = tuple(self.state[i] + self.action_dict[action][i] for i in range(2))
            if self.is_within_bounds(next_state) and not next_state in self.obstacles:
                return next_state
            return self.state

    def show_board(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


# Agent of player

class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0
        self.State.show_board()

    def choose_action(self):
        # choose action with most expected value
        max_next_reward = 0
        action = "up"

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                next_reward = self.state_values[self.State.next_position(a)]
                if next_reward >= max_next_reward:
                    action = a
                    max_next_reward = next_reward
        return action

    def take_action(self, action):
        position = self.State.next_position(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.is_end:
                # back propagate
                reward = self.State.give_reward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                # print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.choose_action()
                # append trace
                self.states.append(self.State.next_position(action))
                # print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.take_action(action)
                # mark is end
                self.State.update_end()
                # print("nxt state", self.State.state)
                # print("---------------------")

    def show_values(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

if __name__ == "__main__":
    ag = Agent()
    ag.play(50)
    ag.show_values()