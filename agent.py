from wimblepong import Wimblepong
import random
import torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 100
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space)

        # TODO: Add another linear layer for the critic
        # self.fcVal = torch.nn.Linear(self.hidden, 1)

        # self.sigma = torch.zeros(1)  # TODO: Implement learned variance (or copy from Ex5)
        # self.sigma = torch.nn.Parameter(torch.tensor(10.0), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def preprocess(state):
        state = np.sum(state, axis=2)
        state[state == 70] = 0
        state[state != 0] = 1
        state = state[::5,::5]
        state = state[:,3:38].astype(np.float)
        return torch.tensor(state, dtype=torch.float).view(-1)

    def forward(self, x):
        # Common part
        x = preprocess(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        action_probs = F.softmax(x)
        return action_probs

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def get_action(self, observation, evaluation=False):
        action_probs = self.policy(observation).to(train_device)

        p = np.random.uniform()

        if p<action_probs[0]:
            return 0
        elif p<action_probs[0]+action_probs[1]:
            return 1
        else:
            return 2
    
    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

        # TODO: Compute state values
        ___, vals = self.policy(states)
        vals = vals.squeeze(-1)

        ### WHY DOES EVERYTHING FAIL WITH THIS BLOCK?
        with torch.no_grad():
            ___, next_vals = self.policy.forward(next_states)
            next_vals = next_vals.squeeze(-1)

        # vals = vals * (1-done_shift)
        next_vals = next_vals * (1-done)

        # TODO: Compute critic loss (MSE)
        critic_loss = F.mse_loss(vals, rewards.detach() + self.gamma*next_vals)

        # Advantage estimates
        # TODO: Compute advantage estimates
        adv = rewards + self.gamma*next_vals - vals

        # TODO: Calculate actor loss (very similar to PG)
        actor_loss = torch.mean(-adv.detach() * action_probs)

        # TODO: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        loss = critic_loss + actor_loss
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.step()
        self.optimizer.zero_grad()










class badAI(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 20                
        self.name = "badAI"

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        # Get the player id from the environmen
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        # Get own position in the game arena
        my_y = player.y
        # Get the ball position in the game arena
        ball_y = self.env.ball.y + (random.random()*self.bpe-self.bpe/2)

        # Compute the difference in position and try to minimize it
        y_diff = my_y - ball_y
        if abs(y_diff) < 2:
            action = 0  # Stay
        else:
            if y_diff > 0:
                action = self.env.MOVE_UP  # Up
            else:
                action = self.env.MOVE_DOWN  # Down

        return action

    def reset(self):
        # Nothing to done for now...
        return


