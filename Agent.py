import copy
import numpy as np
import torch
from torch import optim, nn
from collections import namedtuple, deque
import random

MEMORY_SIZE = 20000
GAMMA = 0.995
TAU = 0.001
LR = 0.00005

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_next', 'done'))


class QAgent():
    # the agent acts based on e-greedy policy, caches experience into memory
    # learns based on DQN using experience from memory
    def __init__(self, input_dim, action_dim, batch_size):
        # define parameters
        #self.cuda = torch.cuda.is_available()
        #self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.cuda = False
        self.device = torch.device('cpu')
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

        # set up memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # define policy net and optimizer
        self.policy_net = QNet(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # define target net and turn off learning/gradient boost
        self.target_net = QNet(input_dim, action_dim).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False

    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if random.random() < eps:
            action = np.random.randint(self.action_dim)
        else:
            self.policy_net.eval()
            if self.cuda:
                q_value = self.policy_net(state).cpu().data
            else:
                q_value = self.policy_net(state).data
            self.policy_net.train()

            action = torch.argmax(q_value).item()

        return action

    def cache(self, state, action, reward, state_next, done):
        # add to memory to the right
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long).unsqueeze(0)
        state_next = torch.tensor(state_next, device=self.device, dtype=torch.float)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float).unsqueeze(0)
        done = torch.tensor(done, device=self.device, dtype=torch.float).unsqueeze(0)
        self.memory.append(Transition(state, action, reward, state_next, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = map(torch.stack, zip(*batch))
        # state,action,state_next,reward,done = zip(*batch)
        # state = torch.stack(state,dim=0)
        return (states, actions, rewards, states_next, dones)

    def learn(self, experiences):
        # get TD difference Q_policy(s,a) - (r + gamma * max[a'](Q_target(s',a'))
        self.policy_net.train()
        states, actions, rewards,states_next, dones = experiences
        q_policy = self.policy_net(states).gather(1, actions)
        q_target = self.target_net(states_next).data.max(dim=1)[0].unsqueeze(1)
        q_target = q_target * (1 - dones) * GAMMA + rewards

        # gradient boost policy net parameters based on the MSE of TD differences
        loss = nn.functional.mse_loss(q_policy, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target_net(self, soft_update=True):
        if soft_update:
            # update target net gradually periodically
            for target_p, policy_p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_p.data.copy_(TAU * policy_p.data + (1. - TAU) * target_p.data)
        else:
            # sync target net to policy net completely after a long time frame
            self.target_net.load(self.policy_net.state_dict())


class QNet(nn.Module):
    # multi-layer neural network with user defined dimensions for hidden layers
    def __init__(self, input_dim, output_dim, hidden_dims=[512,256,256]):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.33)

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size,-1)
        x = self.activation(self.dropout(self.input_layer(x)))
        for h in self.hidden_layers:
            x = self.activation(self.dropout(h(x)))
            
        x = self.output_layer(x)
        return x