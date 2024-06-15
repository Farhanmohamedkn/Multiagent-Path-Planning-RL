import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque

class CustomActorCritic(nn.Module):
    def __init__(self, input_shape, action_size, rnn_size, goal_repr_size):
        super(CustomActorCritic, self).__init__()
        self.rnn_size = rnn_size
        self.goal_repr_size = goal_repr_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(rnn_size // 4, rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(rnn_size // 4, rnn_size // 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(rnn_size // 4, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(rnn_size // 2, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(rnn_size // 2, rnn_size // 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(rnn_size // 2, rnn_size - goal_repr_size, kernel_size=2, stride=1)

        # Fully connected layers
        self.fc_goal = nn.Linear(3, goal_repr_size)
        self.fc_hidden = nn.Linear(rnn_size, rnn_size)
        
        # LSTM layer
        self.lstm = nn.LSTMCell(rnn_size, rnn_size)

        # Output layers
        self.actor_fc = nn.Linear(rnn_size, action_size)
        self.critic_fc = nn.Linear(rnn_size, 1)

    def forward(self, x, goal, hx, cx):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Goal representation
        goal_repr = F.relu(self.fc_goal(goal))

        # Combine features and goal representation
        combined = torch.cat((x, goal_repr), dim=1)

        # Fully connected layer
        combined = F.relu(self.fc_hidden(combined))

        # LSTM layer
        hx, cx = self.lstm(combined, (hx, cx))

        # Actor and Critic output
        action_probs = F.softmax(self.actor_fc(hx), dim=-1)
        state_value = self.critic_fc(hx)

        return action_probs, state_value, (hx, cx)

class PPOAgent:
    def __init__(self, state_shape, action_size, num_agents, rnn_size, goal_repr_size, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_freq=5, batch_size=32):
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_freq = update_freq
        self.batch_size = batch_size

        self.policy_net = CustomActorCritic(state_shape, action_size, rnn_size, goal_repr_size)
        self.old_policy_net = CustomActorCritic(state_shape, action_size, rnn_size, goal_repr_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = []

    def remember(self, state, goal, action, reward, next_state, done, hx, cx, next_hx, next_cx):
        self.memory.append((state, goal, action, reward, next_state, done, hx, cx, next_hx, next_cx))

    def act(self, state, goal, hx, cx):
        with torch.no_grad():
            action_probs, state_value, (hx, cx) = self.policy_net(state, goal, hx, cx)
            action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[:, action].item(), state_value.item(), (hx, cx)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.clip_epsilon * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state, goal, action, reward, next_state, done, hx, cx, next_hx, next_cx = zip(*self.memory)

        state = torch.cat(state)
        goal = torch.cat(goal)
        action = torch.tensor(action).view(-1, 1)
        reward = torch.tensor(reward).view(-1, 1)
        done = torch.tensor(done).view(-1, 1)
        next_state = torch.cat(next_state)

        _, old_probs, old_values, _ = self.old_policy_net(state, goal, hx, cx)
        old_probs = old_probs.gather(1, action)

        _, new_probs, new_values, _ = self.policy_net(state, goal, hx, cx)
        new_probs = new_probs.gather(1, action)

        advantages = self.compute_advantages(reward, old_values, done)

        ratio = (new_probs / old_probs).clamp(0.8, 1.2)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(new_values, reward)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.memory = []

# Initialize Gym environment
env = gym.make('YourCustomMultiAgentEnv-v0')  # Replace with your custom multi-agent environment
num_agents = env.num_agents  # Assume your environment has this attribute
state_shape = env.observation_space.shape
action_size = env.action_space.n
rnn_size = 128  # Example RNN size
goal_repr_size = 16  # Example goal representation size

agents = [PPOAgent(state_shape, action_size, num_agents, rnn_size, goal_repr_size) for _ in range(num_agents)]
done = False
EPISODES = 1000

for e in range(EPISODES):
    states = env.reset()
    hxs = [torch.zeros(1, rnn_size) for _ in range(num_agents)]
    cxs = [torch.zeros(1, rnn_size) for _ in range(num_agents)]
    for time in range(500):
        actions = []
        for i, agent in enumerate(agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            goal = torch.FloatTensor(env.get_goal(i)).unsqueeze(0)  # Assuming env.get_goal(i) gives the goal for agent i
            action, prob, value, (hx, cx) = agent.act(state, goal, hxs[i], cxs[i])
            actions.append(action)
            hxs[i], cxs[i] = hx, cx

        next_states, rewards, dones, _ = env.step(actions)
        for i, agent in enumerate(agents):
            agent.remember(states[i], env.get_goal(i), actions[i], rewards[i], next_states[i], dones[i], hxs[i], cxs[i], hxs[i], cxs[i])

        states = next_states
