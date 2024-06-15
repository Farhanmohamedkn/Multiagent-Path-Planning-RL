import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Parameters for training
GRAD_CLIP = 1000.0
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
RNN_SIZE = 512
GOAL_REPR_SIZE = 12
batch_size=32

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(tensor):
        out = np.random.randn(*tensor.shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return torch.tensor(out)
    return _initializer

class ACNet(nn.Module):
    def __init__(self, scope, a_size, trainer, TRAINING, GRID_SIZE, GLOBAL_NET_SCOPE):
        super(ACNet, self).__init__()
        
        self.scope = scope
        self.a_size=a_size
        
        # input should be integrated here.. Farhan
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(RNN_SIZE//2, RNN_SIZE-GOAL_REPR_SIZE, kernel_size=2, stride=1)

        # Fully connected layers
        self.fc_goal = nn.Linear(3, GOAL_REPR_SIZE)
    
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.dropout1 = nn.Dropout(p=1-KEEP_PROB1)
        self.dropout2 = nn.Dropout(p=1-KEEP_PROB2)
        
        # LSTM layer
        self.lstm = nn.LSTM(RNN_SIZE, RNN_SIZE)
        
        # Output layers
        self.policy_layer = nn.Linear(RNN_SIZE, a_size) #need to use normalized colmns_initializer
        self.value_layer = nn.Linear(RNN_SIZE, 1)
        self.blocking_layer = nn.Linear(RNN_SIZE, 1)
        self.on_goal_layer = nn.Linear(RNN_SIZE, 1)

        # Apply custom weight initialization
        self.apply(self.weights_init) #we need it
    
    
    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data = normalized_columns_initializer(1.0)(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
    
    
    def forward(self, inputs, goal_pos,state_init, training=True):

        # inputs=torch.tensor(inputs, dtype=torch.float32,requires_grad=True)
        # goal_pos=torch.tensor(goal_pos, dtype=torch.float32,requires_grad=True)
        random_tensor = torch.rand(4, 4, 10, 10)
        # Define the list
        goal_pos = [1, 2, 3]

        # Convert the list to a tensor
        goal_tensor = torch.tensor(goal_pos, dtype=torch.float32)

        # Repeat the tensor to get the desired shape [4, 3]
        result_tensor = goal_tensor.unsqueeze(0).repeat(4, 1)
        inputs=random_tensor
        goal_pos=result_tensor

        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        # x = x.view(x.size(0), -1)
        x= x.squeeze() 
        # print(x.shape)
        goal_layer = F.relu(self.fc_goal(goal_pos))
        # print("goal layer before",goal_layer.shape)
        #convert it to a shape of 12x1
        # goal_layer=goal_layer.view(12,1)
        if goal_layer.ndim == 1 and x.ndim == 1: #for tensor of dimension1
            concatenated_tensor = torch.cat([goal_layer, x])
        else:

            concatenated_tensor = torch.cat([goal_layer, x],dim=1) #for batch

        hidden_input =concatenated_tensor

        # hidden_input = torch.cat((x, goal_layer))
        # hid_trans = hidden_input.permute(1,0)
        h1 = self.fc1(hidden_input)
        if training:
            d1 = self.dropout1(h1)
        h2 = self.fc2(d1)
        if training:
            d2 = self.dropout2(h2)
        h3 = F.relu(d2 + hid_trans)

        c_in = torch.zeros(1, RNN_SIZE)
        h_in = torch.zeros(1, RNN_SIZE)

        state_init = (c_in, h_in)
        state_in=state_init

        rnn_in=h3

        lstm_out,(lstm_c, lstm_h) = self.lstm(rnn_in, state_in) #its dynamic rnn may have to use loops for sequence length

        state_out=(lstm_c, lstm_h) #tensorflow sending it in different shape
        
        policy = F.softmax(self.policy_layer(lstm_out),dim=-1) #there is softmax2d and softmaxlog
        policy_sig = torch.sigmoid(self.policy_layer(lstm_out))
        value = self.value_layer(lstm_out)
        blocking = torch.sigmoid(self.blocking_layer(lstm_out))
        on_goal = torch.sigmoid(self.on_goal_layer(lstm_out))
        
        return policy, value, state_in,state_out, blocking, on_goal,policy_sig

    def evaluate_actions(self, inputs, goal_pos, actions,rewards,state_init,training=True):
        # print("we are here")
        self.policy, self.value, self.state_in,self.state_out, self.blocking, self.on_goal,self.policy_sig= self.forward(inputs, goal_pos,state_init,training=True)
        if training:
        
            self.actions = torch.tensor(actions,dtype=torch.int32) #you don’t need to define placeholders explicitly; you can create tensors directly
            self.actions_onehot = torch.nn.functional.one_hot(self.actions.to(torch.int64), num_classes=self.a_size)
            # self.train_valid = torch.empty(size=(None, self.a_size), dtype=torch.float32)
            # self.target_v = torch.empty(size=(None,), dtype=torch.float32) #target values
            self.advantages = torch.tensor(rewards,dtype=torch.float32)
            # self.target_blockings = torch.empty(size=(None,), dtype=torch.float32)
            # self.target_on_goals = torch.empty(size=(None,), dtype=torch.float32)
            self.responsible_outputs = torch.sum(self.policy * self.actions_onehot, dim=1)
            # self.train_value = torch.empty(size=(None,), dtype=torch.float32)
            # self.optimal_actions = torch.empty(size=(None,), dtype=torch.int32)
            # self.optimal_actions_onehot = torch.nn.functional.one_hot(self.optimal_actions, num_classes=self.a_size)


            # Loss functions
            # self.value_loss = torch.sum(self.train_value * torch.square(self.target_v - self.value.view(-1)))
            # self.entropy = -torch.sum(self.policy * torch.log(torch.clamp(self.policy, min=1e-10, max=1.0)))
            self.policy_loss = -torch.sum(torch.log(torch.clamp(self.responsible_outputs, min=1e-15, max=1.0)) * self.advantages)
            # self.valid_loss = -torch.sum(torch.log(torch.clamp(self.valids, min=1e-10, max=1.0)) * self.train_valid + 
                        # torch.log(torch.clamp(1 - self.valids, min=1e-10, max=1.0)) * (1 - self.train_valid))
            # self.blocking_loss = -torch.sum(self.target_blockings * torch.log(torch.clamp(self.blocking, min=1e-10, max=1.0)) +
                        #    (1 - self.target_blockings) * torch.log(torch.clamp(1 - self.blocking, min=1e-10, max=1.0)))
            # self.on_goal_loss = -torch.sum(self.target_on_goals * torch.log(torch.clamp(self.on_goal, min=1e-10, max=1.0)) +
                        #   (1 - self.target_on_goals) * torch.log(torch.clamp(1 - self.on_goal, min=1e-10, max=1.0)))
            
            # Total loss
            # self.loss = 0.5 * self.value_loss + self.policy_loss + 0.5 * self.valid_loss - self.entropy * 0.01 + 0.5 * self.blocking_loss
            self.loss = self.policy_loss

            # Imitation loss (assuming optimal_actions_onehot is a tensor)
            # imitation_loss = torch.mean(F.cross_entropy(self.policy, self.optimal_actions))


        return  self.loss, #imitation_loss

# # Usage example
# net = ACNet('scope', a_size=5, trainer=optim.Adam, TRAINING=True, GRID_SIZE=10, GLOBAL_NET_SCOPE='global')
# inputs = torch.randn(1, 4, 10, 10)
# goal_pos = torch.randn(1, 3)
# hxs = torch.zeros(1, RNN_SIZE)
# cxs = torch.zeros(1, RNN_SIZE)
# policy, value, (hx, cx), blocking, on_goal = net(inputs, goal_pos, hxs, cxs)
# print(policy, value, (hx, cx), blocking, on_goal)
