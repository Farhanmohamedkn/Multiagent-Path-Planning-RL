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
# batch_size=32

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
        
        # LSTM layers
        self.lstmcell=nn.LSTMCell(RNN_SIZE,RNN_SIZE)
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
        # random_tensor = torch.rand(4, 4, 10, 10)
        # # Define the list
        # goal_pos = [1, 2, 3]

        # # Convert the list to a tensor
        # goal_tensor = torch.tensor(goal_pos, dtype=torch.float32)

        # # Repeat the tensor to get the desired shape [4, 3]
        # result_tensor = goal_tensor.unsqueeze(0).repeat(4, 1)
        # inputs=random_tensor
        # goal_pos=result_tensor

        
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        # x= x.squeeze() 
        goal_layer = F.relu(self.fc_goal(goal_pos))
        
        #convert it to a shape of 12x1
        # goal_layer=goal_layer.view(12,1)
        # if goal_layer.ndim == 1 and x.ndim == 1: #for tensor of dimension1
            # concatenated_tensor = torch.cat([goal_layer, x])
        # else:

            # concatenated_tensor = torch.cat([goal_layer, x],dim=1) #for batch

        # hidden_input =concatenated_tensor

        hidden_input = torch.cat((x, goal_layer),dim=1)
        # hid_trans = hidden_input.permute(1,0)
        h1 = self.fc1(hidden_input)
        if training:
            d1 = self.dropout1(h1)
        h2 = self.fc2(d1)
        if training:
            d2 = self.dropout2(h2)
        h3 = F.relu(d2 + hidden_input)

        rnn_in=h3



        
        

        c_init = torch.zeros(rnn_in.shape[0],RNN_SIZE) # should remove from here
        h_init = torch.zeros(rnn_in.shape[0],RNN_SIZE)

        state_init = (c_init, h_init)

        hx,cx=self.lstmcell(rnn_in,state_init)

        state_in=(hx,cx)
        

        
       
        lstm_outputs, state_out = self.lstm(rnn_in, state_in) #its dynamic rnn may have to use loops for sequence length
            
        
        policy = F.softmax(self.policy_layer(lstm_outputs)) #there is softmax2d and softmaxlog
        policy_sig = torch.sigmoid(self.policy_layer(lstm_outputs))
        value = self.value_layer(lstm_outputs)
        blocking = torch.sigmoid(self.blocking_layer(lstm_outputs))
        on_goal = torch.sigmoid(self.on_goal_layer(lstm_outputs))
        
        return policy, value, state_out ,state_in, state_init, blocking, on_goal,policy_sig

# # Usage example
# net = ACNet('scope', a_size=5, trainer=optim.Adam, TRAINING=True, GRID_SIZE=10, GLOBAL_NET_SCOPE='global')
# inputs = torch.randn(1, 4, 10, 10)
# goal_pos = torch.randn(1, 3)
# hxs = torch.zeros(1, RNN_SIZE)
# cxs = torch.zeros(1, RNN_SIZE)
# policy, value, (hx, cx), blocking, on_goal = net(inputs, goal_pos, hxs, cxs)
# print(policy, value, (hx, cx), blocking, on_goal)
