import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from envs.pusher import PusherEnv, MultiPointmassEnv

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# Hyper-parameters 
input_size = 8
hidden_size = 500
num_classes = 2
num_epochs = 50000
batch_size = 49*2
learning_rate = 0.001

# Fully connected neural network with one hidden layer
class PairwiseInteract(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PairwiseInteract, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        # forward pass through the network
        # input = [[pusher][object]action] flattened
        # output = 2d location [pusher]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def one_step(self, state_batch, action_batch):
        onetwo = np.reshape(state_batch,(-1, 6)) #(49,6)
        onetwo = np.append(onetwo, action_batch, axis=1)

        twoone = np.reshape(np.flip(state_batch, axis=1), (-1, 6)) #(49,6)
        twoone = np.append(twoone, action_batch, axis=1)

        inputs_np = np.concatenate((onetwo, twoone))
        inputs = torch.from_numpy(inputs_np).float()
        return model(inputs)

    def one_step_formatted(self, state_batch, action_batch):
        output = self.one_step(state_batch, action_batch).detach().numpy()
        shp =  np.shape(state_batch)[0]
        labels = np.concatenate((np.ones(shp), np.zeros(shp)))
        labels = np.reshape(labels,(-1,1))
        output = np.hstack((output, labels))
        one, two = np.reshape(output[:shp], (shp, 1, 3)), np.reshape(output[shp:], (shp, 1, 3))
        return np.concatenate((one, two), axis = 1)


    def act(self, state):
        states, actions = self._sim_actions_forward(state)
        costs = self._eval_traj_costs(states, actions)
        best_simulated_path = np.argmin(costs)
        return actions[best_simulated_path, 0]

    def _sim_actions_forward(self, state):
        states = [np.tile(state, [2048, 1]).reshape((2048,2,3))]
        actions = np.random.normal(size=(2048, 20 ,2), loc=[.5,0])

        curr_states = states[0]
        for t in range(20):
            curr_states = self.one_step_formatted(curr_states, actions[:,t])
            if t < 20 - 1:
                states.append(curr_states)
        return np.array(states).transpose([1, 0, 2, 3]), actions

    def _eval_traj_costs(self, states, actions):
        costs = np.zeros(states.shape[0])
        for i in range(states.shape[0]):
            cost, traj_blocks, traj_pusher = 0.0, states[i][:,1], states[i][:,0]
            for t in range(20):
                pusher = traj_pusher[t][:2]
                block = traj_blocks[t][:2]
                cost += np.sum(np.abs(np.array([.6,0])-block)) + .3*np.sum(np.abs(np.array(block-pusher)))
            costs[i] = cost
        return costs


    def train(self):
        states, actions, next_states = self.data_buffer
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

        # Train the model
        for i in range(num_epochs):
            idx = np.random.choice(len(states))
            state_batch = states[idx]#(49,2,3)
            action_batch = actions[idx]#(49,2)
            next_batch = next_states[idx]#(49,2,3)

            nextone = next_batch[:,0,:-1]
            nexttwo = next_batch[:,1,:-1]
            labels_np = np.concatenate((nextone, nexttwo))

            labels = torch.from_numpy(labels_np).float()
            outputs = self.one_step(state_batch, action_batch)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i==0:
              print("starting loss", loss.item())
            if i == num_epochs-1:
              print("ending loss", loss.item())

class CostNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CostNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        # forward pass through the network
        # input = [[pusher][object]action] flattened
        # output = 2d location [pusher]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def train(self):
        states, actions, next_states = self.data_buffer
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

        # Train the model
        for i in range(num_epochs):
            idx = np.random.choice(len(states))
            state_batch = states[idx]#(49,2,3)
            action_batch = actions[idx]#(49,2)
            next_batch = next_states[idx]#(49,2,3)

            nextone = next_batch[:,0,:-1]
            nexttwo = next_batch[:,1,:-1]
            labels_np = np.concatenate((nextone, nexttwo))

            labels = torch.from_numpy(labels_np).float()
            outputs = self.one_step(state_batch, action_batch)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i==0:
              print("starting loss", loss.item())
            if i == num_epochs-1:
              print("ending loss", loss.item())




def collect_data():
    #MPC testing
    env = PusherEnv()
    episodes = 5
    states = np.zeros((episodes, 50,2,3))
    actions = np.zeros((episodes, 50,2))


    for ep in range(episodes):
        env.reset()
        print("reset")
        # goal = .6, 0
        state = env.get_state()
        for t in range(50):
            action = model.act(state)
            env.step(action)
            state = env.get_state()
            states[ep,t] = state
            actions[ep,t] = action

    actions = actions[:,:-1]
    next_states = states[:,1:]
    states = states[:,:-1]
    model.data_buffer[0] = np.concatenate((model.data_buffer[0],states))[-50:]
    model.data_buffer[1] = np.concatenate((model.data_buffer[1],actions))[-50:]
    model.data_buffer[2] = np.concatenate((model.data_buffer[2],next_states))[-50:]

            

if __name__ == "__main__":
    states = np.load("pusher_states.npy")
    actions = np.load("pusher_actions.npy")[:,:-1]
    
    next_states = states[:,1:]
    states = states[:,:-1]
    
    model = PairwiseInteract(input_size, hidden_size, num_classes).to(device)
    model.data_buffer = [states, actions, next_states]
    for mpc_iter in range(100):
        if mpc_iter==1:
          num_epochs=10000
        model.train()
        collect_data()
        
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    
# env.get_state() to retrn a game state -> thresholding the representation to get COMs
# pretrain the dynamics model using the AE
# 