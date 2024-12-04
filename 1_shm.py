
"""
Created on Mon Nov 25 12:37:31 2024

@author: gabriel

The objective is to monitor the stiffness of a 1dof system (mx^^ + cx^ + kx = f)
It could represent the motion of a cantilever beam that has some defects that have deteriorate tìits stiffness

The input force is a hammer hit
"""


import numpy as np 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

freq = 50 # Size of 1 layer

def h_to_label(h):
    if h>=0.8:
        label=4
    if h>=0.6 and h<0.8:
        label=3
    if h>=0.4 and h<0.6:
        label=2
    if h>=0.2 and h<0.4:
        label=1
    if h<0.2:
        label=0
        

    # if h>=0.8:
    #     label=7
    # if h>=0.6 and h<0.8:
    #     label=6
    # if h>=0.5 and h<0.6:
    #     label=5
    # if h>=0.4 and h<0.5:
    #     label=4
    # if h>=0.3 and h<0.4:
    #     label=3
    # if h>=0.2 and h<0.3:
    #     label=2
    # if h>=0.1 and h<0.2:
    #     label=1
    # if h<0.1:
    #     label=0
    
    # label = np.round(h*10)
        
    return label

def motion(h, training):
    k=100*h; m=5;  T_end = 20
    
    time = np.linspace(0,T_end,T_end*freq+1); Dt = time[1]
    # u = np.zeros(len(time))
    # u[:int(0.1*freq)]=10
    # u = 10
    u = np.random.uniform(-10,10,(len(time)))
    h_damp = 0.1
    c = h_damp*2*np.sqrt(k*m)
    
    A = np.zeros((2,2)); B = np.zeros((2,1))
    
    A[0,1]=Dt; A[1,1]=-c/m*Dt; A[1,0] = -k*Dt/m
    B[1,0]=Dt/m
    
    x = np.random.uniform(-10,10,(2,1))
    # x = np.zeros((2,1))
    y = []
    k=0
    
    noise = np.random.normal(0,0.0001,(len(time)))
    
    for j in time[:-1]:
        y.append(x[0,0]+noise[k])
        
        x += A@x + B*u[k]
        k+=1
        
        if k > freq*3 and abs(y[-4]-y[-1])<0.001 and k%freq==0:
            break

    if training:
        
        matrix = np.array(y).reshape(k//freq,freq)
        
        # label = np.ones([k//freq,1])*(h*10)
        label = np.ones([k//freq,1])*h_to_label(h)
    
        return np.concatenate((matrix, label),axis=1)
    
    else:
        
        return y, time[:len(y)]
    
def data(N, training):
    
    h_list = np.random.uniform(0,1,(N,1))
    first_time = True
    
    for h in h_list:
        matrix = motion(h, training)
        
        if first_time:
            X = matrix.copy()
            first_time = False
        else:
            X = np.concatenate((X,matrix),axis = 0)
    
    np.random.shuffle(X)
    
    return X

class MIONet(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim, number_of_hidden_layers): 
        
        
        super(MIONet, self).__init__()
        # Define sub-network for first input stream 
        
        activation_fn=nn.ReLU()
        
        layer_width = input_dim
        layers = []
        for n_layer in range(number_of_hidden_layers):
            layers.append(nn.Linear(layer_width,hidden_dim))
            layers.append(activation_fn)
            layer_width = hidden_dim
        
        layers.append(nn.Linear(layer_width,output_dim))
        layers.append(nn.Softmax())
        
        self.network = nn.Sequential(*layers)
        
        
        
    def forward(self, input): # Process each input stream 
        output = self.network(input)
        return output

class model_NN():
    def __init__(self,NN_model, lr):
        self.Network = NN_model
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.Network.parameters(), lr=lr)
    
    def train(self, epochs, X, Y):
       
        lossTracker = []
        self.Network.train()
    
        for idx in range(epochs):
                 
            y_pred = self.Network(X)
            y_pippo = y_pred.detach().numpy()
            loss = self.criterion(y_pred,Y[:,0])
            loss_np = loss.detach().numpy()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            lossTracker.append(loss.item())
            
            print('epoch {:05d}, loss = {:10.4e}'.format(idx,loss))
        
        fig,ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(range(epochs),lossTracker)
        ax.set_title('Training loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid()
        return 
    
    def test_accuracy(self, X_test, Y_test):
        # Make predictions on the test set
        with torch.no_grad():
            outputs = self.Network(X_test)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest logit

        # Compute accuracy
        accuracy = (predicted == Y_test[:,0]).sum().item() / Y_test.size(0)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        
        
        # Step 1: Calculate the confusion matrix
        true_labels = Y_test
        cm = confusion_matrix(true_labels, predicted, labels=list(range(0,N_classes)))
        
        # Step 2: Visualize the confusion matrix using a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2', '3', '4'], 
                    yticklabels=['0', '1', '2', '3', '4'], cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
            
    def predict(self,func):
        f = torch.tensor(func,dtype=torch.float32)
        y_pred = self.Network(f)
        label_predicted = torch.argmax(y_pred)  # Get the class with the highest logit
        
        return label_predicted.item()


N = 1001; N_classes = 5;
data_training = data(N,training=True)

device = torch.device('cpu')

X_training = torch.tensor(data_training[:,:freq], dtype=torch.float32, device=device)
Y_training = torch.tensor(data_training[:,-1:], dtype=torch.long, device=device)
    
new_model = 2
training = 1

if training==1:
    if new_model==1:
        net = MIONet(input_dim=freq, 
                 output_dim=N_classes, hidden_dim=50, 
                 number_of_hidden_layers=4)
        
        net.to(device) 

    else:
        ## Training
        net = torch.load('shm2_stiffness.pth',map_location=torch.device('cpu'))
        net.to(device)
    
    model = model_NN(net,lr=1e-5)
    training = model.train(1001, X_training, Y_training)
    torch.save(net,'shm2_stiffness.pth')
        
else:
    net = torch.load('shm2_stiffness.pth',map_location=torch.device('cpu'))
    net.to(device)
    model = model_NN(net,lr=1e-5)
    training = model.train(1001, X_training, Y_training)
    torch.save(net,'shm2_stiffness.pth')
    
## Accuracy
data_acc = data(200,training=True)

X_acc = torch.tensor(data_acc[:,:freq], dtype=torch.float32, device=device)
Y_acc = torch.tensor(data_acc[:,-1:], dtype=torch.long, device=device)

model.test_accuracy(X_acc, Y_acc)

## Testing
h_sim = np.random.uniform(0,1)
y_sim, time = motion(h_sim,training=False)
label_true = h_to_label(h_sim)

label_pred = []
condition = True
k = 0

## Online estimation of the stiffness is performed at each second
while condition:
    idx = freq*k
    pippo = model.predict(y_sim[idx:idx+freq])
    label_pred.append(pippo)
    k += 1
    
    if freq*k >= len(time):
        condition = False

from matplotlib.ticker import MaxNLocator
fig, ax = plt.subplots(1,1,layout='constrained',figsize=(6,4))
plt.bar(np.arange(k),label_pred) 
plt.plot(np.arange(k),np.ones(k)*label_true,'k--',linewidth=1.5,label='True label')
plt.title('Predicted label')
plt.xlabel('Time [s]')
plt.ylabel('Predicted label')
plt.legend()
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

fig, ax = plt.subplots(1,1,layout='constrained',figsize=(6,4))
plt.plot(time,y_sim,'b') 
plt.title('Motion with k/k_health = {:.2f}'.format(h_sim))
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
# plt.legend()
plt.grid()
