import torch, torch.nn as nn
import snntorch as snn
import time
import numpy as np

from snntorch import surrogate
import snntorch.functional as SF

from scipy.io import savemat
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1 hidden-layer SNN model
class ANN_fc(nn.Module):
    def __init__(self, in_sz, N, out_sz):
        super().__init__()
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_sz, N, bias=False)
        self.h1 = nn.ReLU()
        self.fc2 = nn.Linear(N, out_sz, bias=False)
    def forward(self, x):
        fl_x = self.fl(x)
        x1 = self.h1(self.fc1(fl_x))
        y = self.fc2(x1)
        return y

# 9 layer All CNN model
class All_CNN(nn.Module):
    def __init__(self, in_ch, out_sz):
        super().__init__()
        self.cnn1_1 = nn.Conv2d(in_ch, 96, (3,3), padding='same', bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.cnn1_2 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.cnn1_3 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.cnn2_1 = nn.Conv2d(96, 192, (3,3), padding='same', bias=False)
        self.cnn2_2 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.cnn2_3 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.cnn3_1 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.cnn3_2 = nn.Conv2d(192, 192, (1,1), padding='same', bias=False)
        self.cnn3_3 = nn.Conv2d(192, out_sz, (1,1), padding='same', bias=False)
        self.avp = nn.AvgPool2d(8, 8)
    def forward(self, x):
        cnn1 = self.relu(self.cnn1_1(x))
        cnn1 = self.dropout(cnn1)
        cnn2 = self.relu(self.cnn1_2(cnn1))
        cnn2 = self.dropout(cnn2)
        cnn3 = self.relu(self.cnn1_3(cnn2))
        mpool1 = self.mp1(cnn3)
        cnn4 = self.relu(self.cnn2_1(mpool1))
        cnn4 = self.dropout(cnn4)
        cnn5 = self.relu(self.cnn2_2(cnn4))
        cnn5 = self.dropout(cnn5)
        cnn6 = self.relu(self.cnn2_3(cnn5))
        mpool2 = self.mp2(cnn6)
        cnn7 = self.relu(self.cnn3_1(mpool2))
        cnn7 = self.dropout(cnn7)
        cnn8 = self.relu(self.cnn3_2(cnn7))
        cnn8 = self.dropout(cnn8)
        cnn9 = self.relu(self.cnn3_3(cnn8))
        y = self.avp(cnn9)
        return torch.reshape(y, (y.shape[0], -1))

def test_accuracy(data_loader, net):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = net(data)
            _, predicted = torch.max(spk_rec.data, 1)
            acc += (predicted == targets).sum().item()
            total += spk_rec.size(0)
    return acc/total

if __name__ == '__main__':

    batch_size = 256
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    wt_limit = 0.1
    Wt_bits = 3
    max_Wt_val = 2**Wt_bits - 1
    zero_pt = 2**(Wt_bits-1)-0.5
    quant_step = np.float32((2*wt_limit)/max_Wt_val)

    allowed_weights = (np.arange(0, 2**Wt_bits) - zero_pt)*quant_step
    if Wt_bits == 1:
        allowed_weights = np.float32(np.array([-wt_limit, wt_limit]))
        zero_pt = 0.5
    '''
    if Wt_bits == 1.5:
        allowed_weights = np.float32(np.array([-wt_limit, 0, wt_limit]))
        zero_pt = 0
        quant_step = wt_limit
    '''
    allowed_weights = np.float32(allowed_weights)
    print('quant step: ', quant_step)
    print('max val: ', max_Wt_val)
    print('allowed weights: ', allowed_weights)

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    data_path='./data/CIFAR_100'
    cifar_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    #data_path='./data/CIFAR_10'
    #cifar_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    #cifar_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    data, targets = next(iter(train_loader))
    print(data.shape)
    plt.figure()
    plt.imsave('cifar10_sample.png', np.transpose(data[5].numpy(), (1, 2, 0)))

    in_ch = 3
    #out_sz = 10 #for CIFAR-10
    out_sz = 100#for CIFAR-100

    num_epochs = 200
    reg_lambda = 1*1e-10

    loss_hist = [] # record loss over iterations

    mu_LRS = 3.99e2
    mu_HRS = 5.74e8
    mu_1 = 1.0
    mu_0 = 0.001
    print("mu_0: ", mu_0)
    print("mu_1: ", mu_1)

    sigs_by_mu_0 = [0.0]
    dead_bit_fracs = [0.1]

    W1_1_vars_1 = np.zeros((Wt_bits, 96, in_ch, 3, 3))
    W1_2_vars_1 = np.zeros((Wt_bits, 96, 96, 3, 3))
    W1_3_vars_1 = np.zeros((Wt_bits, 96, 96, 3, 3))

    W2_1_vars_1 = np.zeros((Wt_bits, 192, 96, 3, 3))
    W2_2_vars_1 = np.zeros((Wt_bits, 192, 192, 3, 3))
    W2_3_vars_1 = np.zeros((Wt_bits, 192, 192, 3, 3))

    W3_1_vars_1 = np.zeros((Wt_bits, 192, 192, 3, 3))
    W3_2_vars_1 = np.zeros((Wt_bits, 192, 192, 1, 1))
    W3_3_vars_1 = np.zeros((Wt_bits, out_sz, 192, 1, 1))

    #zero bits
    W1_1_vars_0 = mu_0 + np.zeros((Wt_bits, 96, in_ch, 3, 3))
    W1_2_vars_0 = mu_0 + np.zeros((Wt_bits, 96, 96, 3, 3))
    W1_3_vars_0 = mu_0 + np.zeros((Wt_bits, 96, 96, 3, 3))

    W2_1_vars_0 = mu_0 + np.zeros((Wt_bits, 192, 96, 3, 3))
    W2_2_vars_0 = mu_0 + np.zeros((Wt_bits, 192, 192, 3, 3))
    W2_3_vars_0 = mu_0 + np.zeros((Wt_bits, 192, 192, 3, 3))

    W3_1_vars_0 = mu_0 + np.zeros((Wt_bits, 192, 192, 3, 3))
    W3_2_vars_0 = mu_0 + np.zeros((Wt_bits, 192, 192, 1, 1))
    W3_3_vars_0 = mu_0 + np.zeros((Wt_bits, out_sz, 192, 1, 1))

    start_time = time.time()

    all_accs = []
    record_epochs = []
    run_start = time.time()

    reg_sig = quant_step
    #reg_sig = quant_step/2
    #reg_sig_ch_rate = 4 #4 for 3-bit
    #reg_sig_ch_rate = 5
    #stop_epoch = 50 #setting used for cifar-10
    stop_epoch = 50 #setting used for cifar-100
    lambda_factor = (1e4)**(1/stop_epoch)
    #sig_factor = 2**(1/stop_epoch) #setting used for cifar-10
    sig_factor = 2**(1/stop_epoch) #setting used for cifar-100
    save_folder = str(Wt_bits) + '_bit/'
    var_aware = True
    
    for dead_bit_frac in dead_bit_fracs:
        print('dead bits frac: ', dead_bit_frac)
        print('is dead bit aware? ', var_aware)
        All_CNN_net = All_CNN(in_ch, out_sz).to(device)
        All_CNN_net_test = All_CNN(in_ch, out_sz).to(device)
        allowed_weights = torch.from_numpy(allowed_weights).to(device)
        optimizer = torch.optim.Adam(All_CNN_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        #loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        loss_fn = nn.CrossEntropyLoss()

        W1_1_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 96, in_ch, 3, 3))
        W1_2_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 96, 96, 3, 3))
        W1_3_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 96, 96, 3, 3))

        W2_1_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 192, 96, 3, 3))
        W2_2_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 192, 192, 3, 3))
        W2_3_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 192, 192, 3, 3))

        W3_1_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 192, 192, 3, 3))
        W3_2_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, 192, 192, 1, 1))
        W3_3_vars_1 = np.random.uniform(low=0.0001, high=1.0, size=(Wt_bits, out_sz, 192, 1, 1))
        
        #Random fixed percentage of the bits are clamped to 0
        W1_1_vars_1[W1_1_vars_1<dead_bit_frac] = 0
        W1_2_vars_1[W1_2_vars_1<dead_bit_frac] = 0
        W1_3_vars_1[W1_3_vars_1<dead_bit_frac] = 0
        
        W2_1_vars_1[W2_1_vars_1<dead_bit_frac] = 0
        W2_2_vars_1[W2_2_vars_1<dead_bit_frac] = 0
        W2_3_vars_1[W2_3_vars_1<dead_bit_frac] = 0
        
        W3_1_vars_1[W3_1_vars_1<dead_bit_frac] = 0
        W3_2_vars_1[W3_2_vars_1<dead_bit_frac] = 0
        W3_3_vars_1[W3_3_vars_1<dead_bit_frac] = 0

        W1_1_vars_1[W1_1_vars_1>0] = 1
        W1_2_vars_1[W1_2_vars_1>0] = 1
        W1_3_vars_1[W1_3_vars_1>0] = 1
        
        W2_1_vars_1[W2_1_vars_1>0] = 1
        W2_2_vars_1[W2_2_vars_1>0] = 1
        W2_3_vars_1[W2_3_vars_1>0] = 1
        
        W3_1_vars_1[W3_1_vars_1>0] = 1
        W3_2_vars_1[W3_2_vars_1>0] = 1
        W3_3_vars_1[W3_3_vars_1>0] = 1
        
        #Initialize Validity matrices
        W1_1_validity = np.ones((2**Wt_bits, 96, in_ch, 3, 3))
        W1_2_validity = np.ones((2**Wt_bits, 96, 96, 3, 3))
        W1_3_validity = np.ones((2**Wt_bits, 96, 96, 3, 3))
        
        W2_1_validity = np.ones((2**Wt_bits, 192, 96, 3, 3))
        W2_2_validity = np.ones((2**Wt_bits, 192, 192, 3, 3))
        W2_3_validity = np.ones((2**Wt_bits, 192, 192, 3, 3))
        
        W3_1_validity = np.ones((2**Wt_bits, 192, 192, 3, 3))
        W3_2_validity = np.ones((2**Wt_bits, 192, 192, 1, 1))
        W3_3_validity = np.ones((2**Wt_bits, out_sz, 192, 1, 1))
        
        #build the validity matrices
        for level in range(2**Wt_bits):
            level_cp = level
            level_bits = []
            for bit in range(Wt_bits):
                level_bit = level_cp%2
                level_cp = level_cp//2
                level_bits.append(level_bit)

            #Initialize possible weight matrices
            W1_1_possible = np.zeros((96, in_ch, 3, 3))
            W1_2_possible = np.zeros((96, 96, 3, 3))
            W1_3_possible = np.zeros((96, 96, 3, 3))
            
            W2_1_possible = np.zeros((192, 96, 3, 3))
            W2_2_possible = np.zeros((192, 192, 3, 3))
            W2_3_possible = np.zeros((192, 192, 3, 3))
            
            W3_1_possible = np.zeros((192, 192, 3, 3))
            W3_2_possible = np.zeros((192, 192, 1, 1))
            W3_3_possible = np.zeros((out_sz, 192, 1, 1))
            
            for bit_ind in range(len(level_bits)):
                W1_1_possible = W1_1_possible + (2**bit_ind)*level_bits[bit_ind]*W1_1_vars_1[bit_ind]
                W1_2_possible = W1_2_possible + (2**bit_ind)*level_bits[bit_ind]*W1_2_vars_1[bit_ind]
                W1_3_possible = W1_3_possible + (2**bit_ind)*level_bits[bit_ind]*W1_3_vars_1[bit_ind]
                
                W2_1_possible = W2_1_possible + (2**bit_ind)*level_bits[bit_ind]*W2_1_vars_1[bit_ind]
                W2_2_possible = W2_2_possible + (2**bit_ind)*level_bits[bit_ind]*W2_2_vars_1[bit_ind]
                W2_3_possible = W2_3_possible + (2**bit_ind)*level_bits[bit_ind]*W2_3_vars_1[bit_ind]
                
                W3_1_possible = W3_1_possible + (2**bit_ind)*level_bits[bit_ind]*W3_1_vars_1[bit_ind]
                W3_2_possible = W3_2_possible + (2**bit_ind)*level_bits[bit_ind]*W3_2_vars_1[bit_ind]
                W3_3_possible = W3_3_possible + (2**bit_ind)*level_bits[bit_ind]*W3_3_vars_1[bit_ind]
            
            W1_1_validity[level, W1_1_possible<level] = 0
            W1_2_validity[level, W1_2_possible<level] = 0
            W1_3_validity[level, W1_3_possible<level] = 0
            
            W2_1_validity[level, W2_1_possible<level] = 0
            W2_2_validity[level, W2_2_possible<level] = 0
            W2_3_validity[level, W2_3_possible<level] = 0
            
            W3_1_validity[level, W3_1_possible<level] = 0
            W3_2_validity[level, W3_2_possible<level] = 0
            W3_3_validity[level, W3_3_possible<level] = 0

        W1_1_validity_t = torch.from_numpy(np.float32(W1_1_validity)).to(device)
        W1_2_validity_t = torch.from_numpy(np.float32(W1_2_validity)).to(device)
        W1_3_validity_t = torch.from_numpy(np.float32(W1_3_validity)).to(device)
        
        W2_1_validity_t = torch.from_numpy(np.float32(W2_1_validity)).to(device)
        W2_2_validity_t = torch.from_numpy(np.float32(W2_2_validity)).to(device)
        W2_3_validity_t = torch.from_numpy(np.float32(W2_3_validity)).to(device)
        
        W3_1_validity_t = torch.from_numpy(np.float32(W3_1_validity)).to(device)
        W3_2_validity_t = torch.from_numpy(np.float32(W3_2_validity)).to(device)
        W3_3_validity_t = torch.from_numpy(np.float32(W3_3_validity)).to(device)

        # training loop
        for epoch in range(num_epochs):
            #if epoch%reg_sig_ch_rate == 0 and epoch<=20:
            #    reg_sig = np.float32(reg_sig/1.2)
            #reg_sig = np.float32(reg_sig/1.02)
            if epoch<stop_epoch:
                reg_lambda = reg_lambda*lambda_factor
                reg_sig = reg_sig/sig_factor
            print('reg_lambda: ', reg_lambda)
            print('sig_denominator: ', quant_step/reg_sig)
            for i, (data, targets) in enumerate(iter(train_loader)):
                data = data.to(device)
                targets = targets.to(device)
                All_CNN_net.train()

                #Train without variation
                spk_rec = All_CNN_net(data) # forward-pass
                #print(spk_rec.shape)
                #print(targets.shape)

                loss_val = loss_fn(spk_rec, targets) # loss calculation
                reg_loss = 0
                for j in range(allowed_weights.shape[0]):
                    if var_aware:
                        #Dead bit-aware
                        reg_loss = reg_loss - reg_lambda*torch.sum(W1_1_validity_t[j]*torch.exp(-((All_CNN_net.cnn1_1.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(W1_2_validity_t[j]*torch.exp(-((All_CNN_net.cnn1_2.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(W1_3_validity_t[j]*torch.exp(-((All_CNN_net.cnn1_3.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        
                        reg_loss = reg_loss - reg_lambda*torch.sum(W2_1_validity_t[j]*torch.exp(-((All_CNN_net.cnn2_1.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(W2_2_validity_t[j]*torch.exp(-((All_CNN_net.cnn2_2.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(W2_3_validity_t[j]*torch.exp(-((All_CNN_net.cnn2_3.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        
                        reg_loss = reg_loss - reg_lambda*torch.sum(W3_1_validity_t[j]*torch.exp(-((All_CNN_net.cnn3_1.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(W3_2_validity_t[j]*torch.exp(-((All_CNN_net.cnn3_2.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(W3_3_validity_t[j]*torch.exp(-((All_CNN_net.cnn3_3.weight - allowed_weights[j])**2)/(reg_sig**2)))
                    else:
                        #Only quantization aware
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn1_1.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn1_2.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn1_3.weight - allowed_weights[j])**2)/(reg_sig**2)))

                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn2_1.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn2_2.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn2_3.weight - allowed_weights[j])**2)/(reg_sig**2)))

                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn3_1.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn3_2.weight - allowed_weights[j])**2)/(reg_sig**2)))
                        reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((All_CNN_net.cnn3_3.weight - allowed_weights[j])**2)/(reg_sig**2)))
                loss_val = loss_val + reg_loss
                optimizer.zero_grad() # null gradients
                loss_val.backward() # calculate gradients
                optimizer.step() # update weights

                #clamp weights
                
                All_CNN_net.cnn1_1.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                All_CNN_net.cnn1_2.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                All_CNN_net.cnn1_3.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])

                All_CNN_net.cnn2_1.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                All_CNN_net.cnn2_2.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                All_CNN_net.cnn2_3.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                
                All_CNN_net.cnn3_1.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                All_CNN_net.cnn3_2.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])
                All_CNN_net.cnn3_3.weight.data.clamp_(allowed_weights[0], allowed_weights[-1])

            print(f"Epoch {epoch}")
            acc = test_accuracy(test_loader, All_CNN_net)
            print("test accuracy(%): ", acc * 100)

            if (epoch%4 == 3 and epoch<100) or (epoch>100 and epoch%3 == 2):
                #quantize weights to fixed-point precision
                W1_1_np = All_CNN_net.cnn1_1.weight.data.cpu().numpy()
                W1_2_np = All_CNN_net.cnn1_2.weight.data.cpu().numpy()
                W1_3_np = All_CNN_net.cnn1_3.weight.data.cpu().numpy()
                
                W2_1_np = All_CNN_net.cnn2_1.weight.data.cpu().numpy()
                W2_2_np = All_CNN_net.cnn2_2.weight.data.cpu().numpy()
                W2_3_np = All_CNN_net.cnn2_3.weight.data.cpu().numpy()
                
                W3_1_np = All_CNN_net.cnn3_1.weight.data.cpu().numpy()
                W3_2_np = All_CNN_net.cnn3_2.weight.data.cpu().numpy()
                W3_3_np = All_CNN_net.cnn3_3.weight.data.cpu().numpy()
                
                allowed_weights_np = allowed_weights.cpu().numpy()

                W1_1_quant = np.int32(np.zeros_like(W1_1_np))
                W1_2_quant = np.int32(np.zeros_like(W1_2_np))
                W1_3_quant = np.int32(np.zeros_like(W1_3_np))
                
                W2_1_quant = np.int32(np.zeros_like(W2_1_np))
                W2_2_quant = np.int32(np.zeros_like(W2_2_np))
                W2_3_quant = np.int32(np.zeros_like(W2_3_np))
                
                W3_1_quant = np.int32(np.zeros_like(W3_1_np))
                W3_2_quant = np.int32(np.zeros_like(W3_2_np))
                W3_3_quant = np.int32(np.zeros_like(W3_3_np))

                for k in range(allowed_weights_np.shape[0]):
                    W1_1_quant[(W1_1_np<(allowed_weights_np[k]+(quant_step/2))) & (W1_1_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    W1_2_quant[(W1_2_np<(allowed_weights_np[k]+(quant_step/2))) & (W1_2_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    W1_3_quant[(W1_3_np<(allowed_weights_np[k]+(quant_step/2))) & (W1_3_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    
                    W2_1_quant[(W2_1_np<(allowed_weights_np[k]+(quant_step/2))) & (W2_1_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    W2_2_quant[(W2_2_np<(allowed_weights_np[k]+(quant_step/2))) & (W2_2_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    W2_3_quant[(W2_3_np<(allowed_weights_np[k]+(quant_step/2))) & (W2_3_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    
                    W3_1_quant[(W3_1_np<(allowed_weights_np[k]+(quant_step/2))) & (W3_1_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    W3_2_quant[(W3_2_np<(allowed_weights_np[k]+(quant_step/2))) & (W3_2_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                    W3_3_quant[(W3_3_np<(allowed_weights_np[k]+(quant_step/2))) & (W3_3_np>=(allowed_weights_np[k]-(quant_step/2)))] = k

                W1_1_fixed_fl_var = np.float32(np.zeros_like(W1_1_np))
                W1_2_fixed_fl_var = np.float32(np.zeros_like(W1_2_np))
                W1_3_fixed_fl_var = np.float32(np.zeros_like(W1_3_np))
                
                W2_1_fixed_fl_var = np.float32(np.zeros_like(W2_1_np))
                W2_2_fixed_fl_var = np.float32(np.zeros_like(W2_2_np))
                W2_3_fixed_fl_var = np.float32(np.zeros_like(W2_3_np))
                
                W3_1_fixed_fl_var = np.float32(np.zeros_like(W3_1_np))
                W3_2_fixed_fl_var = np.float32(np.zeros_like(W3_2_np))
                W3_3_fixed_fl_var = np.float32(np.zeros_like(W3_3_np))

                for bit in range(Wt_bits):
                    W1_1_bit = (W1_1_quant % 2)
                    W1_2_bit = (W1_2_quant % 2)
                    W1_3_bit = (W1_3_quant % 2)
                    
                    W2_1_bit = (W2_1_quant % 2)
                    W2_2_bit = (W2_2_quant % 2)
                    W2_3_bit = (W2_3_quant % 2)
                    
                    W3_1_bit = (W3_1_quant % 2)
                    W3_2_bit = (W3_2_quant % 2)
                    W3_3_bit = (W3_3_quant % 2)

                    W1_1_fixed_fl_var = W1_1_fixed_fl_var + np.float32((2**bit)*(W1_1_vars_1[bit]*W1_1_bit + W1_1_vars_0[bit]*(1 - W1_1_bit)))
                    W1_2_fixed_fl_var = W1_2_fixed_fl_var + np.float32((2**bit)*(W1_2_vars_1[bit]*W1_2_bit + W1_2_vars_0[bit]*(1 - W1_2_bit)))
                    W1_3_fixed_fl_var = W1_3_fixed_fl_var + np.float32((2**bit)*(W1_3_vars_1[bit]*W1_3_bit + W1_3_vars_0[bit]*(1 - W1_3_bit)))
                    
                    W2_1_fixed_fl_var = W2_1_fixed_fl_var + np.float32((2**bit)*(W2_1_vars_1[bit]*W2_1_bit + W2_1_vars_0[bit]*(1 - W2_1_bit)))
                    W2_2_fixed_fl_var = W2_2_fixed_fl_var + np.float32((2**bit)*(W2_2_vars_1[bit]*W2_2_bit + W2_2_vars_0[bit]*(1 - W2_2_bit)))
                    W2_3_fixed_fl_var = W2_3_fixed_fl_var + np.float32((2**bit)*(W2_3_vars_1[bit]*W2_3_bit + W2_3_vars_0[bit]*(1 - W2_3_bit)))
                    
                    W3_1_fixed_fl_var = W3_1_fixed_fl_var + np.float32((2**bit)*(W3_1_vars_1[bit]*W3_1_bit + W3_1_vars_0[bit]*(1 - W3_1_bit)))
                    W3_2_fixed_fl_var = W3_2_fixed_fl_var + np.float32((2**bit)*(W3_2_vars_1[bit]*W3_2_bit + W3_2_vars_0[bit]*(1 - W3_2_bit)))
                    W3_3_fixed_fl_var = W3_3_fixed_fl_var + np.float32((2**bit)*(W3_3_vars_1[bit]*W3_3_bit + W3_3_vars_0[bit]*(1 - W3_3_bit)))

                    W1_1_quant = W1_1_quant//2
                    W1_2_quant = W1_2_quant//2
                    W1_3_quant = W1_3_quant//2
                    
                    W2_1_quant = W2_1_quant//2
                    W2_2_quant = W2_2_quant//2
                    W2_3_quant = W2_3_quant//2
                    
                    W3_1_quant = W3_1_quant//2
                    W3_2_quant = W3_2_quant//2
                    W3_3_quant = W3_3_quant//2

                #scale weights back to (-wt_limit, wt_limit) range, but with Fixed-point precision
                W1_1_q = np.float32((W1_1_fixed_fl_var - zero_pt)*quant_step)
                W1_2_q = np.float32((W1_2_fixed_fl_var - zero_pt)*quant_step)
                W1_3_q = np.float32((W1_3_fixed_fl_var - zero_pt)*quant_step)
                
                W2_1_q = np.float32((W2_1_fixed_fl_var - zero_pt)*quant_step)
                W2_2_q = np.float32((W2_2_fixed_fl_var - zero_pt)*quant_step)
                W2_3_q = np.float32((W2_3_fixed_fl_var - zero_pt)*quant_step)
                
                W3_1_q = np.float32((W3_1_fixed_fl_var - zero_pt)*quant_step)
                W3_2_q = np.float32((W3_2_fixed_fl_var - zero_pt)*quant_step)
                W3_3_q = np.float32((W3_3_fixed_fl_var - zero_pt)*quant_step)
                '''
                W1_1_flat = np.reshape(W1_1_np, -1)
                W1_1_q_flat = np.reshape(W1_1_q, -1)
                plt.hist(0.05+W1_1_flat/quant_step, bins=8*(2**Wt_bits), color='blue')
                plt.hist(W1_1_q_flat/quant_step, bins=8*(2**Wt_bits), color='red')
                plt.legend(['before quantizing', 'after quantizing'])
                plt.title('First conv layer weights')
                plt.savefig(save_folder + 'W1_1_quant_epoch_'+str(epoch)+'_bits_'+str(Wt_bits)+'.png', dpi=300)

                W2_2_flat = np.reshape(W2_2_np, -1)
                W2_2_q_flat = np.reshape(W2_2_q, -1)
                plt.hist(0.05+W2_2_flat/quant_step, bins=8*(2**Wt_bits), color='blue')
                plt.hist(W2_2_q_flat/quant_step, bins=8*(2**Wt_bits), color='red')
                plt.legend(['before quantizing', 'after quantizing'])
                plt.title('Middle conv layer weights')
                plt.savefig(save_folder + 'W2_2_quant_epoch_'+str(epoch)+'_bits_'+str(Wt_bits)+'.png', dpi=300)

                W3_3_flat = np.reshape(W3_3_np, -1)
                W3_3_q_flat = np.reshape(W3_3_q, -1)
                plt.hist(0.05+W3_3_flat/quant_step, bins=8*(2**Wt_bits), color='blue')
                plt.hist(W3_3_q_flat/quant_step, bins=8*(2**Wt_bits), color='red')
                plt.legend(['before quantizing', 'after quantizing'])
                plt.title('Last conv layer weights')
                plt.savefig(save_folder + 'W3_3_quant_epoch_'+str(epoch)+'_bits_'+str(Wt_bits)+'.png', dpi=300)
                '''
                #Set weights in the test network
                All_CNN_net_test.cnn1_1.weight = nn.Parameter(torch.from_numpy(W1_1_q).to(device))
                All_CNN_net_test.cnn1_2.weight = nn.Parameter(torch.from_numpy(W1_2_q).to(device))
                All_CNN_net_test.cnn1_3.weight = nn.Parameter(torch.from_numpy(W1_3_q).to(device))
                
                All_CNN_net_test.cnn2_1.weight = nn.Parameter(torch.from_numpy(W2_1_q).to(device))
                All_CNN_net_test.cnn2_2.weight = nn.Parameter(torch.from_numpy(W2_2_q).to(device))
                All_CNN_net_test.cnn2_3.weight = nn.Parameter(torch.from_numpy(W2_3_q).to(device))
                
                All_CNN_net_test.cnn3_1.weight = nn.Parameter(torch.from_numpy(W3_1_q).to(device))
                All_CNN_net_test.cnn3_2.weight = nn.Parameter(torch.from_numpy(W3_2_q).to(device))
                All_CNN_net_test.cnn3_3.weight = nn.Parameter(torch.from_numpy(W3_3_q).to(device))
                
                #print(f"Epoch {epoch}, \nTrain Loss: {loss_val.item():.2f}")
                #Test using the copy network for testing
                post_q_acc = test_accuracy(test_loader, All_CNN_net_test)
                
                #Set weights in the train network
                All_CNN_net = All_CNN(in_ch, out_sz).to(device)
                All_CNN_net.cnn1_1.weight = nn.Parameter(torch.from_numpy(W1_1_q).to(device), requires_grad=True)
                All_CNN_net.cnn1_2.weight = nn.Parameter(torch.from_numpy(W1_2_q).to(device), requires_grad=True)
                All_CNN_net.cnn1_3.weight = nn.Parameter(torch.from_numpy(W1_3_q).to(device), requires_grad=True)
                
                All_CNN_net.cnn2_1.weight = nn.Parameter(torch.from_numpy(W2_1_q).to(device), requires_grad=True)
                All_CNN_net.cnn2_2.weight = nn.Parameter(torch.from_numpy(W2_2_q).to(device), requires_grad=True)
                All_CNN_net.cnn2_3.weight = nn.Parameter(torch.from_numpy(W2_3_q).to(device), requires_grad=True)
                
                All_CNN_net.cnn3_1.weight = nn.Parameter(torch.from_numpy(W3_1_q).to(device), requires_grad=True)
                All_CNN_net.cnn3_2.weight = nn.Parameter(torch.from_numpy(W3_2_q).to(device), requires_grad=True)
                All_CNN_net.cnn3_3.weight = nn.Parameter(torch.from_numpy(W3_3_q).to(device), requires_grad=True)
                optimizer = torch.optim.Adam(All_CNN_net.parameters(), lr=1e-3, betas=(0.9, 0.999))

                print("Post-quantization test accuracy(%): ", post_q_acc * 100)
                all_accs.append(post_q_acc)
                record_epochs.append(epoch)

    run_end = time.time()
    print("total runtime: ", run_end - run_start, "seconds")

    print('record epochs: ', record_epochs)
    print('record post quantization test accuracies: ', all_accs)

    sv_dict = {"record_epochs": record_epochs, "all_accs": all_accs}
    if var_aware:
        save_filename = save_folder + "CIFAR100_training_dead_bit_aware_with_10pc_var_type2_" + str(Wt_bits) + "bit_2.mat"
    else:
        save_filename = save_folder + "CIFAR100_training_only_quant_aware_with_10pc_var_type2_" + str(Wt_bits) + "bit_2.mat"
    savemat(save_filename, sv_dict)
