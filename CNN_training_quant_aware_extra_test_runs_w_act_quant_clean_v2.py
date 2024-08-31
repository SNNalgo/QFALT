import torch, torch.nn as nn
#import snntorch as snn
import time
import numpy as np
import argparse

#from snntorch import surrogate
#import snntorch.functional as SF

#from scipy.io import savemat
#import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='Quant Aware Training')
parser.add_argument('--nn_model', default='all_cnn', type=str, help='neural network model to train')
parser.add_argument('--wt_bits', default=2, type=int, help='number of bits for weight quantization')
#parser.add_argument('--wt_limit', default=0.1, type=int, help='weight limit for weight quantization')
parser.add_argument('--act_bits', default=4, type=int, help='number of bits for activation quantization')
parser.add_argument('--mu_LRS', default=3.99e2, type=float, help='mu of LRS of weights')
parser.add_argument('--mu_HRS', default=5.74e8, type=float, help='mu of HRS of weights')
parser.add_argument('--sig_by_mu_LRS', default=0.01, type=float, help='sigma/mu of LRS of weights')
parser.add_argument('--sig_by_mu_HRS', default=0.01, type=float, help='sigma/mu of HRS of weights')
parser.add_argument('--quant_aware', action='store_true', help='use quant-aware loss for training (only for small wt_bits)')
parser.add_argument('--num_epochs', default=400, type=int, help='number of epochs for training')

args = parser.parse_args()

def quantize_traditional(v, scale, bits):
    q_min = 0
    q_max = 2**bits - 1
    v = v/scale
    v = torch.clamp(v, min=q_min, max=q_max)
    v = torch.round(v)
    v = v*scale
    return v

def quantize_traditional_signed(v, scale, bits):
    q_min = -2**(bits-1)
    q_max = 2**(bits-1) - 1
    v = v/scale
    v = torch.clamp(v, min=q_min, max=q_max)
    v = torch.round(v)
    v = v*scale
    return v

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

    def forward_act(self, x):
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
        return [cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9], torch.reshape(y, (y.shape[0], -1))
    
    def forward_act_quant(self, x, scale_factors, bits):
        cnn1 = self.relu(self.cnn1_1(x))
        cnn1 = quantize_traditional(cnn1, scale_factors[0], bits)
        cnn1 = self.dropout(cnn1)
        cnn2 = self.relu(self.cnn1_2(cnn1))
        cnn2 = quantize_traditional(cnn2, scale_factors[1], bits)
        cnn2 = self.dropout(cnn2)
        cnn3 = self.relu(self.cnn1_3(cnn2))
        cnn3 = quantize_traditional(cnn3, scale_factors[2], bits)
        mpool1 = self.mp1(cnn3)
        cnn4 = self.relu(self.cnn2_1(mpool1))
        cnn4 = quantize_traditional(cnn4, scale_factors[3], bits)
        cnn4 = self.dropout(cnn4)
        cnn5 = self.relu(self.cnn2_2(cnn4))
        cnn5 = quantize_traditional(cnn5, scale_factors[4], bits)
        cnn5 = self.dropout(cnn5)
        cnn6 = self.relu(self.cnn2_3(cnn5))
        cnn6 = quantize_traditional(cnn6, scale_factors[5], bits)
        mpool2 = self.mp2(cnn6)
        cnn7 = self.relu(self.cnn3_1(mpool2))
        cnn7 = quantize_traditional(cnn7, scale_factors[6], bits)
        cnn7 = self.dropout(cnn7)
        cnn8 = self.relu(self.cnn3_2(cnn7))
        cnn8 = quantize_traditional(cnn8, scale_factors[7], bits)
        cnn8 = self.dropout(cnn8)
        cnn9 = self.relu(self.cnn3_3(cnn8))
        cnn9 = quantize_traditional(cnn9, scale_factors[8], bits)
        y = self.avp(cnn9)
        return torch.reshape(y, (y.shape[0], -1))

# VGG13 model
class VGG13(nn.Module):
    def __init__(self,in_ch=3, num_classes=10, affine=False, bias=True):
        super(VGG13, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(64, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512, affine=affine),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward_act(self, x):
        activations = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x)
        return activations, x
    
    def forward_act_quant(self, x, scale_factors, bits):
        count = 0
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = quantize_traditional(x, scale_factors[count], bits)
                count += 1
        x = torch.flatten(x, 1)
        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = quantize_traditional(x, scale_factors[count], bits)
                count += 1
        return x

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

def test_accuracy_w_act_quant(data_loader, net, scale_factors, bits):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = net.forward_act_quant(data, scale_factors, bits)
            _, predicted = torch.max(spk_rec.data, 1)
            acc += (predicted == targets).sum().item()
            total += spk_rec.size(0)
    return acc/total

if __name__ == '__main__':

    batch_size = 256
    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    Wt_bits = args.wt_bits
    act_bits = args.act_bits
    quant_aware = args.quant_aware

    # Define a transform
    train_transform = transforms.Compose([
            # add some data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    data_path='./data/CIFAR_10'
    cifar_train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
    cifar_test = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, num_workers=4)

    data, targets = next(iter(train_loader))
    print(data.shape)

    in_ch = 3
    out_sz = 10 #for CIFAR-10

    num_epochs = args.num_epochs

    if args.nn_model == 'all_cnn':
        model = All_CNN(in_ch, out_sz).to(device)
        model_test = All_CNN(in_ch, out_sz).to(device)
        
        reg_lambda = 1*1e-10
        stop_epoch = 50
        lambda_factor = (1e4)**(1/stop_epoch)
        sig_factor = 2**(1/stop_epoch)
    
    if args.nn_model == 'vgg13':
        model = VGG13(in_ch, out_sz).to(device)
        model_test = VGG13(in_ch, out_sz).to(device)

        reg_lambda = 1*1e-10
        stop_epoch = 50
        lambda_factor = (1e4)**(1/stop_epoch)
        sig_factor = 2**(1/stop_epoch)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    mu_LRS = args.mu_LRS
    mu_HRS = args.mu_HRS
    
    sig_by_mu_LRS = args.sig_by_mu_LRS
    sig_by_mu_HRS = args.sig_by_mu_HRS
    
    W_vars_1 = []
    W_vars_0 = []
    
    param_count = 0
    for (names, params) in model.named_parameters():
        if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
            params_np_shape = params.detach().cpu().numpy().shape
            params_var_shape = (Wt_bits,) + params_np_shape
            params_var_LRS = np.random.normal(loc=mu_LRS, scale=sig_by_mu_LRS*mu_LRS, size=params_var_shape)
            #params_var_LRS = np.random.lognormal(mean=mu_LRS, sigma=sig_by_mu_LRS*mu_LRS, size=params_var_shape)
            params_var_LRS[params_var_LRS<=0] = mu_LRS #ensuring all positive values
            params_var_HRS = np.random.normal(loc=mu_HRS, scale=sig_by_mu_HRS*mu_HRS, size=params_var_shape)
            #params_var_HRS = np.random.lognormal(mean=mu_HRS, sigma=sig_by_mu_HRS*mu_HRS, size=params_var_shape)
            params_var_HRS[params_var_HRS<=0] = mu_HRS #ensuring all positive values
            W_var_1 = mu_LRS/params_var_LRS
            W_var_0 = mu_LRS/params_var_HRS
            W_vars_1.append(W_var_1)
            W_vars_0.append(W_var_0)
            param_count = param_count + 1

    start_time = time.time()

    all_accs = []
    record_epochs = []
    run_start = time.time()

    save_folder = str(Wt_bits) + '_bit/'
    
    allowed_weights_full = []
    reg_sigs = []
    for (names, params) in model.named_parameters():
        if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
            average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
            q_max = (2**Wt_bits - 1)/2

            #Original scaling, based on LSQ paper: works well for 4/4
            scale = 2*average_mag_weights/np.sqrt(q_max)
            wt_limit = q_max*scale

            if args.wt_bits == 2:
                wt_limit = 0.1
            
            max_Wt_val = 2**Wt_bits - 1
            zero_pt = 2**(Wt_bits-1)-0.5
            quant_step = np.float32((2*wt_limit)/max_Wt_val)
            allowed_weights = np.float32((np.arange(0, 2**Wt_bits) - zero_pt)*quant_step).tolist()
            allowed_weights_full.append(allowed_weights)
            reg_sigs.append(quant_step)
            
            print('layer: ', names)
            print('quant step: ', quant_step)
            print('max val: ', max_Wt_val)
            print('allowed weights: ', allowed_weights)
    allowed_weights_full = np.array(allowed_weights_full)
    allowed_weights_full = torch.from_numpy(allowed_weights_full).to(device)
    reg_sigs = np.array(reg_sigs)
    
    max_non_quant_acc = 0
    # training loop
    print('mu_HRS: ', mu_HRS)
    print('mu_LRS: ', mu_LRS)
    print('sig_by_mu_HRS: ', sig_by_mu_HRS)
    print('sig_by_mu_LRS: ', sig_by_mu_LRS)
    for epoch in range(num_epochs):
        #allowed_weights_full = []
        #reg_sigs = []
        #for (names, params) in model.named_parameters():
        #    if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
        #        average_mag_weights = np.mean(np.abs(params.detach().cpu().numpy()))
        #        q_max = (2**Wt_bits - 1)/2
        #        scale = 2*average_mag_weights/np.sqrt(q_max)
        #        wt_limit = q_max*scale

        #        max_Wt_val = 2**Wt_bits - 1
        #        zero_pt = 2**(Wt_bits-1)-0.5
        #        quant_step = np.float32((2*wt_limit)/max_Wt_val)
        #        allowed_weights = np.float32((np.arange(0, 2**Wt_bits) - zero_pt)*quant_step).tolist()
        #        allowed_weights_full.append(allowed_weights)
        #        reg_sigs.append(quant_step)

        #allowed_weights_full = np.array(allowed_weights_full)
        #allowed_weights_full = torch.from_numpy(allowed_weights_full).to(device)
        #reg_sigs = np.array(reg_sigs)

        if epoch<stop_epoch:
            reg_lambda = reg_lambda*lambda_factor
            reg_sigs = reg_sigs/sig_factor
        print('reg_lambda: ', reg_lambda)
        model.train()
        act_scale_factors = []
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            #Compute activation quantization parameters
            if i==0:
                activations, out = model.forward_act(data)
                q_max = 2**act_bits - 1
                for act in activations:
                    act_np = act.data.cpu().numpy()
                    act_nz_np = act_np[act_np>0]
                    scale_factor = 2*np.mean(act_nz_np)/np.sqrt(q_max)
                    act_scale_factors.append(scale_factor)
            
            spk_rec = model(data) # forward-pass
            #spk_rec = model.forward_act_quant(data, act_scale_factors, act_bits) # forward-pass
            loss_val = loss_fn(spk_rec, targets) # loss calculation
            if quant_aware:
                reg_loss = 0
                layer_ind = 0
                for (names, params) in model.named_parameters():
                    if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
                        allowed_weights = allowed_weights_full[layer_ind]
                        reg_sig = reg_sigs[layer_ind]
                        for j in range(allowed_weights.shape[0]):
                            reg_loss = reg_loss - reg_lambda*torch.sum(torch.exp(-((params - allowed_weights[j])**2)/(reg_sig**2)))
                        layer_ind = layer_ind + 1
                loss_val = loss_val + reg_loss
            optimizer.zero_grad() # null gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights
            #clamp weights
            layer_ind = 0
            for (names, params) in model.named_parameters():
                if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
                    allowed_weights = allowed_weights_full[layer_ind]
                    params.data.clamp_(allowed_weights[0], allowed_weights[-1])
                    layer_ind = layer_ind + 1
        print("Epoch: ", epoch)
        #if args.nn_model == 'vgg13':
        #    scheduler.step()
        acc = test_accuracy(test_loader, model)
        print("test accuracy(%): ", acc * 100)
        if (acc*100) > max_non_quant_acc:
            max_non_quant_acc = acc*100
        
        model.eval()
        model_test.eval()
        
        if (epoch%2 == 1) or (epoch>num_epochs-5):
            #act_scale_factors = []
            #q_max = 2**act_bits - 1
            with torch.no_grad():
                model_test.load_state_dict(model.state_dict())
                #for i, (data, targets) in enumerate(iter(train_loader)):
                #    if i==0:
                #        data = data.to(device)
                #        activations, out = model.forward_act(data)
                #        break
                #for act in activations:
                #    act_np = act.data.cpu().numpy()
                #    act_nz_np = act_np[act_np>0]
                #    scale_factor = 2*np.mean(act_nz_np)/np.sqrt(q_max)
                #    act_scale_factors.append(scale_factor)
                param_cnt = 0
                for (names, params) in model_test.named_parameters():
                    if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
                        allowed_weights = allowed_weights_full[param_cnt]
                        allowed_weights_np = allowed_weights.cpu().numpy()
                        quant_step = allowed_weights_np[1] - allowed_weights_np[0]
                        if quant_aware and Wt_bits<5:
                            print('layer: ', names)
                            print('allowed_weights: ', allowed_weights_np)
                        params.data = torch.clamp(params.data, allowed_weights[0], allowed_weights[-1])
                        param_np = params.data.cpu().numpy()
                        params_quant = np.int32(np.zeros_like(param_np))
                        #quantize weights to fixed-point precision
                        for k in range(allowed_weights_np.shape[0]):
                            params_quant[(param_np<(allowed_weights_np[k]+(quant_step/2))) & (param_np>=(allowed_weights_np[k]-(quant_step/2)))] = k
                        params_fixed_fl_var = np.float32(np.zeros_like(param_np))
                        #introduce non-idealities
                        for bit in range(Wt_bits):
                            params_bit = (params_quant % 2)
                            params_fixed_fl_var = params_fixed_fl_var + np.float32((2**bit)*(W_vars_1[param_cnt][bit]*params_bit + W_vars_0[param_cnt][bit]*(1 - params_bit)))
                            params_quant = params_quant//2
                        #scale weights back to (-wt_limit, wt_limit) range, but with Fixed-point precision
                        params_q = np.float32((params_fixed_fl_var - zero_pt)*quant_step)
                        params_q = torch.from_numpy(params_q).to(device)
                        params.data = torch.where(params.data>=allowed_weights[0], params_q, params.data)
                        param_cnt = param_cnt + 1
                #Test using the copy network for testing
                post_q_acc = test_accuracy_w_act_quant(test_loader, model_test, act_scale_factors, act_bits)

                print("Post-quantization test accuracy(%): ", post_q_acc * 100)
                all_accs.append(post_q_acc)
                record_epochs.append(epoch)
                #model.load_state_dict(model_test.state_dict())
                #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    run_end = time.time()
    print("total runtime: ", run_end - run_start, "seconds")

    print('mu_HRS: ', mu_HRS)
    print('mu_LRS: ', mu_LRS)
    print('sig_by_mu_HRS: ', sig_by_mu_HRS)
    print('sig_by_mu_LRS: ', sig_by_mu_LRS)

    if not quant_aware:
        print('max fp32 test accuracy: ', max_non_quant_acc)

    max_test_acc = np.max(np.array(all_accs))
    print('max test accuracy: ', max_test_acc * 100)
    print('record epochs: ', record_epochs)
    print('record post quantization test accuracies: ', all_accs)

    sv_dict = {"record_epochs": record_epochs, "all_accs": all_accs}
    #savemat(save_folder + "CIFAR10_training_quant_aware_4_bit_activations_" + args.nn_model + str(Wt_bits) + "bit.mat", sv_dict)
