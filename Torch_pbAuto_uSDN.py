from random import uniform
import torch
import torch.nn as nn
import os
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import tensorflow as tf
from os.path import join as pjoin
torch.set_printoptions(precision=10)

###LRHSI UNIFORM
class LR_HSI_Uniform(nn.Module ):
    def __init__(self, dimLR, srf, nLRlevel, num):
        super(LR_HSI_Uniform, self).__init__()
        self.srf = srf
        self.nLRlevel = nLRlevel
        self.num = num
        self.layer_11 = nn.Linear(self.srf.shape[0], self.nLRlevel[0])
        self.layer_12 = nn.Linear(self.layer_11.out_features + dimLR[-1], self.nLRlevel[1])
        self.layer_13 = nn.Linear(dimLR[-1] + self.nLRlevel[1] + self.nLRlevel[0], self.nLRlevel[2])
        self.layer_14 = nn.Linear(dimLR[-1] + self.nLRlevel[2] + self.nLRlevel[1] + self.nLRlevel[0], self.nLRlevel[3])
        self.uniform  = nn.Linear(dimLR[-1] + self.nLRlevel[3] + self.nLRlevel[2]+ self.nLRlevel[1] + self.nLRlevel[0], self.num)

    def forward(self, x):
        # x = x.type(torch.DoubleTensor)
        x = x.float()
        layer_1 = torch.mm(x, self.srf.T).float();        
        layer_11 = self.layer_11(layer_1)
        stack_layer_11 = torch.cat((x,layer_11), dim = 1).float(); #None * 34
        layer_12 = self.layer_12(stack_layer_11) #None * 3
        stack_layer_12 = torch.cat((stack_layer_11, layer_12), dim = 1).float(); # None * 37
        layer_13 = self.layer_13(stack_layer_12)
        stack_layer_13 = torch.cat((stack_layer_12, layer_13), dim = 1).float(); # None * 40
        layer_14 = self.layer_14(stack_layer_13)
        stack_layer_14 = torch.cat((stack_layer_13, layer_14), dim = 1).float(); # None * 43
        uniform = self.uniform(stack_layer_14) #None * 12
        return layer_1, uniform
    


##LRHSI BETA
class LR_HSI_BETA(nn.Module):
    def __init__(self, srf, nLRlevel):
        super().__init__()
        self.srf = srf
        self.nLRlevel = nLRlevel
        self.layer_21 = nn.Linear(self.srf.shape[0], self.nLRlevel[0])
        self.layer_22 = nn.Linear(self.srf.shape[0] + self.nLRlevel[0], self.nLRlevel[1])
        self.layer_32 = nn.Linear(self.srf.shape[0] + self.nLRlevel[1] + self.nLRlevel[0], 1)
        

    def forward(self, x):
        layer_21 = self.layer_21(x) # None * 3
        stack_layer_21 = torch.cat((x,layer_21), dim = 1)
        layer_22 = self.layer_22(stack_layer_21)
        stack_layer_22 = torch.cat((layer_22, stack_layer_21), dim = 1)
        layer_32 = self.layer_32(stack_layer_22)
        return layer_32
        


##HR MSI UNIFORM
class Encoder_Uniform_MSI(nn.Module):
    def __init__(self, dimHR, nLRlevel, num ):
        super().__init__()
        self.nLRlevel = nLRlevel
        self.num = num
        self.dimHR = dimHR
        self.layer_11 = nn.Linear(self.dimHR[-1] ,self.nLRlevel[0]) # None * 3
        self.layer_12 = nn.Linear(self.dimHR[-1] + self.nLRlevel[0], self.nLRlevel[1]) #      
        self.layer_13 = nn.Linear(self.dimHR[-1] + self.nLRlevel[1] + self.nLRlevel[0], self.nLRlevel[2]) #
        self.layer_14 = nn.Linear(self.dimHR[-1] + self.nLRlevel[2] + self.nLRlevel[1] + self.nLRlevel[0], self.nLRlevel[3]) #
        self.uniform  = nn.Linear(self.dimHR[-1] + self.nLRlevel[3]  + self.nLRlevel[2] + self.nLRlevel[1] + self.nLRlevel[0], self.num)
        

    def forward(self, x):
        x = x.float()
        layer_11 = self.layer_11(x)
        # print("Layer_11", layer_11)
        stack_layer_11 = torch.cat((x, layer_11), dim = 1).float()
        layer_12 = self.layer_12(stack_layer_11)
        stack_layer_12 = torch.cat((stack_layer_11, layer_12), dim = 1).float()
        layer_13 = self.layer_13(stack_layer_12)
        stack_layer_13 = torch.cat((stack_layer_12, layer_13), dim = 1).float()
        layer_14 = self.layer_14(stack_layer_13)
        stack_layer_14 = torch.concat((stack_layer_13, layer_14), dim = 1).float()
        uniform = self.uniform(stack_layer_14)
        # print("Uniform", uniform)
        return stack_layer_12, uniform


##HR  MSI BETA
class Encoder_Beta_MSI(nn.Module):
    def __init__(self, dimHR,nlRlevel):
        super().__init__()
        self.nLRlevel = nlRlevel
        self.dimHR = dimHR
        self.layer_21 = nn.Linear(dimHR[-1], self.nLRlevel[0])
        self.layer_22 = nn.Linear(dimHR[-1] + self.nLRlevel[0], self.nLRlevel[1])
        self.layer_32 = nn.Linear(dimHR[-1] + self.nLRlevel[1]+ self.nLRlevel[0], 1)    
    
    def forward(self, x):
        x = x.float()
        layer_21 = self.layer_21(x)
        stack_layer_21 = torch.cat((x, layer_21), dim = 1)
        layer_22 = self.layer_22(stack_layer_21)
        stack_layer_22 = torch.cat((layer_22, stack_layer_21), dim = 1)
        layer_32 = self.layer_32(stack_layer_22)
        return layer_32




class Decoder_Hsi(nn.Module):
    def __init__(self, num, dimlr):
        super().__init__()
        self.num = num
        self.dimlr = dimlr
        self.lr_decoder_w1  = nn.Parameter(Variable(torch.normal(0, 0.1, size=(self.num, self.num))))
        self.lr_decoder_w2  = nn.Parameter(Variable(torch.normal(0, 0.1, size=(self.num, self.dimlr[2]))))

    def forward(self, x):
        layer_1 = torch.mm(x, self.lr_decoder_w1)
        layer_2 = torch.mm(layer_1, self.lr_decoder_w2)
        return layer_2

    def decoder_op(self):
        return torch.mm(self.lr_decoder_w1, self.lr_decoder_w2)

class betapan(object):
    def __init__(self, input, lr_rate, p_rate, nLRlevel, nHRlevel, epoch, is_adam,
                 vol_r, sp_r_lsi, sp_r_msi, initp, config):
        ##setting all variables
        self.input = input
        self.initlrate = lr_rate
        self.initprate = p_rate
        self.epoch = epoch
        self.nLRlevel = nLRlevel
        self.nHRlevel = nHRlevel
        self.num = input.num
        self.is_adam = is_adam
        self.vol_r = vol_r
        self.sp_r_lsi = sp_r_lsi
        self.sp_r_msi = sp_r_msi
        self.mean_lrhsi = torch.tensor(input.mean_lr_hsi)
        self.mean_hrmsi = torch.tensor(input.mean_hr_msi)
        self.dimlr = input.dimLR
        self.dimhr = input.dimHR
        self.input_lr_hsi = input.rcol_lr_hsi
        self.input_hr_msi = input.rcol_hr_msi
        self.input_lr_msi = input.rcol_lr_msi
        self.input_hr_msi_h = np.zeros([input.dimLR[0]*input.dimLR[1],input.num])
        self.initp = initp
        self.srf = (torch.tensor(self.input.srf)).float()
        self.encoder_uniform_hsi = LR_HSI_Uniform(self.dimlr, self.srf, self.nLRlevel, self.num);
        self.encoder_beta_hsi    = LR_HSI_BETA(self.srf, self.nLRlevel);
        self.encoder_uniform_msi = Encoder_Uniform_MSI(self.dimhr, self.nLRlevel, self.num);
        self.encoder_beta_msi    = Encoder_Beta_MSI(self.dimhr,self.nLRlevel);
        self.decoder_hsi         = Decoder_Hsi(self.num, self.dimlr);
        params   = list(self.decoder_hsi.parameters()) + list(self.encoder_uniform_hsi.parameters()) + list(self.encoder_beta_hsi.parameters())
        params_1 = list(self.encoder_uniform_msi.parameters()) + list(self.encoder_beta_msi.parameters())
        self.opt_lrhsi = optim.Adam(params, lr=self.initlrate)
        self.opt_init  = optim.Adam(params_1, lr=self.initlrate)
        self.opt_angle = optim.Adam(params_1, lr=self.initlrate)
        self.opt_hrmsi = optim.Adam(params_1, lr=self.initlrate)

    
    def load_params_fromTF(self, path):
        # Retrieve weights from TF checkpoint
        tf_path = os.path.abspath(path)
        init_vars = tf.train.list_variables(tf_path)
        tf_vars = []
        for name, shape in init_vars:
            # print("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            tf_vars.append((name, array.squeeze()))
        
        # Code for loading decoders
        decoders = []
        for name, value in tf_vars:
            if 'lr_decoder' in name and 'ptim' not in name:
                # print(name, value, value.shape)
                decoders.append(torch.from_numpy(value))
        sd = self.decoder_hsi.state_dict()
        count = 0
        print("Decoders are : ", decoders)
        for i in sd:
            sd[i] = decoders[count]
            count += 1
        self.decoder_hsi.load_state_dict(sd)
        print("Decoders loaded Succesfully")
        print("After Loading Decoders : ", self.decoder_hsi.state_dict())
        ##Code for loading weights of hr_msi_uniform
        biases = []
        weights = []
        for name, value in tf_vars:
            if 'hr_msi_uniform' in name and 'ptim' not in name:
                if 'biases' in name:
                    biases.append(torch.from_numpy(value))
                else:
                    weights.append(torch.from_numpy(value.T))

        sd = self.encoder_uniform_msi.state_dict()
        weight_count = 0
        bias_count   = 0
        for i in sd:
            if 'weight' in i:
                sd[i] = weights[weight_count]
                weight_count+=1
            elif 'bias' in i:
                sd[i] = biases[bias_count]
                bias_count+=1
        self.encoder_uniform_msi.load_state_dict(sd)
        print("Loading of encoder msi uniform weights done")


        #Code for loading weights of HR MSI BETA
        biases = []
        weights = []
        for name, value in tf_vars:
            if 'hr_msi_beta' in name and 'ptim' not in name:
                # print(name, value)
                if 'biases' in name:
                    if value.shape == ():
                        biases.append(torch.from_numpy(np.array([value])))
                    else:
                        biases.append(torch.from_numpy(value))
                else:
                    if value.shape == (9,):
                        weights.append(torch.from_numpy(np.array([value.T])))
                    else:
                        weights.append(torch.from_numpy(value.T))

        sd = self.encoder_beta_msi.state_dict()
        weight_count = 0
        bias_count   = 0
        for i in sd:
            if 'weight' in i:
                sd[i] = weights[weight_count]
                weight_count+=1
            elif 'bias' in i:
                sd[i] = biases[bias_count]
                bias_count+=1

        self.encoder_beta_msi.load_state_dict(sd)
        print("Loading of weights of msi beta done")

        #Code for loading weights of LR HSI uniform

        biases = []
        weights = []
        for name, value in tf_vars:
            if 'lr_hsi_uniform' in name and 'ptim' not in name:
                # print(name, value)
                if 'biases' in name:
                    biases.append(torch.from_numpy(value))
                else:
                    weights.append(torch.from_numpy(value.T))


        sd = self.encoder_uniform_hsi.state_dict()
        weight_count = 0
        bias_count   = 0
        for i in sd:
            if 'weight' in i:
                sd[i] = weights[weight_count]
                weight_count+=1
            elif 'bias' in i:
                sd[i] = biases[bias_count]
                bias_count+=1

        self.encoder_uniform_hsi.load_state_dict(sd)
        print("Loading of Variables to LR HSI UNIFORM DONE")


        #Code for loading weights of LR Beta Uniform
        biases = []
        weights = []
        for name, value in tf_vars:
            if 'lr_hsi_beta' in name and 'ptim' not in name:
                # print(name, value)
                if 'biases' in name:
                    if value.shape == ():
                        biases.append(torch.from_numpy(np.array([value])))
                    else:
                        biases.append(torch.from_numpy(value))
                else:
                    if value.shape == (9,):
                        weights.append(torch.from_numpy(np.array([value.T])))
                    else:
                        weights.append(torch.from_numpy(value.T))

        sd = self.encoder_beta_hsi.state_dict()
        weight_count = 0
        bias_count   = 0
        for i in sd:
            if 'weight' in i:
                sd[i] = weights[weight_count]
                weight_count+=1
            elif 'bias' in i:
                sd[i] = biases[bias_count]
                bias_count+=1

        self.encoder_beta_hsi.load_state_dict(sd)
        print("Loading of Variables to LR HSI Beta DONE")
        print(self.encoder_uniform_hsi .state_dict())
        print(self.encoder_beta_hsi    .state_dict())
        print(self.encoder_uniform_msi .state_dict())
        print(self.encoder_beta_msi    .state_dict())
        print(self.decoder_hsi         .state_dict())
        
    
    
    def decoder_msi(self,x):
        ## look for weight retrievals later on
        layer_2 = self.decoder_hsi(x)
        layer_3 = torch.add(layer_2,torch.from_numpy(self.input.mean_lr_hsi))
        return layer_3

    def compute_latent_vars_break(self, i, remaining_stick, v_samples):
        # compute stick segment
        stick_segment = v_samples[:, i] * remaining_stick
        
        remaining_stick = remaining_stick * (1 - v_samples[:, i]) ##//inplace operation causing problem
        return (stick_segment, remaining_stick)


    def construct_stick_break(self,vsample, dim, stick_size):
        size = dim[0]*dim[1]
        size = int(size)
        remaining_stick = torch.ones(size, )
        for i in range(stick_size):
            [stick_segment, remaining_stick] = self.compute_latent_vars_break(i, remaining_stick, vsample)
            if i == 0:
                stick_segment_sum_lr = torch.unsqueeze(stick_segment, 1)
            else:
                stick_segment_sum_lr = torch.cat((stick_segment_sum_lr, torch.unsqueeze(stick_segment, 1)),1)
        return stick_segment_sum_lr
    

    def construct_vsamples(self,uniform,wb,hsize):
        concat_wb = wb
        for iter in range(hsize - 1):
            concat_wb = torch.cat([concat_wb, wb], dim = 1)
        v_samples = uniform ** (1.0 / concat_wb)
        return v_samples



    def encoder_vsamples_msi(self, x, hsize):
        stack_layer_12, uniform = self.encoder_uniform_msi(x)
        sigm = nn.Sigmoid()
        uniform = sigm(uniform)
        wb = self.encoder_beta_msi(x)
        # print("Encoder_beta", wb)
        softplus = nn.Softplus()
        wb = softplus(wb)
        v_samples = self.construct_vsamples(uniform,wb,hsize)
        # print("Vsamples", v_samples)
        # print("Uniform_after_activ", uniform)
        # print("Wb_after_activ", wb)
        return v_samples, uniform, wb


    
    def encoder_hr_msi(self, x):
            v_samples,v_uniform, v_beta = self.encoder_vsamples_msi(x, self.num)
            stick_segment_sum_msi = self.construct_stick_break(v_samples, self.dimhr, self.num)
            # print("Stick", stick_segment_sum_msi)
            return stick_segment_sum_msi
    
    
    def gen_hrhsi(self, x):
        # print("Input is : ", x)
        encoder_op = self.encoder_hr_msi(x)
        # print("Encoder_Op" , encoder_op)
        decoder_hr = self.decoder_msi(encoder_op)
        return decoder_hr

        
    
    def encoder_vsamples_hsi(self, x, hsize):
        layer1, uniform = self.encoder_uniform_hsi(x)
        sigm = nn.Sigmoid()
        uniform = sigm(uniform)
        wb = self.encoder_beta_hsi(layer1)
        softplus = nn.Softplus()
        wb = softplus(wb)
        v_samples = self.construct_vsamples(uniform,wb,hsize)
        return v_samples, uniform, wb
    
    
    def encoder_lr_hsi(self, x):
        v_samples,uniform, wb = self.encoder_vsamples_hsi(x, self.num)
        stick_segment_sum_lr = self.construct_stick_break(v_samples, self.dimlr, self.num)
        return stick_segment_sum_lr
    
    def gen_lrhsi(self, x):
        encoder_op = self.encoder_lr_hsi(x)
        decoder_op = self.decoder_hsi(encoder_op)
        return decoder_op



    def gen_hrmsi(self, x):
        encoder_op = self.encoder_hr_msi(x)
        decoder_hr = self.decoder_msi(encoder_op)
        decoder_op = torch.mm(decoder_hr,self.srf.T)
        decoder_plus_m = torch.add(decoder_op, -self.mean_hrmsi)
        # decoder_sphere = tf.matmul(decoder_plus_m,self.input.invsig_msi)
        return decoder_plus_m


    def compute_lrhsi_volume_loss(self):
        decoder_op = self.decoder_hsi.decoder_op();
        decoder = torch.add(decoder_op, torch.tensor(self.input.mean_lr_hsi))
        lrhsi_volume_loss = torch.mean(torch.mm(torch.t(decoder),decoder))
        return lrhsi_volume_loss
    
    
    def compute_lrhsi_loss_sparse(self, lrhsi_top, eps = 0.00000001):
        lrhsi_base_norm = torch.sum(lrhsi_top, 1, keepdims=True)
        lrhsi_sparse = torch.div(lrhsi_top, (lrhsi_base_norm + eps))
        lrhsi_loss_sparse = torch.mean(-torch.multiply(lrhsi_sparse, torch.log(lrhsi_sparse + eps)))
        return lrhsi_loss_sparse
        

    def compute_lrhsi_loss(self, y_pred_lrhsi, y_true_lrhsi, lrhsi_top):
        error_lrhsi = torch.sub(y_pred_lrhsi , y_true_lrhsi)
        lrhsi_loss_euc = torch.mean(torch.sum(torch.pow(error_lrhsi, 2),0))
        
        lrhsi_volume_loss = self.compute_lrhsi_volume_loss()

        # spatial sparse
        eps = 0.00000001
        lrhsi_loss_sparse = self.compute_lrhsi_loss_sparse(lrhsi_top, eps)
        # print(lrhsi_loss_sparse.shape)
        lrhsi_loss = lrhsi_loss_euc.clone() + self.vol_r * lrhsi_volume_loss.clone() + self.sp_r_lsi * lrhsi_loss_sparse.clone()
        
        return lrhsi_loss 

    def compute_hrmsi_loss_sparse(self, hrmsi_top):
        eps = 0.00000001
        hrmsi_base_norm = torch.sum(hrmsi_top, 1, keepdims=True)
        hrmsi_sparse = torch.div(hrmsi_top, (hrmsi_base_norm + eps))
        hrmsi_loss_sparse = torch.mean(-torch.multiply(hrmsi_sparse, torch.log(hrmsi_sparse + eps)))
        return hrmsi_loss_sparse

    
    def compute_hrmsi_loss(self, y_pred_hrmsi, y_true_hrmsi,hrmsi_top):
        eps = 0.00000001
        error_hrmsi = torch.sub(y_pred_hrmsi,  y_true_hrmsi)
        hrmsi_loss_euc = torch.mean(torch.sum(torch.pow(error_hrmsi, 2), 0))
        hrmsi_loss_sparse = self.compute_hrmsi_loss_sparse(hrmsi_top)
        hrmsi_loss = hrmsi_loss_euc + self.sp_r_lsi * hrmsi_loss_sparse
        return hrmsi_loss
        
    
    def compute_msih_init_loss(self, msi_h, truth):
        error_init = torch.sub(msi_h, truth)
        msih_init_loss = torch.mean(torch.pow(error_init, 2))
        return msih_init_loss


    def compute_angle_loss(self, msi_h):
        eps = 0.00000001
        nom_pred = torch.sum(torch.pow(msi_h, 2),0)
        nom_true = torch.sum(torch.pow(torch.tensor(self.input_hr_msi_h), 2),0)
        nom_base = torch.sqrt(torch.multiply(nom_pred, nom_true))
        nom_top  = torch.sum(torch.multiply(msi_h,torch.tensor(self.input_hr_msi_h)),0)
        angle = torch.mean(torch.acos(torch.div(nom_top, (nom_base + eps))))
        angle_loss = torch.div(angle,3.1416)  # spectral loss
        return angle_loss       


    def encoder_hr_msi_init(self, x):
        v_samples,v_uniform, v_beta = self.encoder_vsamples_msi(x, self.num)
        stick_segment_sum_msi_init = self.construct_stick_break(v_samples, self.dimlr, self.num)
        return stick_segment_sum_msi_init


    def gen_msi_h(self, x):
        encoder_init = self.encoder_hr_msi_init(x)
        return encoder_init

    
    
    def train(self, load_Path, save_dir, loadLRonly, tol,init_num):
        sam_hr = 10
        sam_lr = 10
        rate_decay = 0.99977
        count = 0
        stop_cont = 0
        sam_total = np.zeros(self.epoch+1)
        rmse_total = np.zeros(self.epoch+1)
        sam_total[0] = 50
        rmse_total[0] = 50
        results_file_name = pjoin(save_dir,"sb_" + "lrate_" + str(self.initlrate)+ "_Train_Torch_.txt")
        # results_ckpt_name = pjoin(save_dir,"sb_" + "lrate_" + str(self.initlrate)+ ".ckpt")
        results_file = open(results_file_name, 'a')
 

        for epoch in range(self.epoch):
            if sam_lr > tol:
            ##LRHSI_LOSS
                self.opt_lrhsi.zero_grad()

                y_pred_lrhsi = self.gen_lrhsi(torch.tensor(self.input_lr_hsi))
                lrhsi_top = self.encoder_lr_hsi(torch.tensor (self.input_lr_hsi))
                y_true_lrhsi = torch.tensor(self.input_lr_hsi)
                lrhsi_loss = self.compute_lrhsi_loss(y_pred_lrhsi, y_true_lrhsi, lrhsi_top)
                # print(lrhsi_loss)
                self.initlrate = self.initlrate * rate_decay
                self.vol_r = self.vol_r * rate_decay
                self.sp_r_lsi = self.vol_r * rate_decay

            ##Optimize LRHSI_LOSS
                # self.opt_lrhsi.zero_grad();
                torch.autograd.set_detect_anomaly(True)
                lrhsi_loss.backward();
                self.opt_lrhsi.step();
                
            if (epoch + 1) % 50 == 0:
                # Report and save progress.
                print("Report:")
                results = "epoch {}: LR HSI loss {:.12f} learing_rate {:.9f}"
                # print(type(lr_loss))
                results = results.format(epoch, lrhsi_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n")
                results_file.flush()

                ##HRMSI Loss
                y_pred_hrmsi = self.gen_hrmsi(torch.tensor(self.input_hr_msi))
                y_true_hrmsi = torch.tensor(self.input_hr_msi)
                hrmsi_top = self.encoder_hr_msi(torch.tensor(self.input_hr_msi))
                hrmsi_loss = self.compute_hrmsi_loss(y_pred_hrmsi, y_true_hrmsi, hrmsi_top);
                
                results = "epoch {}: HR MSI loss {:.12f} learing_rate {:.9f}"
                # print(type(msi_loss[0]));
                results = results.format(epoch, hrmsi_loss.item(), self.initprate)
                print (results)
                print ("\n")
                results_file.write(results + "\n")
                results_file.flush()

                v_loss = self.compute_lrhsi_volume_loss()
                results = "epoch {}: volumn loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, v_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n")
                results_file.flush()

                lrhsi_top = self.encoder_lr_hsi(torch.tensor(self.input_lr_hsi))
                sp_hsi_loss = self.compute_lrhsi_loss_sparse(lrhsi_top)
                results = "epoch {}: lr sparse loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, sp_hsi_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()


                img_lr = self.gen_lrhsi(torch.tensor(self.input_lr_hsi)) + self.mean_lrhsi
                rmse_lr, sam_lr = self.evaluation(img_lr,self.input.col_lr_hsi,'LR HSi',epoch,results_file)

            
            if sam_lr <= tol:
                if count == 0:
                    self.input_hr_msi_h = self.encoder_lr_hsi(torch.tensor(self.input_lr_hsi))
                
                if self.initp == True:
                    while self.initp and count < init_num:
                        ##Abundance init
                        msi_h = self.gen_msi_h(torch.tensor(self.input_lr_msi))
                        initloss = self.compute_msih_init_loss(msi_h, torch.tensor(self.input_hr_msi_h));
                        
                        initpanloss = "epoch {}: initloss of the msi: {:.9f}"
                        initpanloss = initpanloss.format(count,initloss)
                        print (initpanloss)
                        results_file.write(initpanloss + "\n")
                        results_file.flush()
                        
                        self.opt_init.zero_grad();
                        initloss.backward();
                        self.opt_init.step();

                        count = count + 1
                        # if (count) % 1000 == 0:
                        #     saver = tf.train.Saver()
                        if initloss<0.00001:
                            self.initp = False


                y_pred_hrmsi = self.gen_hrmsi(torch.tensor(self.input_hr_msi))
                y_true_hrmsi = torch.tensor(self.input_hr_msi)
                hrmsi_top = self.encoder_hr_msi(torch.tensor(self.input_hr_msi))
                hrmsi_loss = self.compute_hrmsi_loss(y_pred_hrmsi, y_true_hrmsi, hrmsi_top);
                
                self.opt_hrmsi.zero_grad();
                hrmsi_loss.backward();
                self.opt_hrmsi.step();

                self.initprate = self.initprate * rate_decay
                self.sp_r_msi = self.sp_r_msi * rate_decay

                
                if (epoch + 1) % 20 == 0:
                    # Report and save progress.
                    results = "epoch {}: HR MSI loss {:.12f} learing_rate {:.9f}"
                    results = results.format(epoch, hrmsi_loss.item(), self.initprate)
                    print(results)
                    print("\n")
                    results_file.write(results + "\n\n")
                    results_file.flush()

                    hrmsi_top = self.encoder_hr_msi(torch.tensor(self.input_hr_msi))
                    sp_msi_loss = self.compute_hrmsi_loss_sparse(hrmsi_top)
                    results = "epoch {}: hr sparse loss {:.12f} learing_rate {:.9f}"
                    results = results.format(epoch, sp_msi_loss, self.initprate)
                    print(results)
                    print("\n")
                    results_file.write(results + "\n\n")
                    results_file.flush()

                    
                    ##Spectral Loss
                    msi_h = self.gen_msi_h(torch.tensor(self.input_lr_msi))
                    angle_loss = self.compute_angle_loss(msi_h);

                    angle = "Angle of the pan: {:.12f}"
                    angle = angle.format(angle_loss)
                    print(angle)
                    results_file.write(angle + "\n")
                    results_file.flush()

                    self.opt_angle.zero_grad();
                    angle_loss.backward();
                    self.opt_angle.step();

                    
                    # img_hr = self.sess.run(self.gen_hrmsi(self.hr_msi, reuse=True), feed_dict=feed_dict) + self.mean_hrmsi
                    # sam_hr = self.evaluation(img_hr,self.input.col_hr_msi,'HR MSI',epoch,results_file)
                    
                    img_hr = self.gen_hrhsi(torch.tensor(self.input_hr_msi))
                    rmse_hr, sam_hr = self.evaluation(img_hr,self.input.col_hr_hsi,'HR MSI',epoch,results_file)
                    stop_cont = stop_cont + 1
                    sam_total[stop_cont] = sam_hr
                    rmse_total[stop_cont] = rmse_hr
                    
                    if ((sam_total[stop_cont-1] / sam_total[stop_cont]) < 1 - 0.0001 and (rmse_total[stop_cont-1]/rmse_total[stop_cont]<1 - 0.0001)):
                        results_ckpt_name = pjoin(save_dir,"epoch_" + str(epoch) + "_sam_" + str(round(sam_hr, 3)) + ".ckpt")
                        # save_path = saver.save(self.sess, results_ckpt_name)
                        print('training is done')
                        break;

        # return save_path
        return 


            
    
    def generate_hrhsi(self, save_dir, others = False):
        # self.sess.run(tf.global_variables_initializer())
        hrhsi = self.gen_hrhsi(torch.from_numpy(self.input_hr_msi).float())
        hrhsi = hrhsi.detach().numpy()
        np.savetxt(save_dir + "/Torch_Vers.csv", hrhsi, delimiter=",")
        if others:
            hrhsi = self.gen_hrhsi(torch.from_numpy(self.input.rcol_hr_msi_01).float())
            hrhsi = hrhsi.detach().numpy()
            np.savetxt(save_dir + "/Torch_Vers_02.csv", hrhsi, delimiter=",")
        
            hrhsi = self.gen_hrhsi(torch.from_numpy(self.input.rcol_hr_msi_02).float())
            hrhsi = hrhsi.detach().numpy()
            np.savetxt(save_dir + "/Torch_Vers_03.csv", hrhsi, delimiter=",")
            
            hrhsi = self.gen_hrhsi(torch.from_numpy(self.input.rcol_hr_msi_03).float())
            hrhsi = hrhsi.detach().numpy()
            np.savetxt(save_dir + "/Torch_Vers_04.csv", hrhsi, delimiter=",")
            


    def evaluation(self,img_hr,img_tar,name,epoch,results_file):
        # evalute the results
        ref = img_tar*255.0
        tar = img_hr.cpu().detach().numpy()*255.0
        lr_flags = tar<0
        tar[lr_flags]=0
        hr_flags = tar>255.0
        tar[hr_flags] = 255.0

        #ref = ref.astype(np.int)
        #tar = tar.astype(np.int)

        diff = ref - tar;
        size = ref.shape
        rmse = np.sqrt( np.sum(np.sum(np.power(diff,2))) / (size[0]*size[1]));
        # rmse_list.append(rmse)
        # print('epoch '+str(epoch)+' '+'The RMSE of the ' + name + ' is: '+ str(rmse))
        results = name + " epoch {}: RMSE  {:.12f} "
        results = results.format(epoch,  rmse)
        print (results)
        results_file.write(results + "\n")
        results_file.flush()

        # spectral loss
        nom_top = np.sum(np.multiply(ref, tar),0)
        nom_pred = np.sqrt(np.sum(np.power(ref, 2),0))
        nom_true = np.sqrt(np.sum(np.power(tar, 2),0))
        nom_base = np.multiply(nom_pred, nom_true)
        angle = np.arccos(np.divide(nom_top, (nom_base)))
        angle = np.nan_to_num(angle)
        sam = np.mean(angle)*180.0/3.14159
        # sam_list.append(sam)
        # print('epoch '+str(epoch)+' '+'The SAM of the ' + name + ' is: '+ str(sam)+'\n')
        results = name + " epoch {}: SAM  {:.12f} "
        results = results.format(epoch,  sam)
        print (results)
        print ("\n")
        results_file.write(results + "\n")
        results_file.flush()
        return rmse, sam





srf = np.array([[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
                0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
                0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
                0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]], dtype = np.float32)

# new = LR_HSI_Uniform((2,2,31), srf, [3,3,3,3], 12);
# # new = Decoder_Hsi(12, (2,2,31))
# 
# 
# 
# new = Encoder_Beta_MSI((2,2,3),[3,3,3,3])

# sd = new.state_dict()
# for i in sd:
#     print(i, sd[i].shape)

# # new = Encoder_Uniform_MSI((2,2,3), [3,3,3,3], 12)
# # sd = new.state_dict()

# new = LR_HSI_BETA(srf, [3,3,3,3]);
# sd = new.state_dict()
# for i in sd:
#     print(i)

# # print(new.get_parameter())
# # print(new.parameters)
# # for i in new.parameters():
# #     print(i.shape)

# # for i,_ in new.named_parameters():
# #     print(i)
# # print("Done")