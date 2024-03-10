from model import RNNModelScratch, BNN_User, convert_model_to_double, rnn, ResBNN_User
from Selected_Data import Customized_LIFE_BusDataset
import json
import argparse
import math
import model
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from DHLOSS import total_loss


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(out_features, out_features)

        # Adjust dimensions if the input size changes
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
            )

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # Apply the residual connection if downsample exists
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out


def split_and_normalize_number(num):
    if num < 0 or num > 999:
        raise ValueError("输入的数必须在0到999之间")

    # 获取百、十、个位数
    hundreds = num // 100
    tens = (num % 100) // 10
    units = num % 10

    # 创建1x3维张量并除以十
    tensor = torch.DoubleTensor([hundreds / 10, tens / 10, units / 10])

    return tensor


def reverse_transform(tensor):
    # 确保输入是一个1x3维张量
    # if tensor.shape != (3):
    #     raise ValueError("Input tensor must be a 1x3 tensor.")
    #     print(tensor)

    # 将张量的每个元素乘以十，取最接近的整数，再相加
    number = torch.round(tensor[0] * 10) * 100 + torch.round(tensor[1] * 10) * 10 + torch.round(tensor[2] * 10)

    return int(number)


torch.autograd.set_detect_anomaly(True)


class BUS_AutoEncoder(nn.Module):
    def __init__(self):
        super(BUS_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=16, out_features=8),
            nn.Tanh(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=8, out_features=16),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=19, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=6),
            nn.Tanh()

        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=6, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=19),

            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


class BPNET(nn.Module):
    def __init__(self):
        super(BPNET, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=27, out_features=48),
            nn.Tanh(),
            nn.Linear(in_features=48, out_features=72),
            nn.Tanh(),
            nn.Linear(in_features=72, out_features=160),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=160, out_features=100),
            nn.Tanh(),
            nn.Linear(in_features=100, out_features=60),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out


class BPNET_Life_time(nn.Module):
    def __init__(self):
        super(BPNET_Life_time, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=27, out_features=500),
            nn.ReLU(),
            # nn.Linear(in_features=1000, out_features=5000),
            # nn.Tanh(),

        )
        self.neck = nn.Sequential(
            ResidualBlock(in_features=500, out_features=1000),
            nn.ReLU(),
            ResidualBlock(in_features=1000, out_features=1000),
            nn.ReLU(),
            ResidualBlock(in_features=1000, out_features=1000),
            nn.ReLU(),
            ResidualBlock(in_features=1000, out_features=1000),
            nn.ReLU(),
            ResidualBlock(in_features=1000, out_features=1000),
            nn.ReLU(),
            ResidualBlock(in_features=1000, out_features=1000),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            # nn.Linear(in_features=720, out_features=480),
            # nn.Tanh(),
            nn.Linear(in_features=1000, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=134),
            # nn.Sigmoid()
        )
        for a in self.encoder:
            if isinstance(a, torch.nn.Linear):
                # nn.init.constant_(a.weight, 0.01)
                # nn.init.constant_(a.bias, 0.01)
                nn.init.kaiming_normal_(a.weight, mode='fan_in', nonlinearity='relu')
        for a in self.decoder:
            if isinstance(a, torch.nn.Linear):
                nn.init.kaiming_normal_(a.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_normal_(a.bias, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        encoder_out = self.encoder(x)
        encoder_out = self.neck(encoder_out)
        decoder_out = self.decoder(encoder_out)
        return decoder_out


def train(
        opt
):
    data = Customized_LIFE_BusDataset(
        opt.data_path)
    autoEncoder = convert_model_to_double(AutoEncoder().to(opt.device))
    busautoEncoder = convert_model_to_double(BUS_AutoEncoder().to(opt.device))
    pre = convert_model_to_double(BPNET_Life_time().to(opt.device))
    # pre_bitch = convert_model_to_double(BPNET().to(opt.device))
    loss_function = nn.MSELoss().to(opt.device)
    Optimizer = optim.Adam(pre.parameters(), lr=opt.lr)

    # optimize
    if opt.load_bool:
        print(f"Start from: {opt.weights.split('log')[-1]}")

        loaded_state = torch.load(opt.weights)
        autoEncoder.load_state_dict(loaded_state['user_auto'])

        # loaded_state = torch.load(opt.weights_bus_auto)

        busautoEncoder.load_state_dict(loaded_state['bus_auto'])

    lr = opt.lr
    dataload = DataLoader(data, batch_size=100, shuffle=False, sampler=None,
                          batch_sampler=None, num_workers=0, collate_fn=None,
                          pin_memory=False, drop_last=False, timeout=0,
                          worker_init_fn=None)
    for epu in range(opt.epoch):

        # if epu % 500 == 0 and epu > 0:
        #     print(f"Update lr to{lr * 0.8: .5f}")
        #     lr = lr * 0.8
        #     Optimizer = optim.SGD(pre.parameters(), lr=lr)

        user_total_loss = 0

        pbar = tqdm(dataload, desc=f'Epoch {epu}/{opt.epoch}', postfix=dict)
        k = 0


        for bus_label, bus_log_label, bus_index, bus_data, user_label, user_log_label, user_index, user_data in pbar:

            k += 1
            if True:

                bus_rows = bus_data.to(opt.device)

                bus_a, User_output_row = busautoEncoder(bus_rows)
                bus_auto_out = (bus_a.detach() + torch.ones_like(bus_a)) * 0.5


                user_rows = torch.cat(
                    [user_data.to(opt.device), bus_auto_out],
                    dim=-1).to(opt.device)
                # output = nn.functional.softmax(pre(user_rows), dim=-1).unsqueeze(1)

                user_label = torch.ceil(user_label / 7)
                # a = torch.zeros_like(user_label)




                loss = total_loss(pre, user_rows, user_label,torch.ones_like(user_index)-user_index, 0.3, 0.3, 1)
                # print(train_data[2][i])
                # label_split = split_and_normalize_number(train_data[2][i]).to(opt.device)

                # print(output)
                # print(train_data[2][i])
                # print("-")


                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()


                user_total_loss += loss.item()
            if k >0:
                break



                    # if ab > mas:
                    #     break
        # b = 0
        # for train_data, val_data in pbar:
        #     b += 1
        #     if True:
        #         for i in range(train_data[0].shape[0]):
        #             ab += 1
        #             bus_a, User_output_row = busautoEncoder(bus_rows)
        #             user_a, User_output_row = autoEncoder(train_data[0][i].to(opt.device))
        #             bus_auto_out = (bus_a.detach() + torch.ones_like(bus_a))/2
        #
        #             user_rows = torch.cat([train_data[0][i].to(opt.device), bus_auto_out], dim=-1).to(opt.device)
        #
        #             output = pre(user_rows)
        #             abs_count += abs(output * 931 - train_data[2][i] * 931).item()
        #             user_total_loss += loss_function(output, train_data[2][i].unsqueeze(0).to(opt.device)).item()
        #             # print(output)
        #             # print(train_data[2][i])
        #             # print("---")
        #
        #         if b > mas:
        #             break

                # User

        # print(user_total_loss)
        user_total_loss_mean = user_total_loss / k

        # user_total_loss_var = torch.var(user_total_loss)

        print('Epoch: {}, Loss: {:.12f}'.format(epu + 1, user_total_loss_mean))

        if opt.save_bool == True:
            torch.save({
                "user_auto": autoEncoder.state_dict(),
                "bus_auto": busautoEncoder.state_dict(),
                "pre": pre.state_dict()
                # "User_encoder_2": User_encoder_2.state_dict(),
                # "User_encoder_3": User_encoder_3.state_dict(),
                # "User_decoder": User_decoder.state_dict(),
            }, opt.log_path + f'User_line_Epu{epu + 1}_{user_total_loss_mean:.12f}.pth')


def opt_generate():
    """
    parameter for training
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='G:\\林昊成个人\\小论文存档\\1-定制公交的多任务深度生存分析\\数据\\定制公交_sample.csv',
                        help='data_csv_'),
    parser.add_argument('--config_path', type=str,
                        default='test.json',
                        help='path_2_model_config')
    parser.add_argument('--log_path', type=str,
                        default='G:\\林昊成个人\\小论文存档\\1-定制公交的多任务深度生存分析\\代码\\log_index\\',
                        help='path_2_model_save')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='model path(s)')
    parser.add_argument('--epoch', type=int,
                        default=10000,
                        help='model path(s)')
    parser.add_argument('--load_bool',
                        default=True,
                        help='load prior model or not')
    parser.add_argument('--save_bool',
                        default=True,
                        help='save model or not')
    parser.add_argument('--weights_user_auto', nargs='+', type=str,
                        default='G:\\林昊成个人\小论文存档\\1-定制公交的多任务深度生存分析\\数据\\重点模型\\NEW_User_auto_Epu161_0.001400132557_0.0.pth',
                        help='model path(s)')
    parser.add_argument('--weights_bus_auto', nargs='+', type=str,
                        default='G:\\林昊成个人\小论文存档\\1-定制公交的多任务深度生存分析\\数据\\重点模型\\NEW_Bus_auto_Epu158_0.000093804128_0.0.pth',
                        help='model path(s)')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='G:\\林昊成个人\小论文存档\\1-定制公交的多任务深度生存分析\\数据\\重点模型\\Sh_User_busauto_pre_Epu790_0.001539929642__abs_19.36269430956978.pth',
                        help='model path(s)')
    parser.add_argument('-device', nargs='+', type=str,
                        default='cuda:1',
                        help='model path(s)')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = opt_generate()
    train(opt)