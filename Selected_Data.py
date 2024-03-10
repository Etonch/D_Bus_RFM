import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import math
import random
pd.set_option('display.max_columns', None)
class Customized_LIFE_BusDataset(Dataset):

    def __init__(self, csv_paths, Martix_similarity, flag='train', keep_rate=1, id = 0, simi_number=10):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.keep_rate = keep_rate
        self.Martix_similarity = Martix_similarity
        self.id = id
        self.simi_number=simi_number

        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.__load_data__(csv_paths)

    def __getitem__(self, index):
        # 根据索引返回数据
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return (self.bus_label[index],
                self.bus_log_label[index],
                self.bus_index[index],
                self.bus_data[index],
                self.user_label[index],
                self.user_log_label[index],
                self.user_index[index],
                self.user_data[index],
                self.id
                )

    def __len__(self):
        # 返回数据的长度
        return len(self.user_data)

    # 以下可以不在此定义。

    # 如果不是直接传入数据data，这里定义一个加载数据的方法
    def __load_data__(self, csv_paths: str):
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        self.user_col = [
            "User_Survival_Days",
            "Log_User_Survival_Days",
            "User_Censored_Flag",
            "Trip_Freq",
            "Trip_Duration",
            "Trip_Peakpct",
            "T_Entropy",
            "S_Entropy",
            "User_O_Entropy",
            "User_O_Specificuse_Transport",
            "User_O_Specificuse_Ecocenter",
            "User_O_Specificuse_University",
            "User_O_transitaccess",
            "User_D_Entropy",
            "User_D_Specificuse_Transport",
            "User_D_Specificuse_Ecocenter"    ,
            "User_D_Specificuse_University"    ,
            "User_D_transitaccess"    ,
            "Delta_taxicost",
            "Delta_taxitime"  ,
            "Delta_transitcost",
            "Delta_transittime"

        ]
        self.bus_col = [
            "Service_Days",
            "Service_Freq",
            "O_Stopnum",
            "D_Stopnum",
            "O_Entropy",
            "O_Specificuse_Transport",
            "O_Specificuse_Ecocenter",
            "O_Specificuse_University",
            "O_Transitaccess"	,
            "D_Entropy"	,
            "D_Specificuse_Transport"	,
            "D_Specificuse_Ecocenter"	,
            "D_Specificuse_University"	,
            "D_Transitaccess"	,
            "O_Jobdensity"	,
            "D_Jobdensity"	,
            "Line_Survival_Days",
            "Log_Line_Survival_Days",
            "Line_Censored_Flag",
            "Most_Used_Line",

        ]
        #定义乘客与公交信息列
        data_raw = pd.read_csv(csv_paths).fillna(0)
        data_raw = data_raw.drop(columns=['User_ID'])

        similarities = self.Martix_similarity[id]
        top_similar_indices = np.argsort(similarities)[-self.simi_number:-1]

        # 从DataFrame中提取相似样本数据
        data_raw = data_raw.iloc[top_similar_indices]

        numeric_columns = data_raw.select_dtypes(include='number').columns

        # data_raw['User_Survival_Days'] = data_raw['User_Survival_Days'].apply(
        #     lambda col: math.log2(col)).fillna(0)

        data_raw[numeric_columns.drop(['User_Survival_Days', 'User_Censored_Flag'])] = data_raw[numeric_columns.drop(['User_Survival_Days', 'User_Censored_Flag'])].apply(lambda col: (col) / (col.max())).fillna(0)

        self.slide_preprocess(data_raw)



    def slide_preprocess(self, data_row):
        # 将data 做一些预处理,拆分成车辆数据和乘客数据
        self.keep_rate = 1
        user_data = data_row[self.user_col]
        bus_data = data_row[self.bus_col]
        self.bus_label = torch.tensor(bus_data['Line_Survival_Days'].values)
        self.bus_log_label = torch.tensor(bus_data['Log_Line_Survival_Days'].values)
        self.bus_index = torch.tensor(bus_data['Line_Censored_Flag'].values)

        self.user_label = torch.tensor(user_data['User_Survival_Days'].values)
        self.user_log_label = torch.tensor(user_data['Log_User_Survival_Days'].values)
        self.user_index = torch.tensor(user_data['User_Censored_Flag'].values)

        self.user_data = torch.tensor(user_data.drop(columns=['User_Survival_Days', 'Log_User_Survival_Days', 'User_Censored_Flag']).values)
        self.bus_data = torch.tensor(bus_data.drop(columns=['Line_Survival_Days', "Log_Line_Survival_Days", 'Line_Censored_Flag', 'Most_Used_Line']).values)


        #原始数据生成

    def generate_random_indices(self, dataframe):
        """
        生成一个随机的索引列表，用于筛选DataFrame的行，同时保留第一行。

        参数:
        - dataframe: 要处理的DataFrame
        - keep_ratio: 保留的比例，默认为0.8

        返回:
        - 随机索引的列表
        """
        # 获取DataFrame的行数
        num_rows = len(dataframe)

        # 计算需要保留的行数
        keep_rows = int(self.keep_rate * num_rows)

        # 生成随机索引（不包括第一行）
        random_indices = np.random.choice(range(1, num_rows), size=keep_rows, replace=False)

        # 将第一行索引添加到列表中
        random_indices = [0] + list(np.sort(random_indices))
        print(random_indices)

        return random_indices