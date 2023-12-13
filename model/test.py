import torch
from torch import nn
import numpy as np
import math

class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        print(position)
        pe = torch.zeros(self.time_len * self.joint_num, channel)
        print(pe.size())
        print(pe)

        # [0, 1, 2, 3]
        # [0, 2]
        
        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        print(div_term)
        print(position.size(), div_term.size())
        print(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        print(pe)
        pe[:, 1::2] = torch.cos(position * div_term)
        print(pe)
        print(pe.size())
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        print(pe.size())
        print(pe)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x



if __name__ == "__main__":
    PositionalEncoding(6, 4, 3, "spatial")
# end main