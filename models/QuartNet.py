import torch
import torch.nn.functional as F
import torch.nn as nn

from activate_fun.Swish import Swish


class SeprationConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=33, last=False, mask=True, dilation=1,
                 stride=1, drop_rate=0.1):
        super(SeprationConv, self).__init__()
        self.last = last
        self.mask = mask
        if dilation > 1:
            self.depthwise_conv = nn.Conv1d(in_ch, in_ch, kernel_size=(k,), stride=(stride,),
                                            padding=((dilation * k) // 2 - 1,), groups=in_ch,
                                            dilation=(dilation,), bias=False)
        else:
            self.depthwise_conv = nn.Conv1d(in_ch, in_ch, kernel_size=(k,), stride=(stride,),
                                            padding=(k // 2,), groups=in_ch, dilation=(dilation,),
                                            bias=False)
        self.pointwise_conv = nn.Conv1d(in_ch, out_ch, kernel_size=(1,), stride=(1,),
                                        bias=False)
        self.bn = nn.BatchNorm1d(out_ch, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.swish = Swish()
        self.maskcnn = MaskCNN()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, input, percents):
        x = self.depthwise_conv(input)
        x = self.pointwise_conv(x)
        x = self.channel_shuffle(x, groups=1)
        if self.mask:
            x = self.maskcnn(x, percents)
        x = self.bn(x)
        if not self.last:
            x = self.relu(x)
        x = self.dropout(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, time = x.data.size()
        channels_per_group = num_channels // groups
        if not channels_per_group * groups == num_channels:
            raise Exception("group数和通道数不匹配，请保证group能够被num_channels整除")
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, time)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, time)
        return x


class QuartNetBlock(nn.Module):
    def __init__(self, repeat=3, in_ch=1, out_ch=32, k=33, mask=True):
        super(QuartNetBlock, self).__init__()
        seq = []
        for i in range(0, repeat - 1):
            sep = SeprationConv(in_ch, in_ch, k, mask)
            seq.append(sep)
        self.reside = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=(1,), bias=False),
            nn.BatchNorm1d(out_ch, eps=1e-3),
        )
        last_sep = SeprationConv(in_ch, out_ch, k=k, last=True, mask=mask)
        seq.append(last_sep)
        self.seq = nn.ModuleList(seq)
        self.last_relu = nn.ReLU()

    def forward(self, x, percents):
        start = x
        for m in self.seq:
            x = m(x, percents)
        res_out = self.reside(start)
        x = x + res_out
        x = self.last_relu(x)
        return x


class QuartNet(nn.Module):
    def __init__(self):
        super(QuartNet, self).__init__()
        self.first_cnn = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=33, stride=2,
                      padding=16),
            nn.BatchNorm1d(256, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        # self.first_cnn = SeprationConv(64,256,33,stride=2,mask=True)
        self.block1 = QuartNetBlock(repeat=5, in_ch=256, out_ch=256, k=33)
        # self.block12 = QuartNetBlock(repeat=5,in_ch=256,out_ch=256,k=33) # add layer
        self.block2 = QuartNetBlock(repeat=5, in_ch=256, out_ch=256, k=39)
        # self.block22 = QuartNetBlock(repeat=5,in_ch=256,out_ch=256,k=39) # add layer
        self.block3 = QuartNetBlock(repeat=5, in_ch=256, out_ch=512, k=51)
        self.block4 = QuartNetBlock(repeat=5, in_ch=512, out_ch=512, k=63)
        self.block5 = QuartNetBlock(repeat=5, in_ch=512, out_ch=512, k=75)
        self.last_cnn = SeprationConv(512, 512, k=87, last=False, mask=True, dilation=1)  # 空洞率为2收敛比1慢 cer0.99-->0.92
        self.last_cnn2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, stride=1),
            nn.BatchNorm1d(1024, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, input, percents):
        x = input.squeeze(dim=1).contiguous()
        x = self.first_cnn(x)
        x = self.block1(x, percents)
        # x = self.block12(x,percents)
        x = self.block2(x, percents)
        # x = self.block22(x,percents)
        x = self.block3(x, percents)
        x = self.block4(x, percents)
        x = self.block5(x, percents)
        x = self.last_cnn(x, percents)
        x = self.last_cnn2(x)
        return x


class QuartNet12(nn.Module):
    def __init__(self):
        super(QuartNet12, self).__init__()
        # self.first_cnn = nn.Sequential(
        #     nn.Conv1d(64, 256, kernel_size=(33,), stride=(2,),
        #               padding=(16,)),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        # )
        self.first_cnn = SeprationConv(in_ch=64, out_ch=256, k=33, last=False, mask=False, stride=2)
        self.block1 = QuartNetBlock(repeat=1, in_ch=256, out_ch=256, k=33, mask=False)
        self.block12 = QuartNetBlock(repeat=1, in_ch=256, out_ch=256, k=33, mask=False)
        self.block13 = QuartNetBlock(repeat=1, in_ch=256, out_ch=256, k=33, mask=False)
        # self.block12 = QuartNetBlock(repeat=5,in_ch=256,out_ch=256,k=33) # add layer
        self.block2 = QuartNetBlock(repeat=1, in_ch=256, out_ch=256, k=39, mask=False)
        self.block22 = QuartNetBlock(repeat=1, in_ch=256, out_ch=256, k=39, mask=False)
        self.block23 = QuartNetBlock(repeat=1, in_ch=256, out_ch=256, k=39, mask=False)
        # self.block22 = QuartNetBlock(repeat=5,in_ch=256,out_ch=256,k=39) # add layer
        self.block3 = QuartNetBlock(repeat=1, in_ch=256, out_ch=512, k=51, mask=False)
        self.block32 = QuartNetBlock(repeat=1, in_ch=512, out_ch=512, k=51, mask=False)
        self.block33 = QuartNetBlock(repeat=1, in_ch=512, out_ch=512, k=51, mask=False)
        self.block4 = QuartNetBlock(repeat=1, in_ch=512, out_ch=512, k=63, mask=False)
        self.block42 = QuartNetBlock(repeat=1, in_ch=512, out_ch=512, k=63, mask=False)
        self.block43 = QuartNetBlock(repeat=1, in_ch=512, out_ch=512, k=63, mask=False)
        self.block5 = QuartNetBlock(repeat=1, in_ch=512, out_ch=512, k=75,  mask=False)
        self.last_cnn2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=(1,), stride=(1,), bias=False),
            nn.BatchNorm1d(1024, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, input, percents):
        # x = input.view(input.size(0),input.size(2),input.size(3))
        x = input.squeeze(dim=1).contiguous()
        x = self.first_cnn(x, percents)
        x = self.block1(x, percents)
        x = self.block12(x, percents)
        x = self.block13(x, percents)
        # x = self.block12(x,percents)
        x = self.block2(x, percents)
        x = self.block22(x, percents)
        x = self.block23(x, percents)
        # x = self.block22(x,percents)
        x = self.block3(x, percents)
        x = self.block32(x, percents)
        x = self.block33(x, percents)
        x = self.block4(x, percents)
        x = self.block42(x, percents)
        x = self.block43(x, percents)
        x = self.block5(x, percents)
        # x = self.last_cnn(x, percents)
        x = self.last_cnn2(x)
        return x

class BatchLSTM(nn.Module):
    def __init__(self, in_ch=64, out_ch=512,
                 batch_f=True, bidirection=True):
        super().__init__()
        self.rnn = nn.LSTM(in_ch, out_ch, num_layers=1,
                           batch_first=batch_f, bidirectional=bidirection)

    def forward(self, x, length):
        x = nn.utils.rnn.pack_padded_sequence(x, enforce_sorted=False,
                                              lengths=length, batch_first=True)
        x, h = self.rnn(x)
        # x, h = self.rnn2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # N*T*C
        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.maskcnn = MaskCNN()
        self.cnn = QuartNet()
        self.rnn = BatchLSTM(64 * 128, 128, True, True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, 29)

    def forward(self, input, percents):
        x = self.cnn(input, percents)  # N*32*F*T
        # x = self.maskcnn(x, percents)
        x = x.view(x.size(0), x.size(1) * x.size(2), -1).transpose(1, 2).contiguous()
        lengths = torch.mul(x.size(1), percents).int()
        x = self.rnn(x, lengths)
        x = x.transpose(1, 2)  # N*C*T
        x = self.bn1(x)
        x = self.fc(x.transpose(1, 2))  # N*T*class
        x = nn.functional.log_softmax(x, dim=-1)
        return x


class MyModel2(nn.Module):
    def __init__(self, labels):
        super(MyModel2, self).__init__()
        # self.maskcnn = MaskCNN()
        self.labels = labels
        self.cnn = QuartNet12()
        # self.last_cnn3 = nn.Sequential(
        #     nn.Conv1d(1024, len(self.labels)+1, kernel_size=1, stride=1, dilation=1),  # 空洞率2较好 cer=0.98-->0.53(dila=1)
        #     nn.BatchNorm1d(len(self.labels)+1),
        #     nn.ReLU(),
        # )
        self.last_cnn3 = nn.Conv1d(1024, len(self.labels) + 1, kernel_size=(1,))
        # self.rnn = BatchLSTM(64*128,128,True,True)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.fc = nn.Linear(256, 29)

    def forward(self, input, percents):
        x = self.cnn(input, percents)  # N*C*T
        x = self.last_cnn3(x)
        # x = self.maskcnn(x, percents)
        # x = x.view(x.size(0), x.size(1)*x.size(2), -1).transpose(1, 2).contiguous()
        # lengths=torch.mul(x.size(1), percents).int()
        # x = self.rnn(x, lengths)
        x = x.transpose(1, 2)  # N*T*C
        # x = self.bn1(x)
        # x = self.fc(x)  # N*T*class
        x = nn.functional.log_softmax(x, dim=-1)
        return x


class MaskCNN1(nn.Module):
    def forward(self, x, percents):
        lengths = torch.mul(x.size(3), percents).int()
        mask = torch.BoolTensor(x.size()).fill_(0)
        if x.is_cuda:
            mask = mask.cuda()
        for i, length in enumerate(lengths):
            length = length.item()
            if (mask[i].size(2) - length) > 0:
                mask[i].narrow(
                    2, length, mask[i].size(2) - length).fill_(1)
        x = x.masked_fill(mask, 0)
        return x


class MaskCNN(nn.Module):
    def forward(self, x, percents):
        lengths = torch.mul(x.size(2), percents).int()
        mask = torch.BoolTensor(x.size()).fill_(0)
        if x.is_cuda:
            mask = mask.cuda()
        for i, length in enumerate(lengths):
            length = length.item()
            if (mask[i].size(1) - length) > 0:
                mask[i].narrow(
                    1, length, mask[i].size(1) - length).fill_(1)
        x = x.masked_fill(mask, 0)
        return x


if __name__ == "__main__":
    input = torch.rand([8, 1, 64, 512], dtype=torch.float32)
    percents = torch.rand([8], dtype=torch.float32)
    model = MyModel2([" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                   "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"])
    out = model(input, percents)
    print(out.size())

    from ptflops import get_model_complexity_info
    import torch
    def prepare_input(inputs):
        """
        对输入的数据进行封装
        """
        x = torch.rand((8, *inputs), dtype=torch.float32)
        percents = torch.rand([8], dtype=torch.float32)
        return dict(input=x, percents=percents)
    macs, params = get_model_complexity_info(model, (1, 64, 512),
                                             as_strings=True,
                                             input_constructor = prepare_input,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30} {:<8}'.format('macs:', macs))
    print('{:<30} {:<8}'.format('params:', params))
