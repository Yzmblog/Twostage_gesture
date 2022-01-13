from matplotlib.pyplot import text
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from model.tcn import TemporalConvNet
from torch import Tensor

###audio shape:  torch.Size([128, 382, 128]) 
# text shape:  torch.Size([128, 64, 300]) 
# pose shape:  torch.Size([128, 64, 45])
# logmel400    [128, 442, 64]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
        downsample=None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample #nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Twostage(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.text_encoder = TemporalConvNet(num_inputs=768, num_channels=[256, 128, 96, 64, 45], dropout=0.3)
        self.proposal_net = BasicBlock(inplanes=1, planes=6, downsample=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1))



        self.audio_encoder = TemporalConvNet(num_inputs=64, num_channels=[64, 32, 16, 32, 45], dropout=0.3)
        self.audio_mapping = nn.Sequential(
            nn.Linear(442, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64)
        )

        self.fusion = BasicBlock(inplanes=6, planes=1, downsample=nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1))

        self.gru = nn.GRU(45, hidden_size=64, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(64, 56),
            nn.LeakyReLU(True),
            nn.Linear(56, 45) ##pose_dim = 15*3
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True



    def forward(self, in_audio, in_text):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat = self.text_encoder(in_text.transpose(1, 2)).transpose(1, 2)
        text_feat = torch.unsqueeze(text_feat, dim=1)
        propasal_pose = self.proposal_net(text_feat)
        
        # pose_speed = self.audio_encoder(in_audio.transpose(1, 2))
        # pose_speed = self.audio_mapping(pose_speed).transpose(1, 2)

        ###----------no need------------------###
        # zeros = torch.zeros((pose_speed.shape[0], 1, pose_speed.shape[2]))
        # zeros = zeros.to(device)
        # audio_feat = torch.cat((pose_speed, zeros), dim=1)
        ###-----------old -----------------###
        # audio_feat = torch.unsqueeze(pose_speed, dim=1)

        # concat_feat = torch.cat((propasal_pose, audio_feat), dim=1)
        latent_code = self.fusion(propasal_pose)
        latent_code = torch.squeeze(latent_code, dim=1)


        output, decoder_hidden = self.gru(latent_code, decoder_hidden)
        output = output[:, :, :64] + output[:, :, 64:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(latent_code.shape[0], latent_code.shape[1], -1)

        return decoder_outputs, None, propasal_pose


class Base(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # if args.text_flag == 'w2v':
        #     self.text_encoder = TemporalConvNet(num_inputs=300, num_channels=[256, 128, 96, 64, 32], dropout=0.3)
            
        # else:
        self.text_encoder = TemporalConvNet(num_inputs=768, num_channels=[256, 128, 96, 64, 32], dropout=0.3)
        # self.text_encoder = TemporalConvNet(num_inputs=1, num_channels=[16, 8, 4, 8, 16], dropout=0.3)

        
        self.text_mapping = nn.Linear(32, 16)

        # if args.audio_flag == 'log_mel_512':
        #     self.audio_encoder = TemporalConvNet(num_inputs=128, num_channels=[128, 96, 64, 32, 16], dropout=0.3)
        #     self.audio_mapping = nn.Sequential(
        #         nn.Linear(382, 256),
        #         nn.LeakyReLU(True),
        #         nn.Linear(256, 128),
        #         nn.LeakyReLU(True),
        #         nn.Linear(128, 64)
        #     )
        # elif args.audio_flag == 'raw':
        #     self.audio_encoder = TemporalConvNet(num_inputs=64, num_channels=[128, 96, 64, 32, 16], dropout=0.3)
        #     self.audio_mapping = nn.Sequential(
        #         nn.Linear(256, 128),
        #         nn.LeakyReLU(True),
        #         nn.Linear(128, 96),
        #         nn.LeakyReLU(True),
        #         nn.Linear(96, 64)
        #     )
        # else:
        self.audio_encoder = TemporalConvNet(num_inputs=64, num_channels=[128, 96, 64, 32, 16], dropout=0.3)
        self.audio_mapping = nn.Sequential(
            nn.Linear(442, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64)
        )

        self.gru = nn.GRU(32, hidden_size=64, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(64, 56),
            nn.LeakyReLU(True),
            nn.Linear(56, 45) ##pose_dim = 15*3
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True



    def forward(self, in_audio, in_text):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        audio_feat = self.audio_encoder(in_audio.transpose(1, 2))
        text_feat = self.text_encoder(in_text.transpose(1, 2)).transpose(1, 2)
        # print(in_audio.shape)
        # print(audio_feat.shape)

        audio_feat = self.audio_mapping(audio_feat).transpose(1, 2)
        text_feat = self.text_mapping(text_feat)

        # print(audio_feat.shape) 
        latent_code = torch.cat((audio_feat, text_feat), dim=2)


        output, decoder_hidden = self.gru(latent_code, decoder_hidden)
        output = output[:, :, :64] + output[:, :, 64:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(latent_code.shape[0], latent_code.shape[1], -1)

        return decoder_outputs
