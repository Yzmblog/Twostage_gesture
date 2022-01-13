from model.tcn import TemporalConvNet
import torch
import torch.nn as nn


class A2GNet(nn.Module):
    def __init__(self, args):
        super(A2GNet, self).__init__()
        self.AudioEncoder = TemporalConvNet(num_inputs=args.AudioSize, num_channels=[args.HiddenSize]*3 + [16])
        self.MotionEncoder = TemporalConvNet(num_inputs=args.PoseDim, num_channels=[args.HiddenSize]*3 + [32])
        self.Mapping = TemporalConvNet(num_inputs=16, num_channels=[args.HiddenSize]*3 + [16])
        self.Decoder = TemporalConvNet(num_inputs=32, num_channels=[args.HiddenSize]*3)

        self.out = nn.Sequential(
            nn.Linear(args.HiddenSize, args.HiddenSize//2),
            nn.LeakyReLU(True),
            nn.Linear(args.HiddenSize//2, args.PoseDim)
        )

    def forward(self, audio, sample_vector, pose=None):
        audio = audio.transpose(1, 2)
        share_code_audio = self.AudioEncoder(audio)
        sample_specific_code1 = self.Mapping(sample_vector[0])
        sample_specific_code2 = self.Mapping(sample_vector[1])


        pose_mm = None
        pose_am = None
        share_code_motion = None
        motion_specific_code = None

        if pose is not None:
            pose = pose.transpose(1, 2)
            motion_code = self.MotionEncoder(pose)
            share_code_motion = motion_code[:, :16, :].transpose(1, 2)

            motion_specific_code = motion_code[:, 16:, :]

            pose_mm_de = self.Decoder(motion_code).transpose(1, 2)
            pose_mm_de = self.out(pose_mm_de.reshape(-1, pose_mm_de.shape[2]))
            pose_mm = pose_mm_de.reshape(audio.shape[0], audio.shape[2], -1)


            # motion_zeros_test = torch.zeros_like(motion_specific_code)
            # audio_zeros_test = torch.zeros_like(share_code_audio)
            conca_code_am = torch.cat((share_code_audio, motion_specific_code), dim=1)
            # conca_code_am = torch.cat((audio_zeros_test, motion_specific_code), dim=1)
            # conca_code_am = torch.cat((share_code_audio, motion_zeros_test), dim=1)


            pose_am_de = self.Decoder(conca_code_am).transpose(1, 2)
            pose_am_de = self.out(pose_am_de.reshape(-1, pose_am_de.shape[2]))
            pose_am = pose_am_de.reshape(audio.shape[0], audio.shape[2], -1)

        # print(share_code_audio.shape, sample_specific_code1.shape)
        conca_code_ar1 = torch.cat((share_code_audio, sample_specific_code1), dim=1)
        conca_code_ar2 = torch.cat((share_code_audio, sample_specific_code2), dim=1)

        pose_ar1_de = self.Decoder(conca_code_ar1).transpose(1, 2)
        pose_ar1_de = self.out(pose_ar1_de.reshape(-1, pose_ar1_de.shape[2]))
        pose_ar1 = pose_ar1_de.reshape(audio.shape[0], audio.shape[2], -1)


        pose_ar2_de = self.Decoder(conca_code_ar2).transpose(1, 2)
        pose_ar2_de = self.out(pose_ar2_de.reshape(-1, pose_ar2_de.shape[2]))
        pose_ar2 = pose_ar2_de.reshape(audio.shape[0], audio.shape[2], -1)

        reconstruct_code1 = self.MotionEncoder(pose_ar1.transpose(1, 2))
        reconstruct_code1 = reconstruct_code1[:, 16:, :]


        return (pose_mm, pose_am, pose_ar1, pose_ar2, share_code_motion, share_code_audio.transpose(1, 2), reconstruct_code1, sample_specific_code1, motion_specific_code)
