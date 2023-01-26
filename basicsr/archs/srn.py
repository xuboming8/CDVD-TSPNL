import torch.nn as nn
import basicsr.archs.blocks as blocks


class SRN(nn.Module):

    def __init__(self, out_channels=3, n_resblock=1, n_feat=32, kernel_size=5):
        super(SRN, self).__init__()
        print("Creating SRN_SVLRM Net")

        InBlock = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        InBlock.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                        for _ in range(n_resblock)])

        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                              for _ in range(n_resblock)])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                               for _ in range(n_resblock)])

        # decoder2
        Decoder_second = [blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)]
        Decoder_second.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))
        # decoder1
        Decoder_first = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                         for _ in range(n_resblock)]
        Decoder_first.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))

        OutBlock = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock)]

        OutBlock_Post = [nn.Conv2d(n_feat * 2, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)]

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)
        self.outBlock_post = nn.Sequential(*OutBlock_Post)

    def forward(self, x):

        first_scale_inblock = self.inBlock(x)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock)
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
        first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
        first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
        first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock)

        recons = self.outBlock_post(first_scale_outBlock)

        return recons