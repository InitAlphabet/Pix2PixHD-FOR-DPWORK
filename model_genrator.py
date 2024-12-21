import torch.optim

import pix2pixHDnetworks as p2phd


def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif className.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class p2phdModel(torch.nn.Module):

    def __init__(self, config):

        super(p2phdModel, self).__init__()
        self.config = config
        self.generator = p2phd.GlobalGenerator(input_channels=3, output_channels=3)
        self.generator.apply(weights_init)
        self.discriminator = p2phd.MultiscaleDiscriminator(input_channels=6,
                                                           use_feature_maps=config.NEED_FEATURE_MAPS,
                                                           num_D=config.NUM_DIS,
                                                           layer_nums=config.LAYER_NUM)
        self.discriminator.apply(weights_init)
        self.optimizer_G = torch.optim.Adam(params=self.generator.parameters(), lr=config.G_LR, betas=config.G_BETA)
        self.optimizer_D = torch.optim.Adam(params=self.discriminator.parameters(), lr=config.D_LR, betas=config.D_BETA)
        self.D_loss_fn = p2phd.GANLoss()
        self.l1_loss_fn = torch.nn.L1Loss()
        self.vgg_loss_fn = p2phd.VGGLoss()
        self.need_feature_maps = config.NEED_FEATURE_MAPS
        self.num_D = config.NUM_DIS
        self.layer_num = config.LAYER_NUM
        self.feature_beta = config.FEATURE_BETA
        self.loss_names = [
            'D_fake_loss', 'D_real_loss', 'GAN_loss', 'GAN_feature_loss', 'VGG_loss'
        ]

    def forward(self, input_img, tar_img=None):
        fake_img = self.generator(input_img)
        if not self.training:
            return fake_img
        # D_fake_loss
        fake_dis_D = self.discriminator(torch.cat((fake_img.detach(), input_img), dim=1))
        D_fake_loss = self.D_loss_fn(fake_dis_D, False)
        # D_real_loss
        real_dis_D = self.discriminator(torch.cat((tar_img, input_img), dim=1))
        D_real_loss = self.D_loss_fn(real_dis_D, True)
        # GAN_loss
        fake_dis_G = self.discriminator(torch.cat((fake_img, input_img), dim=1))
        GAN_loss = self.D_loss_fn(fake_dis_G, True)
        # GAN_feature_loss 代替l1 loss
        GAN_feature_loss = 0
        if self.config.NEED_FEATURE_MAPS:
            feat_weights = 4.0 / (self.layer_num + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(fake_dis_G[i]) - 1):
                    GAN_feature_loss += feat_weights * D_weights * \
                                        self.l1_loss_fn(fake_dis_G[i][j], real_dis_D[i][j].detach()) * self.feature_beta
        VGG_loss = 0
        VGG_loss += self.vgg_loss_fn(fake_img, tar_img) * self.feature_beta
        return [D_fake_loss, D_real_loss, GAN_loss, GAN_feature_loss, VGG_loss]
