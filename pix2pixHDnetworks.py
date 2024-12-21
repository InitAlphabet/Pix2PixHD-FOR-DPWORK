from torchvision import models
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, ngf=32, local_generator_nums=1, global_downsample_times=3,
                 local_resnet_block_nums=3, global_resnet_block_nums=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        """
        :param input_channels: 输入通道
        :param output_channels: 输出通道
        :param ngf: 第一层的特征数
        :param local_generator_nums: 局部特征层次数 ，每一层缩放0.5*0.5
        :param global_downsample_times: 全局特征下采样次数
        :param local_resnet_block_nums: 局部特征残差模块数
        :param global_resnet_block_nums: 全局特征残差模块数
        :param norm_layer: 局部特征正则化方法
        :param padding_type:填充模式
        """
        super(LocalGenerator, self).__init__()
        self.local_generator_nums = local_generator_nums
        global_ngf = ngf * (2 ** local_generator_nums)
        global_model = GlobalGenerator(input_channels=input_channels, output_channels=output_channels,
                                       ngf=global_ngf, downsample_times=global_downsample_times,
                                       resnet_block_nums=global_resnet_block_nums)
        global_model = global_model.nets[:-3]  # 抛弃最后的输出
        self.global_nets = nn.Sequential(*global_model)
        for n in range(1, local_generator_nums + 1):
            # downsample
            global_ngf = ngf * (2 ** (local_generator_nums - n))
            model_downsample = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_channels, global_ngf, kernel_size=7, padding=0),
                norm_layer(global_ngf), nn.ReLU(True),

                nn.Conv2d(global_ngf, global_ngf * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(global_ngf * 2), nn.ReLU(True)]
            # residual blocks
            model_upsample = []
            for i in range(local_resnet_block_nums):
                model_upsample += [ResnetBlock(global_ngf * 2, padding_type=padding_type, norm_layer=norm_layer)]

            # upsample
            model_upsample += [
                nn.ConvTranspose2d(global_ngf * 2, global_ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(global_ngf),
                nn.ReLU(True)]

            # final convolution 最后的输出需要连接
            if n == local_generator_nums:
                model_upsample += [nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_p1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_p2', nn.Sequential(*model_upsample))
            self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)  # 共享参数

    def forward(self, x):
        downsampled_x = [x]
        for i in range(self.local_generator_nums):
            downsampled_x.append(self.downsample(downsampled_x[-1]))
        output = self.global_nets(downsampled_x[-1])  # 全局特征
        for n in range(self.local_generator_nums):
            down = getattr(self, 'model' + str(n) + '_p1')
            up = getattr(self, 'model' + str(n) + 'p2')
            output = up(down(downsampled_x[-n - 1]) + output)
        return output


class GlobalGenerator(nn.Module):
    # 全局特征提取生成器
    def __init__(self, input_channels, output_channels, ngf=64, downsample_times=3, resnet_block_nums=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(GlobalGenerator, self).__init__()
        assert (resnet_block_nums >= 0)
        activation = nn.ReLU(True)
        # input_channels * s * s
        feature_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_channels, out_channels=ngf, kernel_size=7, padding=0),
            norm_layer(ngf), activation
        ]
        # ngf * s * s
        down_blocks = []
        for i in range(downsample_times):
            mult = 2 ** i
            down_blocks.append(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1))  # 进一整除2
            down_blocks.append(norm_layer(ngf * mult * 2))
            down_blocks.append(activation)

        # 8ngf * (s//8) *(s//8)
        resnet_blocks = []
        mult = 2 ** downsample_times
        for i in range(resnet_block_nums):
            resnet_blocks.append(
                ResnetBlock(dim=ngf * mult, norm_layer=norm_layer, padding_type=padding_type, activation=activation)
            )
        # 8ngf * (s//8) *(s//8)
        up_blocks = []
        for i in range(downsample_times):
            mult = 2 ** (downsample_times - i)
            up_blocks += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                             output_padding=1),  # *2
                          norm_layer(int(ngf * mult / 2)), activation]
        # ngf * s * s
        feature_out = [nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
                       nn.Tanh()]

        self.nets = nn.Sequential(
            *feature_conv,
            *down_blocks,
            *resnet_blocks,
            *up_blocks,
            *feature_out
        )

    def forward(self, x):
        return self.nets(x)


class ResnetBlock(nn.Module):
    # 局部用InstanceNorm,全局用BatchNormal
    def __init__(self, dim, norm_layer, padding_type='reflect', activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MultiscaleDiscriminator(nn.Module):
    # 多尺度判别器 由通用判别器组合
    def __init__(self, input_channels, ndf=64, layer_nums=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, use_feature_maps=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.layer_nums = layer_nums
        self.use_feature_maps = use_feature_maps

        for i in range(num_D):
            netD = NLayerDiscriminator(input_channels, ndf, layer_nums, norm_layer, use_sigmoid, use_feature_maps)
            if use_feature_maps:
                for j in range(layer_nums + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'module_' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=(1, 1), count_include_pad=False)

    def singleD_forward(self, model, input_x):
        if self.use_feature_maps:
            result = [input_x]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input_x)]

    def forward(self, input_x):
        num_D = self.num_D
        result = []
        input_downsampled = input_x
        for i in range(num_D):
            if self.use_feature_maps:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.layer_nums + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(nn.Module):
    # 通用判别器，添加了自定义布局数量
    def __init__(self, input_channels, ndf=64, layer_nums=3, normalizer=nn.BatchNorm2d,
                 use_sigmoid=False, use_feature_maps=False):
        """
        :param input_channels: 输入维度
        :param ndf: 下采样核的起始数量
        :param layer_nums: 下采样层数
        :param normalizer: 正则层
        :param use_sigmoid: 是否使用这个，用了这个，损失函数就不要用lscan了
        :param use_feature_maps: 是否返回所有层次的特征图
        """
        super(NLayerDiscriminator, self).__init__()
        self.use_mult_features = use_feature_maps
        self.layer_nums = layer_nums
        # 下采样层
        module_list = [[
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ]]
        nf = ndf
        for n in range(1, layer_nums):
            nf_prev = nf
            nf = min(nf * 2, 512)
            module_list += [[
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=2),
                normalizer(nf),
                nn.LeakyReLU(0.2, True)
            ]]
        # 特征提取
        nf_prev = nf
        nf = min(nf * 2, 512)
        module_list += [[
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=2),
            normalizer(nf),
            nn.LeakyReLU(0.2, True)
        ]]
        module_list += [[nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=2)]]
        if use_sigmoid:
            module_list.append([nn.Sigmoid()])
        if use_feature_maps:
            for i in range(len(module_list)):
                setattr(self, f'module_{i}', nn.Sequential(*module_list[i]))
        else:
            sequence = []
            for each in module_list:
                sequence += each
            self.model = nn.Sequential(*sequence)

    def forward(self, input_x):
        if self.use_mult_features:
            output = [input_x]
            for i in range(self.layer_nums + 2):
                module = getattr(self, f'module_{i}')
                output.append(module(output[-1]))
            return output[1:]
        else:
            return self.model(input_x)


class GANLoss(nn.Module):
    def __init__(self, real=1, fake=0, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.real_value = real
        self.fake_value = fake
        self.real_target = None
        self.fake_target = None
        if use_lsgan:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.BCELoss()

    def generate_target(self, input_x, need_real):
        if need_real:
            if (self.real_target is None) or (self.real_target.numel() != input_x.numel()):
                real_tensor = torch.ones(input_x.size(), device=device).fill_(self.real_value)
                self.real_target = Variable(real_tensor, requires_grad=False)
            return self.real_target
        else:
            if (self.fake_target is None) or (self.fake_target.numel() != input_x.numel()):
                fake_tensor = torch.ones(input_x.size(), device=device).fill_(self.fake_value)
                self.fake_target = Variable(fake_tensor, requires_grad=False)
            return self.fake_target

    def __call__(self, input_x, need_real):
        if isinstance(input_x[0], list):
            loss = 0
            for each in input_x:
                target = self.generate_target(each[-1], need_real)
                loss += self.loss_fn(each[-1], target)
            return loss
        else:
            target = self.generate_target(input_x[-1], need_real)
            return self.loss_fn(input_x[-1], target)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # 强高层，低浅层

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        # 提取不同层次的特征，用于感知损失 ,用Vgg19的感知特征图来代替直接的图对图的l1loss
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
