import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

#DWT and IDWT
##########################################################################
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

def idwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return idwt_init(x)

#Coordinate Attention
##########################################################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CAttention(nn.Module):
    def __init__(self, inp, reduction_ratio=4.):
        super(CAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(4, int(inp/reduction_ratio))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = x * a_w * a_h
        return out
    
class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

#ARB
##########################################################################
class ARB(nn.Module):
    def __init__(self, dim, att_reduction_ratio=4., bias=False):
        super(ARB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            CAttention(dim, att_reduction_ratio),
        )

    def forward(self, x):
        return x + self.conv(x)
    
class ARB_new(nn.Module):
    def __init__(self, dim, att_reduction_ratio=4., bias=False):
        super(ARB_new, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            SimAM(dim, att_reduction_ratio),
        )

    def forward(self, x):
        return x + self.conv(x)


## Layer Norm
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

#GRB
##########################################################################
class GRB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2., bias=False):
        super(GRB, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.conv1 = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x0 = self.norm(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x0)
        x3 = F.gelu(x1) * x2
        out = self.out(x3)
        return x + out


#EFEM
##########################################################################
class EFEM(nn.Module):
    def __init__(self, dim=48, att_reduction_ratio=4., ffn_expansion_factor=2., bias=False):
        super(EFEM, self).__init__()

        self.att = ARB(dim, att_reduction_ratio, bias)
        self.ffn = GRB(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.att(x)
        x = self.ffn(x)
        return x
    
class EFEM_new(nn.Module):
    def __init__(self, dim=48, att_reduction_ratio=4., ffn_expansion_factor=2., bias=False):
        super(EFEM_new, self).__init__()

        self.att = ARB_new(dim, att_reduction_ratio, bias)
        self.ffn = GRB(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.att(x)
        x = self.ffn(x)
        return x


#WaveCNN_CR
##########################################################################
class WaveCNN_CR(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dim=48, num_blocks=3, att_reduction_ratio=4.0, ffn_expansion_factor=2.0, bias=False):

        super(WaveCNN_CR, self).__init__()

        self.patch_embed = nn.Conv2d(input_nc, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.down0_1 = DWT()  ## From Level 0 to Level 1
        self.encoder_level1 = nn.Sequential(*[EFEM(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down1_2 = DWT()  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[EFEM(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down2_3 = DWT()  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[EFEM(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down3_4 = DWT()  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up4_3 = IDWT()  ## From Level 4 to Level 3
        self.decoder_level3 = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up3_2 = IDWT()  ## From Level 3 to Level 2
        self.decoder_level2 = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up2_1 = IDWT()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up1_0 = IDWT()  ## From Level 1 to Level 0  (NO 1x1 conv to reduce channels)
        self.refinement = nn.Sequential(*[EFEM(dim, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.output = nn.Conv2d(int(dim), output_nc, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_l, inp_enc_level1_h = self.down0_1(inp_enc_level1)

        out_enc_level1_h = self.encoder_level1(inp_enc_level1_h)
        inp_enc_level2_l, inp_enc_level2_h = self.down1_2(inp_enc_level1_l)

        out_enc_level2_h = self.encoder_level2(inp_enc_level2_h)
        inp_enc_level3_l, inp_enc_level3_h = self.down2_3(inp_enc_level2_l)

        out_enc_level3_h = self.encoder_level3(inp_enc_level3_h)
        inp_enc_level4_l, inp_enc_level4_h = self.down3_4(inp_enc_level3_l)

        inp_enc_level4 = torch.cat([inp_enc_level4_l, inp_enc_level4_h], 1)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3_l = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3_l, out_enc_level3_h], 1)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2_l = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2_l, out_enc_level2_h], 1)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1_l = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1_l, out_enc_level1_h], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.up1_0(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
    
class WaveCNN_CR_new(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dim=48, num_blocks=3, att_reduction_ratio=4.0, ffn_expansion_factor=2.0, bias=False):

        super(WaveCNN_CR_new, self).__init__()

        self.patch_embed = nn.Conv2d(input_nc, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.down0_1 = DWT()  ## From Level 0 to Level 1
        self.encoder_level1 = nn.Sequential(*[EFEM_new(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down1_2 = DWT()  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[EFEM_new(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down2_3 = DWT()  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[EFEM_new(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down3_4 = DWT()  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[EFEM_new(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up4_3 = IDWT()  ## From Level 4 to Level 3
        self.decoder_level3 = nn.Sequential(*[EFEM_new(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up3_2 = IDWT()  ## From Level 3 to Level 2
        self.decoder_level2 = nn.Sequential(*[EFEM_new(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up2_1 = IDWT()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[EFEM_new(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up1_0 = IDWT()  ## From Level 1 to Level 0  (NO 1x1 conv to reduce channels)
        self.refinement = nn.Sequential(*[EFEM_new(dim, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.output = nn.Conv2d(int(dim), output_nc, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_l, inp_enc_level1_h = self.down0_1(inp_enc_level1)

        out_enc_level1_h = self.encoder_level1(inp_enc_level1_h)
        inp_enc_level2_l, inp_enc_level2_h = self.down1_2(inp_enc_level1_l)

        out_enc_level2_h = self.encoder_level2(inp_enc_level2_h)
        inp_enc_level3_l, inp_enc_level3_h = self.down2_3(inp_enc_level2_l)

        out_enc_level3_h = self.encoder_level3(inp_enc_level3_h)
        inp_enc_level4_l, inp_enc_level4_h = self.down3_4(inp_enc_level3_l)

        inp_enc_level4 = torch.cat([inp_enc_level4_l, inp_enc_level4_h], 1)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3_l = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3_l, out_enc_level3_h], 1)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2_l = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2_l, out_enc_level2_h], 1)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1_l = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1_l, out_enc_level1_h], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.up1_0(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
