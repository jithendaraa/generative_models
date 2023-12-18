import flax
from flax import linen as nn
from jax import numpy as jnp
import jax


class ChannelShuffle(nn.Module):
    groups: int

    @nn.compact
    def __call__(self, x):
        n, h, w, c = x.shape
        x = x.reshape(n, h, w, self.groups, c // self.groups)
        x = jnp.transpose(x, (0, 1, 2, 4, 3))
        x = x.reshape(n, h, w, -1)
        return x


class ConvBnSilu(nn.Module):
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    name: str = 'convbnsilu'

    @nn.compact
    def __call__(self, x, train: bool):
        kernel_size = (self.kernel_size, ) * 2
        x = nn.Conv(
            features=self.out_channels, 
            kernel_size=kernel_size, 
            padding=(self.padding, self.padding),
            name=self.name
        )(x)
        x = nn.BatchNorm(use_running_average = not train)(x)
        x = nn.swish(x)
        return x


class ResidualBottleneck(nn.Module):
    in_channels: int
    out_channels: int 

    @nn.compact
    def __call__(self, x, train: bool):
        """
            x: (B, H, W, C)
        """
        
        conv1 = nn.Conv(
            features=self.in_channels//2, 
            kernel_size=(3, 3), 
            strides=1, 
            padding=(1, 1), 
            feature_group_count=self.in_channels//2,
            name='res_conv1'
        )

        conv_bn_silu1 = ConvBnSilu(
            self.out_channels // 2,
            kernel_size=1, 
            stride=1, 
            padding=(0, 0),
            name='convbnsilu1'
        )

        conv_bn_silu2 = ConvBnSilu(
            self.in_channels // 2,
            kernel_size=1, 
            stride=1, 
            padding=(0, 0),
            name='convbnsilu2'
        )

        conv2 = nn.Conv(
            features=self.in_channels//2, 
            kernel_size=(3, 3), 
            strides=1, 
            padding=(1, 1), 
            feature_group_count=self.in_channels//2,
            name='res_conv2'
        )

        conv_bn_silu3 = ConvBnSilu(
            self.out_channels // 2,
            kernel_size=1, 
            stride=1, 
            padding=(0, 0),
            name='convbnsilu3'
        )

        chunked_x = jnp.split(x, 2, axis=-1) # chunk along channel axis

        out1 = conv1(chunked_x[0])
        out1 = nn.BatchNorm(use_running_average=not train)(out1)
        out1 = conv_bn_silu1(out1, train=train)

        out2 = conv2(conv_bn_silu2(chunked_x[1], train=train))
        out2 = nn.BatchNorm(use_running_average=not train)(out2)
        out2 = conv_bn_silu3(out2, train=train)

        out = jnp.concatenate((out1, out2), axis=-1) # channel is last axis
        out = ChannelShuffle(2)(out)
        return out


class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x, train: bool):
        """
            x: (B, H, W, C)
        """
        
        conv1 = nn.Conv(
            features=self.in_channels, 
            kernel_size=(3, 3), 
            strides=2, 
            padding=(1, 1), 
            feature_group_count=self.in_channels,
            name='res_downsample_conv1'
        )

        conv_bn_silu1 = ConvBnSilu(
            self.out_channels // 2,
            kernel_size=1, 
            stride=1, 
            padding=(0, 0),
            name='downsample_convbnsilu1'
        )

        conv_bn_silu2 = ConvBnSilu(
            self.out_channels // 2,
            kernel_size=1, 
            stride=1, 
            padding=(0, 0),
            name='downsample_convbnsilu2'
        )

        conv2 = nn.Conv(
            features=self.out_channels//2, 
            kernel_size=(3, 3), 
            strides=2, 
            padding=(1, 1), 
            feature_group_count=self.out_channels//2,
            name='res_conv2'
        )

        conv_bn_silu3 = ConvBnSilu(
            self.out_channels // 2,
            kernel_size=1, 
            stride=1, 
            padding=(0, 0),
            name='downsample_convbnsilu3'
        )

        out1 = conv1(x)
        out1 = nn.BatchNorm(use_running_average=not train, name='downsample_bn1')(out1)
        out1 = conv_bn_silu1(out1, train=train)
        
        out2 = conv2(conv_bn_silu2(x, train=train))
        out2 = nn.BatchNorm(use_running_average=not train, name='downsample_bn2')(out2)
        out2 = conv_bn_silu3(out2, train=train)

        out = jnp.concatenate((out1, out2), axis=-1) # channel is last axis
        out = ChannelShuffle(2)(out)
        return out


class TimeMLP(nn.Module):
    embedding_dim: int
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x, t):
        linear1 = nn.Dense(self.hidden_dim)
        linear2 = nn.Dense(self.out_dim)
        t_emb = linear2(nn.swish(linear1(t)))[:, None, None, :]
        x = x + t_emb
        return nn.swish(x)


class EncoderBlock(nn.Module):
    in_channels: int
    out_channels: int
    time_embedding_dim: int

    @nn.compact
    def __call__(self, x, t, train):
        res_bottleneck1 = ResidualBottleneck(self.in_channels, self.in_channels)
        res_bottleneck2 = ResidualBottleneck(self.in_channels, self.in_channels)
        res_bottleneck3 = ResidualBottleneck(self.in_channels, self.in_channels)
        res_bottleneck4 = ResidualBottleneck(self.in_channels, self.out_channels // 2)

        x_shortcut = res_bottleneck1(x, train)
        x_shortcut = res_bottleneck2(x_shortcut, train)
        x_shortcut = res_bottleneck3(x_shortcut, train)
        x_shortcut = res_bottleneck4(x_shortcut, train)

        if t is not None:
            time_mlp = TimeMLP(
                embedding_dim=self.time_embedding_dim,
                hidden_dim=self.out_channels,
                out_dim=self.out_channels // 2
            )
            x = time_mlp(x_shortcut, t)
        else:
            import pdb; pdb.set_trace()
        
        conv1 = ResidualDownsample(self.out_channels // 2, self.out_channels)
        x = conv1(x, train)
        return [x, x_shortcut]


class DecoderBlock(nn.Module):
    in_channels: int
    out_channels: int
    time_embedding_dim: int

    @nn.compact
    def __call__(self, x, x_shortcut, train, t=None):
        b, h, w, c = x.shape
        resized_shape = (b, h * 2, w * 2, c)
        x = jax.image.resize(x, shape=resized_shape, method='bilinear')
        x = jnp.concatenate((x, x_shortcut), axis=-1) # concat along channel dim
        
        x = ResidualBottleneck(self.in_channels, self.in_channels)(x, train)
        x = ResidualBottleneck(self.in_channels, self.in_channels)(x, train)
        x = ResidualBottleneck(self.in_channels, self.in_channels)(x, train)
        x = ResidualBottleneck(self.in_channels, self.in_channels // 2)(x, train)

        if t is not None:
            time_mlp=TimeMLP(
                embedding_dim = self.time_embedding_dim,
                hidden_dim = self.in_channels,
                out_dim = self.in_channels//2
            )
            x = time_mlp(x,t)

        conv1 = ResidualBottleneck(self.in_channels//2, self.out_channels//2)
        x = conv1(x, train)
        return x


class UNet(nn.Module):
    timesteps: int
    time_embedding_dim: int
    in_channels: int = 3
    out_channels: int = 2
    base_dim: int = 32
    dim_mults: tuple = (2,4,8,16)

    def setup(self):
        assert isinstance(self.dim_mults,(list,tuple))
        assert self.base_dim % 2 == 0
        self.channels=self._cal_channels(self.base_dim, self.dim_mults)

    @nn.compact
    def __call__(self, x, t, train):
        init_conv = ConvBnSilu(
            out_channels=self.base_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            name='unet_init_conv'
        )

        x = init_conv(x, train=train)
        if t is not None:
            time_embedding = nn.Embed(self.timesteps, self.time_embedding_dim)
            t = time_embedding(t)

        encoder_blocks = [EncoderBlock(c[0], c[1], self.time_embedding_dim) for c in self.channels]        
        encoder_shortcuts = []
        for encoder_block in encoder_blocks:
            x, x_shortcut = encoder_block(x, t, train)
            encoder_shortcuts.append(x_shortcut)

        x = ResidualBottleneck(self.channels[-1][1], self.channels[-1][1])(x, train)
        x = ResidualBottleneck(self.channels[-1][1], self.channels[-1][1])(x, train)
        x = ResidualBottleneck(self.channels[-1][1], self.channels[-1][1] // 2)(x, train)

        decoder_blocks = [DecoderBlock(c[1] ,c[0], self.time_embedding_dim) for c in self.channels[::-1]]
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, train, t)
        
        final_conv = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            strides=1,
            padding=0,
            name='final_conv'
        )
        x = final_conv(x)
        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [ base_dim*x for x in dim_mults ]
        dims.insert(0, base_dim)
        channels=[]
        for i in range(len(dims)-1):
            channels.append((dims[i],dims[i+1])) # in_channel, out_channel
        return channels


if __name__=="__main__":
    pass
    