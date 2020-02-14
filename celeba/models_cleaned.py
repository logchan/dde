import tensorflow as tf
import numpy as np


# pixel-wise norm
def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    # our format is (bs, h, w, c), not (bs, c, h, w)
    if tf.rank(x) == 4:
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)
    else:
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def dde_32(t_in, act=None, reuse=True, scope='dde', eq_lr=True):
    """
    maps a noisy input image t_in to log_p(t_in)
    :param scope: scope name
    :param t_in: noisy input image, 32x32x3
    :param reuse:
    :return: log_p
    """
    if act is None:
        act = tf.nn.softplus

    def _bias(x, layer_idx, lr_mul=1., conv=True):
        if conv:
            b = tf.get_variable('conv_bias_' + str(layer_idx),
                                shape=[x.shape[-1].value], initializer=tf.initializers.zeros()) * lr_mul
            return x + tf.reshape(b, [1, 1, 1, -1])
        else:
            b = tf.get_variable('fc_bias_' + str(layer_idx),
                                shape=[x.shape[-1].value], initializer=tf.initializers.zeros()) * lr_mul
            return x + b

    def _get_weight(s, layer_idx, conv=True, gain=np.sqrt(2), lr_mul=1.):
        # he init
        fan_in = np.prod(s[:-1])    # s is [k_s, k_s, n_chan_in, n_chan_out]
        he_stddev = gain / np.sqrt(fan_in)

        if eq_lr:
            init_stddev = 1. / lr_mul
            rt_coeff = he_stddev * lr_mul
        else:
            init_stddev = he_stddev / lr_mul
            rt_coeff = lr_mul

        initializer = tf.initializers.random_normal(0, init_stddev)
        if conv:
            return tf.get_variable('conv_weights_' + str(layer_idx),
                                   shape=s, initializer=initializer, dtype=tf.float32) * rt_coeff
        else:
            return tf.get_variable('fc_weights_' + str(layer_idx),
                                   shape=s, initializer=initializer, dtype=tf.float32) * rt_coeff

    def conv2d(t_in, n_chan, layer_idx=-1, kernel_size=3, stride=1, gain=np.sqrt(2)):
        w = _get_weight([kernel_size, kernel_size, t_in.shape[-1].value, n_chan], layer_idx, gain=gain)

        return tf.nn.conv2d(t_in, w, strides=[stride, stride], padding='SAME', data_format="NHWC")

    def fc(t_in, n_out, layer_idx=-1, gain=np.sqrt(2), lr_mul=1.):
        w = _get_weight([t_in.shape[-1].value, n_out], layer_idx, conv=False, gain=gain, lr_mul=lr_mul)

        return tf.matmul(t_in, w)

    def downsample(t_in, factor=2):
        t_out = tf.nn.avg_pool(t_in, [1, factor, factor, 1], [1, factor, factor, 1], padding='VALID',
                               data_format='NHWC')
        return t_out

    def post_conv(t_in, layer_idx=-1):
        t_out = _bias(t_in, layer_idx)
        t_out = act(t_out)
        #t_out = _instance_norm(t_out)

        return t_out

    with tf.variable_scope(scope, reuse=reuse):
        l1 = conv2d(t_in, 32, layer_idx=0)  # bsx32x32x32
        l1 = post_conv(l1, layer_idx=0)

        l2 = conv2d(l1, 64, layer_idx=1)  # bsx32x32x64
        l2 = post_conv(l2, layer_idx=1)
        l2 = downsample(l2, factor=2)  # bsx16x16x64

        l3 = conv2d(l2, 64, layer_idx=3)  # bsx16x16x64
        l3 = post_conv(l3, layer_idx=3)

        l4 = conv2d(l3, 128, layer_idx=4)  # bsx16x16x128
        l4 = post_conv(l4, layer_idx=4)
        l4 = downsample(l4, factor=2)  # bsx8x8x128

        l5 = conv2d(l4, 128, layer_idx=5)  # bsx8x8x128
        l5 = post_conv(l5, layer_idx=5)

        l6 = conv2d(l5, 256, layer_idx=6)  # bsx8x8x256
        l6 = post_conv(l6, layer_idx=6)
        l6 = downsample(l6, factor=2)  # bsx4x4x256

        l7 = conv2d(l6, 256, layer_idx=7)  # bsx4x4x256
        l7 = post_conv(l7, layer_idx=7)

        l8 = conv2d(l7, 512, layer_idx=8)  # bsx4x4x512
        l8 = post_conv(l8, layer_idx=8)
        l8 = downsample(l8, factor=2)  # bsx2x2x512

        l9 = conv2d(l8, 512, layer_idx=9)  # bsx2x2x512
        l9 = post_conv(l9, layer_idx=9)

        conv_out = tf.reshape(l9, [-1, 2*2*512])
        l10 = fc(conv_out, 512, layer_idx=10)
        l10 = act(_bias(l10, layer_idx=10, conv=False))

        l11 = fc(l10, 1, layer_idx=11, gain=1.)
        log_p = _bias(l11, layer_idx=11, conv=False)

    return log_p


def generator_32(t_in, bs, noises=None, avg_styles=None, psi=1., reuse=True, return_styler=False,
                             scope='generator', act=tf.nn.softplus, linear=False, clip=False, norm=False,
                             eq_lr=True):
    """
    maps Gaussian noise to fake image
    :param noises: array of noises foreach layer
    :param bs: batch size
    :param t_in: 32d Gaussian noise
    :param reuse:
    :param scope: scope name
    :return: 32x32x3 fake image
    """

    def _bias(x, layer_idx, lr_mul=1., conv=True):
        if conv:
            b = tf.get_variable('conv_bias_' + str(layer_idx),
                                shape=[x.shape[-1].value], initializer=tf.initializers.zeros()) * lr_mul
            return x + tf.reshape(b, [1, 1, 1, -1])
        else:
            b = tf.get_variable('fc_bias_' + str(layer_idx),
                                shape=[x.shape[-1].value], initializer=tf.initializers.zeros()) * lr_mul
            return x + b

    def _instance_norm(t_in, eps=1e-8):
        """
        normalizes each channel to mean/stddev over all pixels in channel
        """
        # input is bs h, w, c
        m = tf.reduce_mean(t_in, axis=[1, 2], keep_dims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(t_in), axis=[1, 2], keepdims=True) + eps)
        return (t_in - m) / stddev

    def _style_mlp(z, lr_mul=1.):
        latent_dim = z.shape[-1].value

        # normalize latents
        z = pixel_norm(z)

        l1 = fc(z, latent_dim, layer_idx=0, lr_mul=lr_mul)
        l1 = _bias(l1, layer_idx=0, lr_mul=lr_mul, conv=False)
        l1 = act(l1)
        l2 = fc(l1, latent_dim, layer_idx=1, lr_mul=lr_mul)
        l2 = _bias(l2, layer_idx=1, lr_mul=lr_mul, conv=False)
        l2 = act(l2)
        l3 = fc(l2, latent_dim, layer_idx=2, lr_mul=lr_mul)
        l3 = _bias(l3, layer_idx=2, lr_mul=lr_mul, conv=False)
        l3 = act(l3)

        return l3

    def _add_learned_noise(t_in, layer_idx):
        bs, h, w, c = t_in.shape.as_list()
        # same noise foreach channel
        if noises is None:
            noise = tf.random_normal([bs, h, w, 1], mean=0., stddev=1.)
        else:
            noise = noises[layer_idx]
        per_chan_w = tf.get_variable('noise_weight_' + str(layer_idx), trainable=True, shape=[c], initializer=tf.initializers.zeros())

        return t_in + noise * tf.reshape(per_chan_w, [1, 1, 1, -1])

    def _stylize(t_in, a, layer_idx=-1):
        # t_in has shape [bs, h, w, c]
        c = t_in.shape[-1].value
        # a has shape [bs, latent_dim]

        # linearly map (latent_dim values) to 2*c (scale and bias foreach channel in t_in)
        style = fc(a, c * 2, layer_idx=layer_idx+n_layers+1, gain=1.)
        style = _bias(style, layer_idx=layer_idx+n_layers+1, conv=False)
        style = tf.reshape(style, [-1, 2, 1, 1, c])

        return t_in * (style[:, 0, :, :, :]+1) + style[:, 1, :, :, :]

    def _get_weight(s, layer_idx, conv=True, gain=np.sqrt(2), lr_mul=1.):
        # he init
        fan_in = np.prod(s[:-1])    # s is [k_s, k_s, n_chan_in, n_chan_out]
        he_stddev = gain / np.sqrt(fan_in)

        if eq_lr:
            init_stddev = 1. / lr_mul
            rt_coeff = he_stddev * lr_mul
        else:
            init_stddev = he_stddev / lr_mul
            rt_coeff = lr_mul

        initializer = tf.initializers.random_normal(0, init_stddev)
        if conv:
            return tf.get_variable('conv_weights_' + str(layer_idx),
                                   shape=s, initializer=initializer, dtype=tf.float32) * rt_coeff
        else:
            return tf.get_variable('fc_weights_' + str(layer_idx),
                                   shape=s, initializer=initializer, dtype=tf.float32) * rt_coeff

    def upsample(t_in, factor=2):
        s = t_in.shape
        t_out = tf.reshape(t_in, [-1, s[1], 1, s[2], 1, s[3]])  # [bs, h, 1, w, 1, c]
        t_out = tf.tile(t_out, [1, 1, factor, 1, factor, 1])
        return tf.reshape(t_out, [-1, s[1] * factor, s[2] * factor, s[3]])

    def conv2d(t_in, n_chan, layer_idx=-1, kernel_size=3, stride=1, gain=np.sqrt(2)):
        w = _get_weight([kernel_size, kernel_size, t_in.shape[-1].value, n_chan], layer_idx, gain=gain)

        return tf.nn.conv2d(t_in, w, strides=[stride, stride], padding='SAME', data_format="NHWC")

    def fc(t_in, n_out, layer_idx=-1, gain=np.sqrt(2), lr_mul=1.):
        w = _get_weight([t_in.shape[-1].value, n_out], layer_idx, conv=False, gain=gain, lr_mul=lr_mul)

        return tf.matmul(t_in, w)

    def post_conv(t_in, styler, layer_idx=-1):
        t_out = _add_learned_noise(t_in, layer_idx)
        t_out = _bias(t_out, layer_idx)
        t_out = act(t_out)
        t_out = _instance_norm(t_out)
        t_out = _stylize(t_out, styler, layer_idx=layer_idx)

        return t_out

    with tf.variable_scope(scope, reuse=reuse):
        n_layers = 9    # no noise/style for last layer
        # style affine trans learned from latent input
        styler_ = _style_mlp(t_in, lr_mul=0.01)  # [bs, latent_dim]
        if avg_styles is not None:
            # truncation trick
            avg_styles = tf.tile(avg_styles[np.newaxis, :], [bs, 1])  # broadcast to batch_size
            styler_ = avg_styles + psi * (styler_ - avg_styles)

        # broadcast (use same styler foreach layer)
        styler = tf.tile(styler_[:, np.newaxis], [1, n_layers, 1])    # [bs , n_layers, latent_dim]

        # base layer [bs, 2, 2, 512]
        base = tf.get_variable('gen_base', shape=[1, 2, 2, 512], trainable=True, initializer=tf.initializers.ones())
        # replicate foreach batch
        base = tf.tile(base, [bs, 1, 1, 1])
        # conv
        base = conv2d(base, 512, layer_idx=0, kernel_size=2)
        base = post_conv(base, styler[:, 0, :], layer_idx=0)

        l1 = upsample(base, factor=2)    # [bs, 4, 4, 512]
        l1 = conv2d(l1, 256, layer_idx=1)    # [bs, 4, 4, 256]
        l1 = post_conv(l1, styler[:, 1, :], layer_idx=1)

        l2 = conv2d(l1, 256, layer_idx=2)    # [bs, 4, 4, 256]
        l2 = post_conv(l2, styler[:, 2, :], layer_idx=2)

        l3 = upsample(l2, factor=2)  # [bs, 8, 8, 256]
        l3 = conv2d(l3, 128, layer_idx=3)    # [bs, 8, 8, 128]
        l3 = post_conv(l3, styler[:, 3, :], layer_idx=3)

        l4 = conv2d(l3, 128, layer_idx=4)  # [bs, 8, 8, 128]
        l4 = post_conv(l4, styler[:, 4, :], layer_idx=4)

        l5 = upsample(l4, factor=2)   # [bs, 16, 16, 128]
        l5 = conv2d(l5, 64, layer_idx=5)     # [bs, 16, 16, 64]
        l5 = post_conv(l5, styler[:, 5, :], layer_idx=5)

        l6 = conv2d(l5, 64, layer_idx=6)  # [bs, 16, 16, 64]
        l6 = post_conv(l6, styler[:, 6, :], layer_idx=6)

        l7 = upsample(l6, factor=2)   # [bs, 32, 32, 64]
        l7 = conv2d(l7, 32, layer_idx=7)   # [bs, 32, 32, 32]
        l7 = post_conv(l7, styler[:, 7, :], layer_idx=7)

        l8 = conv2d(l7, 32, layer_idx=8)  # [bs, 32, 32, 32]
        l8 = post_conv(l8, styler[:, 8, :], layer_idx=8)

        # 2rgb: no noise/style here...
        out = _bias(conv2d(l8, 3, layer_idx=9, kernel_size=1, gain=1.), 9)
        if linear:
            if norm:
                out -= tf.reduce_min(out, axis=[1, 2, 3], keepdims=True)
                out /= tf.reduce_max(out, axis=[1, 2, 3], keepdims=True)
                out -= 0.5
            else:
                if clip:
                    out = tf.clip_by_value(out, -1., 1.)

            if return_styler:
                return out, styler_
            else:
                return out

        # data is in [-0.5, 0.5]
        out = tf.nn.tanh(out) * 0.5

    if return_styler:
        return out, styler_
    else:
        return out
