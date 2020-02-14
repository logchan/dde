import numpy as np
import tensorflow as tf
import cv2
from functools import partial

import models_cleaned

flags = tf.flags
logging = tf.logging

flags.DEFINE_float("sigma", .5, "fake dde sigma")
flags.DEFINE_integer("gpu", 0, "which gpu to use")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("seed", -1, "seed for reproducibility, -1 for random")
flags.DEFINE_integer("crop_size", 128,
                     "size of image crops (texture resolution is 64^3)")
flags.DEFINE_string("model_name", "/output/dde-celeba/chpts/run_00/checkpoint_e50-158250", "chpt file to load")
flags.DEFINE_string("avg_styles", "", "styles centroid")
flags.DEFINE_string("out_file",
                    '/output/dde-celeba/dde_results.png',
                    'out fn')
flags.DEFINE_float("trun_factor", 1., "truncation strength (psi), should be in [-1, 1], 0 is avg face,"
                                      "1/-1 is untruncated")
flags.DEFINE_string("dde_act", "softplus", "must be in ['tanh', 'lrelu', 'softplus', 'relu', 'swish']")
flags.DEFINE_string("g_act", "lrelu", "must be in ['tanh', 'lrelu', 'softplus', 'relu', 'swish']")
flags.DEFINE_bool("linear", False, "whether to use linear or [low, high] tanh output in generator")
flags.DEFINE_bool("clip", True, "whether to use unbound linear or clip output in generator (only useful if linear)")
flags.DEFINE_bool("normalize", False, "whether to normalize g output to [0, 1] (instead of clipping).only makes "
                                      "sense when using linear output")
flags.DEFINE_bool("store_single", False, "")
flags.DEFINE_string("out_single_file",
                    '/output/dde-celeba/images/%05d.png',
                    'out fn')
flags.DEFINE_integer("id_offset", 0, "")

FLAGS = flags.FLAGS


def swish(x):
    return x * tf.nn.sigmoid(x)


def leaky_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y


def imgrid(img_batch):
    n_x = int(img_batch ** 0.5)
    n_y = n_x
    img_size = img_batch.shape[1]
    img_grid = np.zeros((img_size*n_y, img_size*n_x, 3), dtype=np.float32)
    for idx, img in enumerate(img_batch):
        i = idx % n_x
        j = idx // n_x
        img_grid[j * img_size: j * img_size + img_size, i * img_size:i * img_size + img_size, :] = img
    return img_grid


def get_noise_placeholders(bs):
    passphs = [
        tf.placeholder(tf.float32, shape=[bs, 2, 2, 1]),
        tf.placeholder(tf.float32, shape=[bs, 4, 4, 1]),
        tf.placeholder(tf.float32, shape=[bs, 4, 4, 1]),
        tf.placeholder(tf.float32, shape=[bs, 8, 8, 1]),
        tf.placeholder(tf.float32, shape=[bs, 8, 8, 1]),
        tf.placeholder(tf.float32, shape=[bs, 16, 16, 1]),
        tf.placeholder(tf.float32, shape=[bs, 16, 16, 1]),
        tf.placeholder(tf.float32, shape=[bs, 32, 32, 1]),
        tf.placeholder(tf.float32, shape=[bs, 32, 32, 1])
    ]
    return passphs


def get_layer_noise(bs, n, zeros=False):
    layer_noise = []
    if zeros:
        layer_noise.append(np.zeros((bs, 2, 2, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 4, 4, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 4, 4, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 8, 8, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 8, 8, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 16, 16, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 16, 16, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 32, 32, 1), dtype=np.float32))
        layer_noise.append(np.zeros((bs, 32, 32, 1), dtype=np.float32))
    else:
        n0 = n.rvs(FLAGS.batch_size * 2*2*1)
        n0 = np.reshape(n0, [bs, 2, 2, 1])
        layer_noise.append(n0)

        n0 = n.rvs(FLAGS.batch_size * 4 * 4 * 1)
        n0 = np.reshape(n0, [bs, 4, 4, 1])
        layer_noise.append(n0)
        n0 = n.rvs(FLAGS.batch_size * 4 * 4 * 1)
        n0 = np.reshape(n0, [bs, 4, 4, 1])
        layer_noise.append(n0)

        n0 = n.rvs(FLAGS.batch_size * 8 * 8 * 1)
        n0 = np.reshape(n0, [bs, 8, 8, 1])
        layer_noise.append(n0)
        n0 = n.rvs(FLAGS.batch_size * 8 * 8 * 1)
        n0 = np.reshape(n0, [bs, 8, 8, 1])
        layer_noise.append(n0)

        n0 = n.rvs(FLAGS.batch_size * 16 * 16 * 1)
        n0 = np.reshape(n0, [bs, 16, 16, 1])
        layer_noise.append(n0)
        n0 = n.rvs(FLAGS.batch_size * 16 * 16 * 1)
        n0 = np.reshape(n0, [bs, 16, 16, 1])
        layer_noise.append(n0)

        n0 = n.rvs(FLAGS.batch_size * 32 * 32 * 1)
        n0 = np.reshape(n0, [bs, 32, 32, 1])
        layer_noise.append(n0)
        n0 = n.rvs(FLAGS.batch_size * 32 * 32 * 1)
        n0 = np.reshape(n0, [bs, 32, 32, 1])
        layer_noise.append(n0)

    return layer_noise


def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

    with tf.device('/gpu:' + str(FLAGS.gpu)):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            # setting seed for reproducibility
            if FLAGS.seed is not -1:
                tf.compat.v1.random.set_random_seed(FLAGS.seed)
                np.random.seed(FLAGS.seed)
                print("setting seed for reproducibility to " + str(FLAGS.seed))

            # inputs
            gen_input = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 32])
            # model
            g = models_cleaned.generator_32
            dde = models_cleaned.dde_32

            # generator
            avg_styles = None
            if FLAGS.avg_styles is not "":
                # load avg styles
                import pickle
                with open(FLAGS.avg_styles, 'rb') as f:
                    avg_styles = pickle.load(f)
                    avg_styles = avg_styles.astype(np.float32)

            noise_ph = get_noise_placeholders(FLAGS.batch_size)

            if FLAGS.g_act == 'lrelu':
                act = partial(leaky_relu, leak=0.2)
            elif FLAGS.g_act == 'relu':
                act = tf.nn.relu
            elif FLAGS.g_act == 'tanh':
                act = tf.nn.tanh
            elif FLAGS.g_act == 'swish':
                act = swish
            else:
                act = tf.nn.softplus

            fake_img = g(gen_input, FLAGS.batch_size, noises=noise_ph, avg_styles=avg_styles,
                         psi=FLAGS.trun_factor, scope='generator', act=act, linear=FLAGS.linear,
                         clip=FLAGS.clip, norm=FLAGS.normalize, reuse=False)

            if FLAGS.dde_act == 'lrelu':
                dde_act = partial(leaky_relu, leak=0.2)
            elif FLAGS.dde_act == 'relu':
                dde_act = tf.nn.relu
            elif FLAGS.dde_act == 'tanh':
                dde_act = tf.nn.tanh
            elif FLAGS.dde_act == 'swish':
                dde_act = swish
            else:
                dde_act = tf.nn.softplus

            # fake dde
            noisy_fake_img = fake_img + tf.random.normal(shape=tf.shape(fake_img), mean=0., stddev=FLAGS.sigma)
            fake_dde = dde(noisy_fake_img, act=dde_act, scope='fake_dde', reuse=False)
            grad_fake_dde = tf.gradients(fake_dde, noisy_fake_img)[0]
            denoised_fake_img = noisy_fake_img + grad_fake_dde * FLAGS.sigma ** 2

            # load model
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, FLAGS.model_name)
            print('loaded model from :' + FLAGS.model_name)

            mu = 0.
            sigma = 1.
            import scipy.stats as stats
            n = stats.norm(loc=mu, scale=sigma)

            # get some samples
            noise_batch = n.rvs(FLAGS.batch_size*32)
            layer_noise = get_layer_noise(FLAGS.batch_size, n)
            noise_batch = np.reshape(noise_batch, (FLAGS.batch_size, 32))
            fake_imgs, fake_imgs_den = sess.run(
                [fake_img, denoised_fake_img],
                feed_dict={gen_input: noise_batch,
                           noise_ph[0]: layer_noise[0],
                           noise_ph[1]: layer_noise[1],
                           noise_ph[2]: layer_noise[2],
                           noise_ph[3]: layer_noise[3],
                           noise_ph[4]: layer_noise[4],
                           noise_ph[5]: layer_noise[5],
                           noise_ph[6]: layer_noise[6],
                           noise_ph[7]: layer_noise[7],
                           noise_ph[8]: layer_noise[8]})

            if FLAGS.store_single:
                print()
                for i in range(FLAGS.batch_size):
                    img = fake_imgs[i]
                    img = cv2.cvtColor(np.clip(img + 0.5, 0, 1), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(FLAGS.out_single_file % (i+1+FLAGS.id_offset), img * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    print('\rwrote %d / %d images' % (i+1, FLAGS.batch_size), end='')
                print()

            else:
                out_img = imgrid(fake_imgs)
                # out_img_den = imgrid(fake_imgs_den)

                # data is in [-0.5, 0.5]
                out_img += 0.5
                # out_img_den += 0.5

                # we store clipped and normalized version
                out_img_norm = out_img - np.min(out_img)
                out_img_norm = out_img_norm / np.max(out_img_norm)
                out_img_norm = cv2.cvtColor(out_img_norm, cv2.COLOR_BGR2RGB)
                out_img_clip = cv2.cvtColor(np.clip(out_img, 0., 1.), cv2.COLOR_BGR2RGB)
                cv2.imwrite(FLAGS.out_file, out_img_clip*255, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                out_file_norm = FLAGS.out_file.replace('.png', '_norm.png')
                cv2.imwrite(out_file_norm, out_img_norm * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                # write denoised
                # tups = FLAGS.out_file.split('/')
                # p = FLAGS.out_file.replace(tups[-1], 'den/' + tups[-1])
                # p = FLAGS.out_file.replace('.png', '_den.png')
                # out_img_den_norm = out_img_den - np.min(out_img_den)
                # out_img_den_norm = out_img_den_norm / np.max(out_img_den_norm)
                # out_img_den_norm = cv2.cvtColor(out_img_den_norm, cv2.COLOR_BGR2RGB)
                # out_img_den_clip = cv2.cvtColor(np.clip(out_img_den, 0., 1.), cv2.COLOR_BGR2RGB)
                # cv2.imwrite(p, out_img_den_clip*255, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                # cv2.imwrite(p.replace('.png', '_norm.png'), out_img_den_norm * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == "__main__":
    tf.app.run()
