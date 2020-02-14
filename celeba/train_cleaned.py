import os
import time
import tensorflow as tf
import sys
from functools import partial

import models_cleaned
#from celebA.feeder import Feeder
import numpy as np
import torch
import torchvision.datasets as tvds
import torchvision.transforms as tvtr

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("gpu", 0, "which gpu to use")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("fake_dde_iter", 5, "how many dde steps per generator step")
flags.DEFINE_integer("real_dde_iter", 5, "how many dde steps per generator step")
flags.DEFINE_integer("steps_per_chpt", 1000, "")
flags.DEFINE_integer("epochs", 50, "")
flags.DEFINE_integer("crop_size", 128,
                     "size of image crops (texture resolution is 64^3)")
flags.DEFINE_float("sigma_real", .5, "DAE sigma")
flags.DEFINE_float("sigma_fake", .5, "DAE sigma")
flags.DEFINE_float("real_dde_lr", 5e-3, "learning rate dde")
flags.DEFINE_float("fake_dde_lr", 5e-3, "learning rate dde")
flags.DEFINE_float("g_lr", 3e-3, "learning rate generator")
flags.DEFINE_string("g_act", "lrelu", "must be in ['tanh', 'lrelu', 'softplus', 'relu', 'swish']")
flags.DEFINE_string("dde_act", "softplus", "must be in ['tanh', 'lrelu', 'softplus', 'relu', 'swish']")
flags.DEFINE_bool("linear", False, "whether to use linear or [low, high] tanh output in generator")
flags.DEFINE_bool("clip", True, "whether to use unbound linear or clip output in generator (only useful if linear)")
flags.DEFINE_bool("normalize", False, "whether to normalize g output to [0, 1] (instead of clipping).only makes "
                                      "sense when using linear output")
flags.DEFINE_bool("normalize_data", False, "")
flags.DEFINE_string("chpt_dir", "/output/dde-celeba/chpts",
                    "where to store checkpoints")
flags.DEFINE_string("sum_dir",  "/output/dde-celeba/summaries",
                    "where to store summaries")
flags.DEFINE_bool("resume", False, "whether to resume or train from scratch")
flags.DEFINE_string("resume_chpt", "", "chpt file to resume from")
flags.DEFINE_string("real_dde_chpt", "", "chpt file to load for pretrained real dde")
flags.DEFINE_string("dataset_dir",
                    '/data/datasets/celeba_32/train',
                    'directory with training data')
flags.DEFINE_string('resume_data_dir', "", "leave empty, error if missing!")
flags.DEFINE_bool("test_data", False, "")
FLAGS = flags.FLAGS


def print_over(string):
    sys.stdout.write('\r' + string + ' ' * 30)
    sys.stdout.flush()


def swish(x):
    return x * tf.nn.sigmoid(x)


def leaky_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y


def l2(x, x_):
    return tf.nn.l2_loss(x - x_)

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        y = tensor
        y -= torch.min(y)
        y /= torch.max(y)
        return y

def train():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85, allow_growth=True)
    clip_summaries = False

    with tf.device('/gpu:' + str(FLAGS.gpu)):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            # inputs
            gen_input = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 32])
            real_img = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 32, 32, 3])
            # models
            dde = models_cleaned.dde_32
            g = models_cleaned.generator_32

            # generator
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

            fake_img = g(gen_input, FLAGS.batch_size, scope='generator', act=act, linear=FLAGS.linear,
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

            noisy_real_img = real_img + tf.random_normal(shape=tf.shape(real_img), mean=0., stddev=FLAGS.sigma_real)
            noisy_fake_img = fake_img + tf.random_normal(shape=tf.shape(fake_img), mean=0., stddev=FLAGS.sigma_fake)

            # real dde (real batch)
            log_p_real = dde(noisy_real_img, act=dde_act, scope='real_dde', reuse=False)
            grad_log_p_real = tf.gradients(log_p_real, noisy_real_img)[0]
            real_dae_out_real = noisy_real_img + grad_log_p_real * FLAGS.sigma_real ** 2

            # real dde (fake batch)
            log_p_fake = dde(noisy_fake_img, act=dde_act, scope='real_dde', reuse=True)
            # fake dde (fake batch)
            log_q_fake = dde(noisy_fake_img, act=dde_act, scope='fake_dde', reuse=False)
            grad_log_q_fake = tf.gradients(log_q_fake, noisy_fake_img)[0]
            fake_dae_out_fake = noisy_fake_img + grad_log_q_fake * FLAGS.sigma_fake ** 2

            # fake dde (real batch)
            log_q_real = dde(noisy_real_img, act=dde_act, scope='fake_dde', reuse=True)
            grad_log_q_real = tf.gradients(log_q_real, noisy_real_img)[0]

            # losses
            fake_dde_rec_loss = l2(fake_dae_out_fake, fake_img)
            real_dde_rec_loss = l2(real_dae_out_real, real_img)
            fake_dde_loss = fake_dde_rec_loss
            real_dde_loss = real_dde_rec_loss

            kl_loss = tf.reduce_sum(log_q_fake - log_p_fake)
            gen_loss = kl_loss

            if FLAGS.linear and not FLAGS.normalize:
                clip_summaries = True

            # train steps
            fake_dde_vars = tf.trainable_variables(scope='fake_dde')
            fake_dde_train_step = tf.train.AdamOptimizer(
                FLAGS.fake_dde_lr, beta1=0.9, beta2=0.999).minimize(fake_dde_loss, var_list=fake_dde_vars)
            real_dde_vars = tf.trainable_variables(scope='real_dde')
            real_dde_train_step = tf.train.AdamOptimizer(
                FLAGS.real_dde_lr, beta1=0.9, beta2=0.999).minimize(real_dde_loss, var_list=real_dde_vars)
            gen_vars = tf.trainable_variables(scope='generator')
            gen_train_step = tf.train.AdamOptimizer(
                FLAGS.g_lr, beta1=0.9, beta2=0.999).minimize(gen_loss, var_list=gen_vars)

            # summaries
            fake_dde_summaries = [tf.summary.scalar('fake_dde_rec_loss', fake_dde_rec_loss),
                                  tf.summary.image('fake_dae_input', noisy_fake_img),
                                  tf.summary.image('fake_dae_output', fake_dae_out_fake)]
            real_dde_summaries = [tf.summary.scalar('real_dde_rec_loss', real_dde_rec_loss),
                                  tf.summary.image('real_dae_input', noisy_real_img),
                                  tf.summary.image('real_dae_output', real_dae_out_real),
                                  tf.summary.histogram('noisy_real_img', noisy_real_img),
                                  tf.summary.histogram('real_img', real_img)]

            if clip_summaries:
                fake_img_clip = tf.clip_by_value(fake_img, -0.5, 0.5)
                fake_img_sum = tf.summary.image('fake_img', fake_img_clip)
            else:
                fake_img_sum = tf.summary.image('fake_img', fake_img)
            gen_summaries = [tf.summary.scalar('kl_loss', kl_loss),
                             fake_img_sum,
                             tf.summary.histogram('fake_img', fake_img)]
            if clip_summaries:
                gen_summaries.append(tf.summary.histogram('fake_img_clipped', fake_img_clip))

            # saver for chpts (resuming)
            saver = tf.train.Saver(max_to_keep=None)
            # initialize variables
            if FLAGS.resume:
                latest_chkpt = FLAGS.resume_chpt
                if latest_chkpt is not None:
                    print('')
                    print("resuming from " + latest_chkpt + "\n"
                          + "saving in " + FLAGS.chpt_dir)
                    print('')
                    saver.restore(sess, latest_chkpt)
                else:
                    print('')
                    print("starting from scratch\n"
                          + "saving in " + FLAGS.chpt_dir)
                    print('')
                    sess.run(tf.initialize_all_variables())
            else:
                print('')
                print("starting from scratch\n"
                      + "saving in " + FLAGS.chpt_dir)
                print('')
                sess.run(tf.initialize_all_variables())

            # load pretrained real dde
            if FLAGS.real_dde_chpt is not "":
                saver_real_dde = tf.train.Saver(var_list=real_dde_vars, max_to_keep=None)
                print("loading real dde checkpoint from " + FLAGS.real_dde_chpt)
                saver_real_dde.restore(sess, FLAGS.real_dde_chpt)

            center_transform = tvtr.Lambda(lambda x: x - 0.5)
            if FLAGS.normalize and FLAGS.normalize_data:
                print('NORMALIZE DATA')
                transform = tvtr.Compose([tvtr.ToTensor(), Normalize(), center_transform])
            else:
                transform = tvtr.Compose([tvtr.ToTensor(), center_transform])

            dataset = tvds.ImageFolder(FLAGS.dataset_dir, transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=8)
            loader_enumerator = enumerate(loader)

            get_noise_batch_op = tf.random_normal(tf.shape(gen_input), mean=0., stddev=1.)

            # start feeder queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # summary writer
            sum_writer = tf.summary.FileWriter(FLAGS.sum_dir, sess.graph)

            # train
            progress_interval = 10
            steps_per_epoch = len(dataset) // FLAGS.batch_size
            try:
                total_steps = 0

                # foreach epoch
                for epoch in range(1, FLAGS.epochs + 1):
                    epoch_start = time.time()
                    #epoch += 1
                    step = 0

                    # foreach iteration
                    while step < steps_per_epoch:
                        # fake dde train step
                        for i in range(FLAGS.fake_dde_iter):
                            # get batch
                            noise_batch = sess.run(get_noise_batch_op)

                            if step % 10 == 0 and i == 0:
                                run_results = sess.run(fake_dde_summaries + [fake_dde_train_step],
                                                       feed_dict={gen_input: noise_batch})

                                # write summaries
                                for s in run_results[:len(fake_dde_summaries)]:
                                    sum_writer.add_summary(s, total_steps)
                            else:
                                sess.run(fake_dde_train_step,
                                         feed_dict={gen_input: noise_batch})

                        # real dde train step
                        for i in range(FLAGS.real_dde_iter):
                            # get batch
                            _, batch_data = next(loader_enumerator, None)
                            if batch_data is None or len(batch_data[0]) < FLAGS.batch_size:
                                loader_enumerator = enumerate(loader)
                                _, batch_data = next(loader_enumerator)

                            img_batch = batch_data[0].numpy()
                            img_batch = np.transpose(img_batch, [0, 2, 3, 1])

                            if step % 10 == 0 and i == 0:
                                run_results = sess.run(real_dde_summaries + [real_dde_train_step],
                                                       feed_dict={real_img: img_batch})
                                # write summaries
                                for s in run_results[:len(real_dde_summaries)]:
                                    sum_writer.add_summary(s, total_steps)

                            else:
                                sess.run(real_dde_train_step, feed_dict={real_img: img_batch})

                        # generator train step
                        if step % 10 == 0:
                            run_results = sess.run(gen_summaries + [gen_train_step],
                                                   feed_dict={gen_input: noise_batch})
                            # write summaries
                            for s in run_results[:len(gen_summaries)]:
                                sum_writer.add_summary(s, total_steps)
                        else:
                            sess.run(gen_train_step, feed_dict={gen_input: noise_batch})

                        step += 1
                        total_steps += 1

                        # monitor progress
                        if step % progress_interval == 0:
                            print_over('e' + str(epoch) + ': ' + str(round(step * 100 / steps_per_epoch))
                                       + '% completed')
                        # store chpt
                        if total_steps % FLAGS.steps_per_chpt == 0:
                            print('saving checkpoint')
                            saver.save(sess, FLAGS.chpt_dir + '/checkpoint_e'
                                       + str(epoch) + '-' + str(total_steps))

                    print('epoch ' + str(epoch) + ' finished, took '
                          + str(round((time.time() - epoch_start) / 60)) + ' min.')

            except tf.errors.OutOfRangeError:
                print('Training done, epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

def test_data():
    center_transform = tvtr.Lambda(lambda x: x - 0.5)
    if FLAGS.normalize and FLAGS.normalize_data:
        print('NORMALIZE DATA')
        transform = tvtr.Compose([tvtr.ToTensor(), Normalize(), center_transform])
    else:
        transform = tvtr.Compose([tvtr.ToTensor(), center_transform])

    dataset = tvds.ImageFolder(FLAGS.dataset_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=8)
    loader_enumerator = enumerate(loader)

    _, batch_data = next(loader_enumerator, None)


def main(_):
    # create directories for data and summaries if necessary
    if not os.path.exists(FLAGS.chpt_dir):
        FLAGS.data_dir += '/run_00'
        os.makedirs(FLAGS.chpt_dir)
        if FLAGS.resume_data_dir == "":
            FLAGS.resume_data_dir = FLAGS.chpt_dir
    else:
        # find last run number
        dirs = next(os.walk(FLAGS.chpt_dir))[1]
        for idx, d in enumerate(dirs):
            dirs[idx] = d[-2:]

        runs = sorted(map(int, dirs))
        run_nr = 0
        if len(runs) > 0:
            run_nr = runs[-1] + 1

        prefix = FLAGS.chpt_dir
        FLAGS.chpt_dir = prefix + '/run_' + str(run_nr).zfill(2)
        os.makedirs(FLAGS.chpt_dir)

        if FLAGS.resume_data_dir == "":
            FLAGS.resume_data_dir = prefix + '/run_' + str(run_nr - 1).zfill(2)

    if not os.path.exists(FLAGS.sum_dir):
        FLAGS.sum_dir += '/run_00'
        os.makedirs(FLAGS.sum_dir)
    else:
        # find last run number
        dirs = next(os.walk(FLAGS.sum_dir))[1]
        for idx, d in enumerate(dirs):
            dirs[idx] = d[-2:]

        runs = sorted(map(int, dirs))
        run_nr = 0
        if len(runs) > 0:
            run_nr = runs[-1] + 1

        FLAGS.sum_dir = FLAGS.sum_dir + '/run_' + str(run_nr).zfill(
            2)
        os.makedirs(FLAGS.sum_dir)

    # train
    train()


if __name__ == "__main__":
    if FLAGS.test_data:
        test_data()
    else:
        tf.app.run()