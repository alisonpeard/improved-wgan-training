# %%
import os, sys
import time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

from dotenv import load_dotenv, find_dotenv
load_dotenv()

# %%
MODE = 'wgan-gp'                        # dcgan, wgan, or wgan-gp
DIM = 64                                # Model dimensionality
BATCH_SIZE = 50                         # Batch size
CRITIC_ITERS = 5                        # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10                             # Gradient penalty lambda hyperparameter
ITERS =int(os.getenv('ITERS'))          # How many generator iterations to train for 
GUMBEL = bool(os.getenv('GUMBEL'))
CHANNELS = int(os.getenv('CHANNELS'))
OUTPUT_DIM = 784 * CHANNELS             # Number of pixels in MNIST (28*28)

# Dataset iterator
# train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
# train_gen, dev_gen, test_gen = lib.mnist.load2(BATCH_SIZE, BATCH_SIZE) # uniform single channel
# train_gen, dev_gen, test_gen = lib.mnist.load3(BATCH_SIZE, BATCH_SIZE)   # Gumbel two channel
train_gen, dev_gen, test_gen = lib.mnist.load4(BATCH_SIZE, BATCH_SIZE)   # Uniform two channel
print(next(train_gen())[0].shape)

def inf_train_gen():
    while True:
        for images, targets in train_gen():
            yield images

tf.compat.v1.disable_eager_execution() # NB: after loading data
lib.print_model_settings(locals().copy())

# %%
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


def Generator(n_samples, noise=None):
    """NCHW"""
    if noise is None:
        if GUMBEL:
            noise = tf.random.uniform([n_samples, 128], minval=0, maxval=0.9999)
            noise = -tf.math.log(-tf.math.log(noise))
        else:
            noise = tf.random.normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7] # some sort of clipping, why?

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, CHANNELS, 5, output)
    if not GUMBEL:
        output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
    """NCHW"""
    output = tf.reshape(inputs, [-1, CHANNELS, 28, 28])
    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)
    return tf.reshape(output, [-1])

# %%
real_data = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

fake_data = Generator(BATCH_SIZE)
disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# %%
if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random.uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )

    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.compat.v1.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
if GUMBEL:
    fixed_noise = tf.constant(np.random.uniform(size=(128, 128), minval=0, maxval=0.9999).astype('float32'))
    fixed_noise = -tf.math.log(-tf.math.log(fixed_noise))
else:
    fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28, CHANNELS))[..., 0], # only save the winds for no
        'imgs/samples_{}.png'.format(frame)
    )
    np.savez('arrs/latest_sample.npz'.format(frame), samples=samples)

# %% Train loop
print(tf.config.list_physical_devices('GPU')) # check GPU being used
with tf.compat.v1.Session() as session:
    print("Starting session...")
    session.run(tf.compat.v1.initialize_all_variables())
    saver = tf.compat.v1.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2) # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    gen = inf_train_gen()

    for iteration in range(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            print("Iteration: ", iteration)
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()

    # Save model weights
    print("Saving model weights...")
    saver.save(session, 'wts/model_weights')

print("Done!")

# %%
# # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
# with tf.Session() as sess:    
#     saver = tf.train.import_meta_graph('my-model-1000.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./'))
#     print(sess.run('w1:0'))
# %%