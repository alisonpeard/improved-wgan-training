import numpy
import os
import urllib
import gzip
import pickle

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data
    n, hwc = images.shape

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS").format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, hwc)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print ("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.request.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f) # https://github.com/MichalDanielDobrzanski/DeepLearningPython/issues/15#issuecomment-689788837
        u.encoding = 'latin1'
        train_data, dev_data, test_data = u.load()

    print("Train data shape: {}".format(train_data[0].shape))
    print("Dev data shape: {}".format(dev_data[0].shape))
    print("Test data shape: {}".format(test_data[0].shape))

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )


def load2(batch_size, test_batch_size, im_size=(28, 28), n_labelled=None):
    """For single channel (wind) uniform data."""
    import tensorflow as tf
    filepath = 'train/data_pretrain.npz'
    data = numpy.load(filepath)['arr_0']
    data = data[..., numpy.newaxis]
    data = tf.image.resize(data, im_size)
    data = data.numpy()
    data = data.reshape([data.shape[0], 784, 1])
    numpy.random.shuffle(data) # only shuffle first dimension

    train_data = data[:50000, ...]
    dev_data = data[50000:60000, ...]
    test_data = data[60000:70000, ...]

    train_data = (train_data, numpy.max(train_data, axis=(1,2)))
    dev_data = (dev_data, numpy.max(dev_data, axis=(1,2)))
    test_data = (test_data, numpy.max(test_data, axis=(1,2)))

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )

def load3(batch_size, test_batch_size, im_size=(28, 28), n_labelled=None):
    """For two-channel (wind, tp) Gumbel data."""
    import tensorflow as tf
    filepath = 'train/gumbel.npz'
    data = numpy.load(filepath)['data']
    data = tf.image.resize(data, im_size)
    data = data.numpy()
    data = data.reshape([data.shape[0], 784, 2])
    numpy.random.shuffle(data) # only shuffles first dimension

    train_data = data[:50000, ...]
    dev_data = data[50000:60000, ...]
    test_data = data[60000:70000, ...]

    train_data = (train_data, numpy.max(train_data[..., 0], axis=(1)))
    dev_data = (dev_data, numpy.max(dev_data[..., 0], axis=(1)))
    test_data = (test_data, numpy.max(test_data[..., 0], axis=(1)))

    _, hw,c = train_data[0].shape
    train_data = (train_data[0].reshape(50000, hw*c), train_data[1])
    dev_data = (dev_data[0].reshape(10000, hw*c), dev_data[1])
    test_data = (test_data[0].reshape(10000, hw*c), test_data[1])

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )


def load4(batch_size, test_batch_size, im_size=(28, 28), n_labelled=None):
    """For two-channel (wind, tp) Gumbel data."""
    import tensorflow as tf
    filepath = 'train/uniform.npz'
    data = numpy.load(filepath)['data']
    data = tf.image.resize(data, im_size)
    data = data.numpy()
    data = data.reshape([data.shape[0], 784, 2])
    numpy.random.shuffle(data) # only shuffles first dimension

    train_data = data[:50000, ...]
    dev_data = data[50000:60000, ...]
    test_data = data[60000:70000, ...]

    train_data = (train_data, numpy.max(train_data[..., 0], axis=(1)))
    dev_data = (dev_data, numpy.max(dev_data[..., 0], axis=(1)))
    test_data = (test_data, numpy.max(test_data[..., 0], axis=(1)))

    _, hw,c = train_data[0].shape
    train_data = (train_data[0].reshape(50000, hw*c), train_data[1])
    dev_data = (dev_data[0].reshape(10000, hw*c), dev_data[1])
    test_data = (test_data[0].reshape(10000, hw*c), test_data[1])

    return (
        mnist_generator(train_data, batch_size, n_labelled), 
        mnist_generator(dev_data, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_batch_size, n_labelled)
    )