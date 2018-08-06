# IMPORT EXTERNAL LIBRAIRES
import tensorflow as tf
import numpy as np
from skimage.transform import resize

# CLASSES DEFINITION

class convnet(object):

    def __init__(self,
                 input_shape,
                 filters = (32,64),
                 kernel_size = (5,5),
                 pool_size = (2,2),
                 dense_units = 1024,
                 learning_rate=0.001
                 ):

        # Input Placeholder

        h, w, ch = input_shape
        input_layer = self.input = tf.placeholder(dtype=tf.float32, shape=[None, h, w, ch])

        ############################### debug mnist
        labels = self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 2]) # one more class for null example

        # Convolutions and poolings

        convs = []
        pools = []

        pool_layer = input_layer
        for i in range(len(filters)):

            conv_layer = tf.layers.conv2d(
                            inputs=pool_layer,
                            filters=filters[i],
                            kernel_size=[kernel_size[i], kernel_size[i]],
                            padding="same",
                            activation=tf.nn.relu)

            convs.append(conv_layer)

            pool_layer = tf.layers.max_pooling2d(
                            inputs=conv_layer,
                            pool_size=[pool_size[i], pool_size[i]],
                            strides=pool_size[i])

            pools.append(pool_layer)

        # Dense Layer

        last_h, last_w = h//np.prod(pool_size), w//np.prod(pool_size)
        last_pool_flat_size = last_h * last_w * filters[-1]
        last_pool_flat = self.last_pool_flat = tf.reshape(pools[-1], [-1, last_pool_flat_size])

        dense = self.dense = tf.layers.dense(
                            inputs=last_pool_flat,
                            units=dense_units,
                            activation=tf.sigmoid)

        # Logits

        logits = self.logits = tf.layers.dense(
                            inputs=dense,
                            units=2)

        # Prediction

        self.prediction = tf.argmax(self.logits, 1)

        # Calculate Loss

        loss = self.loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits_v2(
                                    labels=labels, logits=logits))

        # Configure the Training Operation

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(
                            loss=loss,
                            global_step=tf.train.get_global_step())

        # Add evaluation metrics

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

class unet(object):

    def __init__(self,
                 input_shape,
                 filters = (32,64),
                 kernel_size = (5,5),
                 pool_size = (2,2),
                 dense_units = 1024,
                 decode_kernel_size = 3,
                 learning_rate = 0.001
                 ):

        # Input Placeholder

        h, w, ch = input_shape
        input_layer = self.input = tf.placeholder(dtype=tf.float32, shape=[None, h, w, ch])
        labels = self.labels = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 1])

        # ENCODER

        convs = []
        pools = []

        pool_layer = input_layer
        for i in range(len(filters)):

            conv_layer = tf.layers.conv2d(
                            inputs=pool_layer,
                            filters=filters[i],
                            kernel_size=[kernel_size[i], kernel_size[i]],
                            padding="same",
                            activation=tf.nn.relu)

            convs.append(conv_layer)

            pool_layer = tf.layers.max_pooling2d(
                            inputs=conv_layer,
                            pool_size=[pool_size[i], pool_size[i]],
                            strides=pool_size[i])

            pools.append(pool_layer)

        encode = self.encode = pools[-1]

        # Dense Layer

        last_h, last_w = h//np.prod(pool_size), w//np.prod(pool_size)
        last_pool_flat_size = last_h * last_w * filters[-1]
        last_pool_flat = tf.reshape(pools[-1], [-1, last_pool_flat_size])

        dense = self.dense = tf.layers.dense(
                            inputs=last_pool_flat,
                            units=dense_units,
                            activation=tf.nn.relu)

        # DECODER

        deconvs = []

        deconv_layer = pools[-1]
        for i in range(len(filters)):

            deconv_layer = tf.layers.conv2d_transpose(
                            inputs=deconv_layer,
                            filters=filters[-i-1],
                            kernel_size=[kernel_size[-i-1], kernel_size[-i-1]],
                            padding='same',
                            strides=pool_size[-i-1],
                            activation=tf.sigmoid)

            deconv_layer = deconv_layer + convs[-i-1] # hourglass bridge

            deconvs.append(deconv_layer)

        self.last_tensor = deconvs[-1]

        decode = tf.layers.conv2d(
                            inputs=self.last_tensor,
                            filters=1,
                            kernel_size=[decode_kernel_size, decode_kernel_size],
                            padding="same",
                            use_bias=False)

        self.decode = tf.sigmoid(decode)

        # Calculate Loss
        #todo: verificare
        loss = self.loss = tf.losses.sigmoid_cross_entropy(
                            labels,
                            decode )

        # Configure the Training Operation

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(
                            loss=loss,
                            global_step=tf.train.get_global_step())

        #############################################
        # This operation are added to better pretrain M

        decode_flat = tf.reshape(decode, [-1, h*w])

        classification_hidden_layer = tf.layers.dense( decode_flat,
                                                       units=300,
                                                       activation=tf.nn.relu,
                                                       use_bias=True)
        classification_output = tf.layers.dense( classification_hidden_layer,
                                                 units=1,
                                                 activation=tf.sigmoid,
                                                 use_bias=False)

        input_class = self.input_class = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        loss_pretrain = loss + tf.losses.mean_squared_error(input_class, classification_output)

        train_op_pretrain = optimizer.minimize(
                            loss=loss_pretrain,
                            global_step=tf.train.get_global_step())

        self.input_class = input_class
        self.loss_pretrain = loss_pretrain
        self.train_op_pretrain = train_op_pretrain

        #############################################


class CityMap(object):

    def __init__(self,
                 environment_shape,
                 scales,
                 patch_size,
                 action_grid_size, # a 2by2 grid
                 number_of_zeros=6,
                 number_of_other_digits=50):

        self.environment_shape = environment_shape

        # this is a list of the size of the patches at different scales
        #
        # for example:
        #
        # scales[0] = 2**3 * 28
        # scales[1] = 2**2 * 28
        # scales[2] = 2**1 * 28
        # scales[3] = 2**0 * 28
        #
        self.scales = scales

        # zoom out, no zoom, zoom in
        self.action_grid_size = action_grid_size

        self.patch_size = patch_size

        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.number_of_zeros = number_of_zeros
        self.number_of_other_digits = number_of_other_digits

    def generate_environment(self):

        IMG = np.zeros(self.environment_shape)
        LABELS = np.zeros(self.environment_shape)

        number_of_zeros = self.number_of_zeros
        number_of_other_digits = self.number_of_other_digits

        # create an empty square in a random position

        empty_patch_size = self.environment_shape[0] // 3
        zeros_margin_size = self.environment_shape[0] // 6

        rnd_X, rnd_Y = (
            np.random.randint(0, self.environment_shape[0] - empty_patch_size),
            np.random.randint(0, self.environment_shape[1] - empty_patch_size))

        LABELS[rnd_X:rnd_X + empty_patch_size, rnd_Y:rnd_Y + empty_patch_size] = -1

        for i in range(number_of_zeros):

            # Pick a digit
            done = False
            while not done:
                digit, labels = self.mnist.train.next_batch(1)
                if labels[0, 0] == 1: done = True

            digit = np.reshape(digit, newshape=(28, 28, 1))

            label_patch = np.zeros(np.shape(digit))
            label_patch[digit > 0] = 1.

            done = False
            while not done:

                rnd_x = np.random.randint(0, self.environment_shape[0] - 28)
                rnd_y = np.random.randint(0, self.environment_shape[1] - 28)

                if ((rnd_X - zeros_margin_size < rnd_x < rnd_X + empty_patch_size + zeros_margin_size - 28) and
                        (rnd_Y - zeros_margin_size < rnd_y < rnd_Y + empty_patch_size + zeros_margin_size - 28)):

                    if (np.abs(LABELS[rnd_x:rnd_x + 28,
                               rnd_y:rnd_y + 28]) * label_patch).sum() == 0:  # if labels do not superimpose
                        new_patch = np.concatenate([IMG[rnd_x:rnd_x + 28, rnd_y:rnd_y + 28], digit], axis=2)
                        IMG[rnd_x:rnd_x + 28, rnd_y:rnd_y + 28] = np.reshape(np.amax(new_patch, axis=2), [28, 28, 1])
                        LABELS[rnd_x:rnd_x + 28, rnd_y:rnd_y + 28][label_patch > 0] = 1
                        done = True

        for i in range(number_of_other_digits):

            # Pick a digit
            done = False
            while not done:
                digit, labels = self.mnist.train.next_batch(1)
                if not labels[0, 0] == 1: done = True

            digit = np.reshape(digit, newshape=(28, 28, 1))

            label_patch = np.zeros(np.shape(digit))
            label_patch[digit > 0] = 1.

            done = False
            while not done:

                rnd_x = np.random.randint(0, self.environment_shape[0] - 28)
                rnd_y = np.random.randint(0, self.environment_shape[1] - 28)

                if (np.abs(LABELS[rnd_x:rnd_x + 28,
                           rnd_y:rnd_y + 28]) * label_patch).sum() == 0:  # if labels do not superimpose
                    new_patch = np.concatenate([IMG[rnd_x:rnd_x + 28, rnd_y:rnd_y + 28], digit], axis=2)
                    IMG[rnd_x:rnd_x + 28, rnd_y:rnd_y + 28] = np.reshape(np.amax(new_patch, axis=2), [28, 28, 1])
                    LABELS[rnd_x:rnd_x + 28, rnd_y:rnd_y + 28][label_patch > 0] = -1
                    done = True

        LABELS[LABELS < 0] = 0

        return IMG, LABELS

    def new_observation(self, img, new_Position, size, markVisited=False):
        """

        :param img:
        :param new_Position:
        :param size:
        :param scale: than the patch is 2**(k-scale)
        :param markVisited:
        :return:
        """

        x_start = new_Position[0]
        x_stop = x_start + size
        y_start = new_Position[1]
        y_stop = y_start + size

        crop = img[x_start:x_stop, y_start:y_stop]

        if markVisited:
            self.VisitedLocations[x_start:x_stop, y_start:y_stop] = 1

        return resize(crop, output_shape=(self.patch_size, self.patch_size))

    def reset(self):

        # Initial position in the middle
        self.Position = [0, 0]

        self.Scale = 0

        # Generate new environment
        self.img, self.labels = self.generate_environment()

        # MEMORY save information about last prediction
        # and, in the second channel, the Scale at which
        # that prediction has been taken.
        self.MEMORY = np.zeros((self.environment_shape[0], self.environment_shape[1], 2))
        self.MEMORY[:,:,0] += .01

        # Visited locations
        self.VisitedLocations = np.zeros((self.environment_shape[0],
                                          self.environment_shape[1],
                                          1))

        # State

        self.PER = self.new_observation(
            img=np.concatenate([self.img, self.VisitedLocations],
                               axis=2),
            new_Position=self.Position,
            size=self.scales[self.Scale])

        return self.PER

    def zoom_out(self):

        # Initial position in the middle
        self.Position = [0, 0]

        self.Scale = 0

        # State

        self.PER = self.new_observation(
            img=np.concatenate([self.img, self.VisitedLocations],
                               axis=2),
            new_Position=self.Position,
            size=self.scales[self.Scale])

        return self.PER

    def step(self, a):

        # compute actual deltas

        delta_x = (a // self.action_grid_size) * (self.scales[self.Scale] // self.action_grid_size)
        delta_y = (a % self.action_grid_size) * (self.scales[self.Scale] // self.action_grid_size)

        # compute potential new position and scale

        self.Position[0] += delta_x
        self.Position[1] += delta_y

        # update position and scale, if it does not go out of the ranges

        self.Scale += 1

        # State

        if self.Scale == len(self.scales) - 1:
            markVisited = True
        else:
            markVisited = False

        self.PER = self.new_observation(
            img=np.concatenate([self.img, self.VisitedLocations], axis=2),  # TODO: verificare
            new_Position=self.Position,
            size=self.scales[self.Scale],
            markVisited=markVisited)

        self.PER_labels = self.new_observation(
            img=self.labels,
            new_Position=self.Position,
            size=self.scales[self.Scale])

        return self.PER, self.PER_labels

    def update_MEMORY(self,
                    last_prediction):

        new_MEMORY_patch = np.ones(
                (self.scales[self.Scale],
                 self.scales[self.Scale],
                 2))

        new_MEMORY_patch[:,:,0] = resize(
                 np.reshape(last_prediction, newshape=(self.patch_size, self.patch_size)),
                 output_shape=np.shape(new_MEMORY_patch[:,:,0]))

        new_MEMORY_patch[:,:,1] *= self.Scale

        # # start reward computation
        # REWARD = np.square(self.labels - self.MEMORY[:,:,0]).mean()

        # now change memory

        size = self.scales[self.Scale]

        x_start = self.Position[0]
        x_stop = x_start + size
        y_start = self.Position[1]
        y_stop = y_start + size

        # TODO: optimize this operation

        for i_crop, i in enumerate(range(x_start, x_stop)):
            for j_crop, j in enumerate(range(y_start, y_stop)):

                if ((0<=i<self.environment_shape[0]) and (0<=j<self.environment_shape[1])):

                    if self.MEMORY[i, j, 1] <= new_MEMORY_patch[i_crop, j_crop, 1]:

                        self.MEMORY[i, j] = new_MEMORY_patch[i_crop, j_crop]

        # finalizing REWARD computation
        # REWARD -= np.square(self.labels - self.MEMORY[:,:,0]).mean()

        return True

class MyModel(object):

    def __init__(self,
                 patch_size,
                 environment_shape,
                 action_grid_size,
                 number_of_scales,
                 decode_kernel_size = 3,
                 learning_rate = 0.001):

        """

        :param PER_size:
        :param environment_shape:
        :param k: number of possible scales.

        """
        #########################################
        # model M for inference the belief

        M = []

        for s in range(number_of_scales):
            M.append(unet(
                input_shape=[patch_size, patch_size, environment_shape[2]],
                filters=(32, 64),
                kernel_size=(5, 5),
                pool_size=(2, 2),
                dense_units=1024,
                decode_kernel_size=decode_kernel_size,
                learning_rate=0.001))

            # notice: loss, placeholders, ecc to train M's parameters
            #         are already defined in unet() class

        self.M = M

        #########################################
        # model Q for actions value learning

        number_of_actions = action_grid_size**2


        targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        action_onehot = tf.one_hot(action_holder, number_of_actions, dtype=tf.float32)

        Q_ = []
        Qout = []
        predict = []
        Q = []
        loss = []
        train_op = []

        for s in range(number_of_scales-1):

            # todo: make this decision more accurate (more parameters?)

            Q_.append( convnet(input_shape=[patch_size, patch_size, environment_shape[2] + 1],
                         filters=(32, 64),
                         kernel_size=(5, 5),
                         pool_size=(2, 2),
                         dense_units=1024,
                         learning_rate=0.001) )

            Qout.append( tf.layers.dense(Q_[-1].dense,
                               units=number_of_actions,
                               activation=tf.sigmoid) # important
                         )

            # best action
            predict.append( tf.argmax(Qout[-1],-1) )

            Q.append( tf.reduce_sum(tf.multiply(Qout[-1], action_onehot), axis=1) )

            # loss and optimization
            loss.append( tf.reduce_mean(tf.square(targetQ  - Q[-1])) )
            train_op.append( tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss[-1]) )

        self.predict = predict
        self.Q_ = Q_
        self.train_op = train_op
        self.targetQ = targetQ
        self.action_holder = action_holder
        self.Qout = Qout




