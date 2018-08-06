import numpy as np
import tensorflow as tf
from blocks import *
from utils import *
from skimage.transform import resize


def create_dataset(
        env,
        scale,
        dataset_size = 10):

    """

    :param env: pre-generated environment
    :param scale: scale at which we want to take examples
    :param dataset_size: number of examples to produce
    :return:
    """

    number_of_actions = env.action_grid_size ** 2

    INPUT = []
    OUTPUT = []

    for i in range(dataset_size):

        PER = env.reset()
        PER_labels = resize(env.labels, (env.patch_size, env.patch_size))

        for _ in range(scale):

            PER, PER_labels = env.step(np.random.randint(0, number_of_actions))

        INPUT.append( PER[:,:,:-1] )
        OUTPUT.append( PER_labels )

    return np.array(INPUT), np.array(OUTPUT)



def pretrain_M(
        M,
        env,
        sess,
        max_epochs = 200,
        stamp_each = 10,
        dataset_size= 10*128,
        batch_size = 1*128):

    number_of_scales = len(env.scales)


    for s in range(number_of_scales):

        # create dataset for that scale
        print "\nGenerating dataset for scale number ", s, "..."
        INPUT, OUTPUT = create_dataset(env, s, dataset_size=dataset_size)

        # create validation set for that scale
        VAL_INPUT, VAL_OUTPUT = create_dataset(env, s, dataset_size=dataset_size//10)

        means = np.mean(OUTPUT, axis=(1, 2))
        mean = np.mean(means[means > 0])

        # train M[s] until convergence
        print "\nTraining model M[", s, "]...\n"

        for epoch in range(max_epochs):

            loss_epoch = 0

            for minibatch in range(dataset_size//batch_size):

                batch_x = INPUT[minibatch*batch_size:(minibatch+1)*batch_size]
                batch_y = OUTPUT[minibatch*batch_size:(minibatch+1)*batch_size]

                # TODO: verify! I ask to predict 1 if the number of pixels belonging to a zero-digit are more than the average
                batch_class = (np.mean(batch_y, axis=(1,2)) > mean).astype(int)

                loss_minibatch, loss_pre, _ = sess.run( (M[s].loss, M[s].loss_pretrain, M[s].train_op_pretrain),
                                              feed_dict={
                                                  M[s].input: batch_x,
                                                  M[s].labels: batch_y,
                                                  M[s].input_class: batch_class
                                              })

                loss_epoch += loss_minibatch

            # validation

            batch_x = VAL_INPUT
            batch_y = VAL_OUTPUT

            # TODO: verify! I ask to predict 1 if the number of pixels belonging to a zero-digit are more than the average
            batch_class = (np.mean(batch_y, axis=(1, 2)) > mean).astype(int)

            val_loss, val_loss_pre = sess.run((M[s].loss, M[s].loss_pretrain),
                                                   feed_dict={
                                                       M[s].input: batch_x,
                                                       M[s].labels: batch_y,
                                                       M[s].input_class: batch_class
                                                   })

            # print stats
            if epoch % stamp_each == stamp_each-1 or epoch == 0:
                print "Epoch = ", epoch+1, "\t| Loss = ", loss_epoch / (dataset_size // batch_size ), "\t| Loss (pre) = ", loss_pre, "\t| Validation Loss = ", val_loss, "\t| Validation Loss (pre) = ", val_loss_pre

            if heardEnter(): break





