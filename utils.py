### IMPORT EXTERNAL LIBRARIES
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import select
import sys
import random
import matplotlib.patches as patches


def heardEnter():

    ''' Listen for the user pressing ENTER '''

    i,o,e = select.select([sys.stdin],[],[],0.0001)

    for s in i:

        if s == sys.stdin:
            input = sys.stdin.readline()
            return True

    return False

class experience_buffer():
    """ This buffer saves data for RL training """

    def __init__(self,
                buffer_columns,
                buffer_size=5000):

        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_columns = buffer_columns

    def add(self, experience):

        if len(self.buffer) + len(experience) >= self.buffer_size:
            del self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size]
        self.buffer.extend(experience)

    def sample(self, size):

        if size > len(self.buffer): size = len(self.buffer)
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, self.buffer_columns])

def discount_rewards(r, discount=.99):

    """ take 1D float array of rewards and compute discounted reward """

    discounted_r = np.zeros_like(r)
    running_add = 0

    for t in reversed(xrange(0, r.size)):
        running_add = running_add * discount + r[t]
        discounted_r[t] = running_add

    return discounted_r


def stampa(lista, title=""):

    fig = plt.figure()
    plt.title(title)

    for i in range(len(lista)):
        fig.add_subplot(1, len(lista), i+1)
        plt.imshow(np.squeeze(lista[i]))

    plt.show()
    plt.close()


def salva(lista, title="", filename="prova.png"):

    fig = plt.figure()
    plt.title(title)

    for i in range(len(lista)):
        fig.add_subplot(1, len(lista), i+1)
        plt.imshow(np.squeeze(lista[i]))

    plt.savefig(filename)
    plt.close()

def salva_patches(IMG, lista, title="", filename="prova.png"):

    fig = plt.figure()
    plt.title(title)

    plt.axis('off')

    for i in range(len(lista)):
        fig.add_subplot(5, 4, i+1)
        IMG = np.reshape(IMG, newshape=(np.shape(IMG)[0], np.shape(IMG)[1]))

        lista[i] = np.reshape(lista[i], newshape=(np.shape(IMG)[0], np.shape(IMG)[1]))

        if i%2==0:
            plt.imshow(IMG+lista[i])
        else:
            plt.imshow(lista[i])
        plt.title("step " + str(i + 1))

        plt.axis('off')

    plt.savefig(filename)
    plt.close()

def salva_belief(lista, title="", filename="prova.png"):

    fig = plt.figure()
    plt.title(title)

    plt.axis('off')

    for i in range(12):
        fig.add_subplot(4, 5, i+1)

        # lista[i] = np.reshape(lista[i], newshape=(np.shape(IMG)[0], np.shape(IMG)[1]))
        plt.imshow(lista[i])

        plt.axis('off')

    plt.savefig(filename)
    plt.close()

def get_new_batch(buffer, config):

    trainBatch = buffer.sample(config.batch_size)

    PER_batch = np.vstack(trainBatch[:, 0])
    PER_batch = np.reshape(PER_batch,
                           newshape=(-1,
                                     config.patch_size,
                                     config.patch_size,
                                     config.environment_shape[2] + 1))

    actions_batch = np.vstack(trainBatch[:, 1])
    actions_batch = np.reshape(actions_batch,
                               newshape=[-1])

    r_batch = np.vstack(trainBatch[:, 2])
    r_batch = np.reshape(r_batch,
                         newshape=[-1])

    # don't need successive peripheral observation for now
    # because we're training greedy, without reward

    return (PER_batch, actions_batch, r_batch, False)

def belief_map(Q, config):

    Q = np.reshape(Q, newshape=(config.action_grid_size,
                                config.action_grid_size))

    bmap = np.ones((2*3*4*5, 2*3*4*5))

    for i in range(config.action_grid_size):
        for j in range(config.action_grid_size):

            step = np.shape(bmap)[0] // config.action_grid_size
            bmap[i*step:(i+1)*step,
                 j*step:(j+1)*step] *= Q[i,j]

    return bmap

def salva_belief_maps(lista, title="", filename="prova.png"):

    fig = plt.figure()
    plt.title(title)

    plt.axis('off')

    for i in range(len(lista)):
        fig.add_subplot(2, len(lista)//2, i+1)

        plt.imshow(lista[i])
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("step " + str(i + 1))

        plt.axis('off')

    plt.savefig(filename)
    plt.close()


def stampa_try(lista, row = 1, title=""):

    fig = plt.figure()
    plt.title(title)
    plt.axis('off')


    for i in range(len(lista)):
        fig.add_subplot(row, len(lista) // row, i+1)
        plt.imshow(np.squeeze(lista[i]))
        plt.axis('off')

    plt.show()
