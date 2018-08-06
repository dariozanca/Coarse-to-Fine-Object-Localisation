# IMPORT EXTERNAL LIBRARIES
import random
import os
import matplotlib.pyplot as plt

# IMPORT OWN LIBRARIES
from blocks import *
from utils import *

# Global variables definition

class ExperimentConfiguration(object):

    def __init__(self):

        # Images shapes
        self.k = 4
        self.patch_size = 28

        self.action_grid_size = 2

        self.environment_shape = ((self.action_grid_size**self.k)*28,
                                  (self.action_grid_size**self.k)*28,
                                  1)

        self.number_of_zeros = 10
        self.number_of_other_digits = 150


        self.scales = []
        n = self.environment_shape[0]
        while n > self.patch_size:
            self.scales.append(n)
            n = n // self.action_grid_size

        # Both training specifications
        self.batch_size = 128 # How many experiences to use for each training step.
        self.num_episodes = 5001  # How many episodes_1 of game environment to train network with.
        self.max_epLength = 10 # number of action taken

        self.buffer_size = 20 * 10**3
        # RL training
        self.update_freq = 1 # How often to perform a RL training step.
        self.startE = 1 # Starting chance of random action
        self.endE = 0.1 # Final chance of random action
        self.annealing_steps = 2000 # How many steps of training to reduce startE to endE.
        self.pre_RL_train_episodes = 500 # How many steps of random episodes_1 before training begins.
        self.last_episodes_without_randomness = 500 #
        self.y = .0 # Discount factor on the target Q-values

        # Savings
        self.load_pretrained_M = True
        self.pretrained_M_path = "./pretrained_M_k" + str(self.k)

        self.LOG_filename = "LOG.txt"

        self.save_plots = True
        self.save_plots_each = 500
        self.save_plots_folder = "./episodes_savings"

        self.last_n_print = 10

        self.save_model = True
        self.save_model_folder = "./model_savings"


# Training definition
def CityMapTrain(config):

    # Create the environment
    env = CityMap(
        environment_shape=config.environment_shape,
        scales=config.scales,
        patch_size=config.patch_size,
        action_grid_size=config.action_grid_size,
        number_of_zeros=config.number_of_zeros,
        number_of_other_digits=config.number_of_other_digits)

    if config.save_plots:
        PLOT_PER = env.reset()
        PLOT_IMG = env.img
        PLOT_LABELS = env.labels

    # Reset the graph
    tf.reset_default_graph()

    # Create the model
    m = MyModel(
        patch_size=config.patch_size,
        environment_shape=config.environment_shape,
        action_grid_size=config.action_grid_size,
        number_of_scales=len(config.scales))



    # TF
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    ######################################################

    # PRE-TRAINED MODEL M


    with tf.Session() as sess:

        if config.load_pretrained_M == True:
            print('Loading Pretrained M...')
            ckpt = tf.train.get_checkpoint_state(config.pretrained_M_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        else:
            sess.run(init)
            from pretrain_M import pretrain_M
            pretrain_M(M=m.M, env=env, sess=sess)
            print "Saving the model..."
            saver.save(sess, config.pretrained_M_path + '/model_with_pretrained_M.ckpt')


        ## Try M

        trymodelFlag = False


        if trymodelFlag:
            print "\n\nTrying model M"
            from pretrain_M import create_dataset

            dataset_size = 5

            for s in range(len(config.scales)):

                # create dataset for that scale
                INPUT, OUTPUT = create_dataset(env, s, dataset_size=dataset_size)

                # train M[s] until convergence
                print "\nTrying model M[", s, "]...\n"

                decodes = sess.run((m.M[s].decode,),
                                             feed_dict={
                                                 m.M[s].input: INPUT,
                                                 m.M[s].labels: OUTPUT
                                             })[0]
                lista = []
                for i in range(dataset_size):
                    lista.append(INPUT[i])
                    lista.append(OUTPUT[i])
                    lista.append(decodes[i])

                stampa_try(lista, row=dataset_size)


        ######################################################

        # create the experience buffer
        myBuffer = []
        for _ in range(len(config.scales)-1):
            myBuffer.append(experience_buffer(
                    buffer_columns=4,
                    buffer_size=config.buffer_size))

        # Set the rate of random action decrease.
        e = config.startE
        stepDrop = (config.startE - config.endE) / config.annealing_steps



        # create lists to contain total rewards per episode
        r_List = []

        for episode in range(config.num_episodes):

            if episode > config.num_episodes - config.last_episodes_without_randomness:
                e = 0

            # Reset environment and get first new observation
            PER_list = []
            PER_list.append(env.reset())

            belief = sess.run(m.M[env.Scale].decode,
                              feed_dict={m.M[env.Scale].input: [np.reshape(PER_list[-1][:,:,0],
                                                                           newshape=(
                                                                               config.patch_size, config.patch_size,
                                                                               config.environment_shape[2]))]})

            # compute REWARDS
            # todo: use this for the reward, if it is better
            _ = env.update_MEMORY(
                last_prediction=belief)

            rAll = 0 # total reward for the episode

            j = 1
            while (j <= config.max_epLength):

                j = j+1

                # reset scale position without changing environment
                PER_list = []
                a_list = []
                PER_list.append( env.zoom_out())

                # r_list = []

                # do an action for each level
                # except for the last one where no action is needed
                while (env.Scale < len(config.scales)-1):

                    # exploration of exploitation
                    if (np.random.rand(1) < e or episode < config.pre_RL_train_episodes):
                        a = np.random.randint(0, config.action_grid_size**2)
                    else:
                        a = sess.run(m.predict[env.Scale],
                                        feed_dict={m.Q_[env.Scale].input: [PER_list[-1]]})[0]

                    # do an action and get new env configuration
                    newPER, newPER_labels = env.step(a=a)

                    belief = sess.run(m.M[env.Scale].decode,
                                      feed_dict={m.M[env.Scale].input: [np.reshape(newPER[:, :, 0],
                                                                                   newshape=(
                                                                                   config.patch_size, config.patch_size,
                                                                                   config.environment_shape[2]))],
                                                 m.M[env.Scale].labels: [newPER_labels]})

                    # compute REWARDS
                    # todo: use this for the reward, if it is better
                    _ = env.update_MEMORY(
                        last_prediction=belief)

                    PER_list.append(newPER)
                    a_list.append(a)

                # # compute REWARDS
                # r_arr = np.array(r_list)
                # r_arr = discount_rewards(r_arr)
                r = int(newPER_labels.sum() > 0) * int(newPER[:,:,-1].sum() == 0)

                # add to episode buffer
                for s in range(len(config.scales)-1):
                    myBuffer[s].add(
                            np.reshape(np.array([PER_list[s], a_list[s], r, PER_list[s+1]]), [1, 4]))  # Save the experience to our episode buffer.

                # del r_list[:]
                rAll += r

            # Save total reward for the episode
            r_List.append(rAll)

            # RL TRAINING

            if episode > config.pre_RL_train_episodes:

                if episode % (config.update_freq) == 0 and config.batch_size <= len(myBuffer[0].buffer):

                    if e > config.endE:
                        e -= stepDrop

                    for s in range(len(config.scales)-1):

                        # Get a random batch of experiences.
                        PER_batch, actions_batch, r_batch, _ = get_new_batch(
                                    buffer=myBuffer[s],
                                    config=config)

                        targetQ = r_batch # + y * Q

                        # Update the Next network.
                        _ = sess.run(m.train_op[s],
                                     feed_dict={m.Q_[s].input: PER_batch,
                                                m.targetQ: targetQ,
                                                m.action_holder: actions_batch})


            # Periodically save the model and plot some stats
            last_n = config.last_n_print
            if episode % last_n == 0 and len(r_List) > last_n:

                stamp = "Episode = "+str(episode)\
                        +" \t| Mean Reward = "+str(np.mean(r_List[-last_n:]))\
                        +" \t| Epsilon = "+str(e)\
                        +" \t| Buffer size = "+str(len(myBuffer[0].buffer))
                print(stamp)

                f = open(config.LOG_filename, "a")
                f.write(stamp + "\n")
                f.close()

            if config.save_plots and (episode % config.save_plots_each == 0):

                # Make a path for our model to be saved in.
                if not os.path.exists(config.save_plots_folder):
                    os.makedirs(config.save_plots_folder)

                _ = env.reset()
                env.img = PLOT_IMG
                env.labels = PLOT_LABELS
                PER_list = []
                PER_list.append(PLOT_PER)

                plt.imshow(PER_list[-1][:, :, 0])
                plt.show()
                belief = sess.run(m.M[env.Scale].decode,
                                  feed_dict={m.M[env.Scale].input: [np.reshape(PER_list[-1][:, :, 0],
                                                                               newshape=(
                                                                                   config.patch_size, config.patch_size,
                                                                                   config.environment_shape[2]))]})


                plt.imshow(np.squeeze(belief))
                plt.show()
                # compute REWARDS
                # todo: use this for the reward, if it is better
                _ = env.update_MEMORY(
                    last_prediction=belief)

                lista = []
                lista_belief = []

                lista_belief.append(np.reshape(np.copy(env.MEMORY[:, :, 0]),
                                            newshape=(config.environment_shape[0], config.environment_shape[1])))

                # compute optimal behavior

                j = 1
                while (j <= config.max_epLength):

                    j = j+1

                    # reset scale position without changing environment
                    PER_list = []
                    PER_list.append( env.zoom_out())

                    while (env.Scale < len(config.scales)-1):

                        Qout, a = sess.run((m.Qout[env.Scale], m.predict[env.Scale]),
                                            feed_dict={m.Q_[env.Scale].input: [PER_list[-1]]})

                        a = a[0]

                        newPER, newPER_labels = env.step(a=a)

                        belief = sess.run((m.M[env.Scale].decode, ),
                                          feed_dict={m.M[env.Scale].input: [np.reshape(newPER[:,:,:-1],
                                                                                       newshape=(
                                                                                           config.patch_size,
                                                                                           config.patch_size,
                                                                                           1))]})[0]


                        _ = env.update_MEMORY(
                            last_prediction=belief)

                        lista_belief.append(np.reshape(np.copy(env.MEMORY[:, :, 0]),
                                                       newshape=(
                                                       config.environment_shape[0], config.environment_shape[1])))

                        PER_list.append(newPER)

                    lista.append(np.reshape(np.copy(env.VisitedLocations),
                                            newshape=(config.environment_shape[0], config.environment_shape[1])))

                    lista.append(np.reshape(np.copy(env.MEMORY[:, :, 0]),
                                            newshape=(config.environment_shape[0], config.environment_shape[1])))

                salva_patches(PLOT_IMG, lista, filename=config.save_plots_folder +'/'+ str(episode) + "_steps")
                salva_belief(lista_belief, filename=config.save_plots_folder +'/'+ str(episode) + "_belief")

                # compute belief maps (for each scale, except for the last)

                lista = []

                lista.append(np.reshape(np.copy(env.img[:, :, 0]),
                                        newshape=(config.environment_shape[0], config.environment_shape[1])))

                lista.append(np.reshape(np.copy(env.labels[:, :, 0]),
                                        newshape=(config.environment_shape[0], config.environment_shape[1])))

                Q = sess.run(m.Qout[0],
                             feed_dict={m.Q_[0].input: [PLOT_PER]})

                bmap = belief_map(Q, config)
                bmap += .1 * resize(np.copy(env.img[:, :, 0]),
                           (np.shape(bmap)[0], np.shape(bmap)[1]))
                bmap /= bmap.max()
                lista.append(bmap)


                bmap = np.zeros((np.shape(bmap)[0]*config.action_grid_size, np.shape(bmap)[0]*config.action_grid_size))
                for ind_i in range(config.action_grid_size):
                    for ind_j in range(config.action_grid_size):

                        env.Scale = 0
                        env.Position = [0, 0]
                        newPER, _ = env.step(ind_i*config.action_grid_size + ind_j)

                        Q = sess.run(m.Qout[1],
                                     feed_dict={m.Q_[1].input: [newPER]})

                        submap = belief_map(Q, config)

                        bmap[ind_i*np.shape(submap)[0]:(ind_i+1)*np.shape(submap)[0],
                             ind_j*np.shape(submap)[0]:(ind_j+1)*np.shape(submap)[0] ] = submap


                bmap += .1 * resize(np.copy(env.img[:, :, 0]),
                           (np.shape(bmap)[0], np.shape(bmap)[1]))
                bmap /= bmap.max()
                lista.append(bmap)

                salva_belief_maps(lista, filename=config.save_plots_folder+'/'+str(episode)+"_belief_maps")

                if config.save_model and episode % 1000 == 0:
                    # Make a path for our model to be saved in.
                    if not os.path.exists(config.save_model_folder):
                        os.makedirs(config.save_model_folder)
                    print "Saving ckpt of the the model..."
                    if not os.path.exists(config.save_model_folder + '/episode' + str(episode) +'/'):
                        os.makedirs(config.save_model_folder + '/episode' + str(episode) +'/')
                    saver.save(sess, config.save_model_folder + '/episode' + str(episode) +'/model_version.ckpt')


###################################################################################################
''' Main '''

def main(_):

    config = ExperimentConfiguration()
    CityMapTrain(config)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

