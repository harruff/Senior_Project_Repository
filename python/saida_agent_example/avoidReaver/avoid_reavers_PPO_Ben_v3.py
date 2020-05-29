# Updated work from:
# Ben Harruff
# harrbs02@pfw.edu
# 03/23/2020
#
# Original work from:
# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
#
# This code is distribued under the terms and conditions from the MIT 
#   License (MIT).
#
# Authors : Uk Jo, Iljoo Yoon, Hyunjae Lee, Daehun Jun
# 
# Initial framework taken from 
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np
import os
from datetime import datetime
import math
import argparse

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

from core.algorithm.PPO import PPOAgent
from core.common.processor import Processor
from saida_gym.starcraft.avoidReavers import AvoidReavers
from core.callbacks import DrawTrainMovingAvgPlotCallback
import saida_gym.envs.conn.connection_env as Config
from core.common.util import OPS

# Argument Parser
# Hyper Parameter definitions
parser = argparse.ArgumentParser(description='PPO Configuration for Avoid_Reaver')
parser.add_argument(OPS.NO_GUI.value,       help='gui',         
    default=False,  type=bool   )
parser.add_argument(OPS.MOVE_ANG.value,     help='move angle',  
    default=18,     type=int    )
parser.add_argument(OPS.MOVE_DIST.value,    help='move dist',   
    default=2,      type=int    )
parser.add_argument(OPS.GAMMA.value,        help='gamma',       
    default=0.999,  type=float  )
parser.add_argument(OPS.EPOCHS.value,       help='Epochs',      
    default=3,     type=int    )

args = parser.parse_args()

# Populate post_fix with hyper-parameter definitions for file names when saving models and graphs.
# Used to diffentiate results from one another based on the parameters used.
dict_args = vars(args)
post_fix = ''
for k in dict_args.keys():
    if k == OPS.NO_GUI():
        continue
    post_fix += '_' + k + '_' + str(dict_args[k])

# Hyper parameter definitions
# Determine whether to render the environment or not. 
# True results in messier but quicker training
NO_GUI          = dict_args[OPS.NO_GUI()]

# Horizon range
NB_STEPS        = 100000 

# Size of state for network
STATE_SIZE      = 8 + 3 * 8

# Only implemented clipping for the surrogate loss
# Paper said it was best
LOSS_CLIPPING   = 0.2    

# Learning Rate (3e-3 to 5e-6)
LR              = 1e-3   

# Exploration noise 
NOISE           = 0.1     

# Discount Factor (Usually .9900, range 0.8000 - 0.9997)
GAMMA           = dict_args[OPS.GAMMA()] 

# Entropy Coefficient 
# Used to determine how quickly to converge on an optimal
# action given a situation. The smaller the coefficient, the 
# slower the rate of randomness reduction for the action 
# decision-making process
# Typical range is (0 to 1e-2)
ENTROPY_LOSS    = 1e-2   

# How many steps to perform and collect data from before 
# updating the model. Should be a multiple of batch size.
# Larger values lead to slower but more stable updates.
# Typical range is 2,048 to 409,600 for discrete
#BUFFER_SIZE     = 256
BUFFER_SIZE     = 5120

# Number of steps before next iteration of gradient descent
# Should always be multiple of buffer size.
# Typical range is 32 to 512 for discrete
#BATCH_SIZE      = 64     
BATCH_SIZE      = 512

# Size of hidden layers in the neural network.
# More complex problems demand larger hidden layer sizes
# Typical range is 32 for a straightforward problem and
# as high as 512 for very complex problems.
#HIDDEN_SIZE     = 80  
HIDDEN_SIZE     = 128

# Number of hidden layers present after observation input
# More layers are necessary for more complex problems
# Typical range is 1 to 3
NUM_LAYERS      = 3       

#
EPOCHS          = dict_args[OPS.EPOCHS()]    

def scale_velocity(v):
    return v


def scale_angle(angle):
    return (angle - math.pi) / math.pi

# Scale position 
def scale_pos(pos):
    return pos / 16


def scale_pos2(pos):
    return pos / 8


def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


# Reshape the reward in a way you want
def reward_reshape(reward):
    """ Reshape the reward
        Starcraft Env returns the reward according to following conditions.
        1. Invalid action : -0.1
        2. get hit : -1
        3. goal : 1
        4. farther : 0
        5. closer : 2

    # Argument
        reward (float): The observed reward after executing the action

    # Returns
        reshaped reward
        1. Invalid action : -1
        2. get hit : -5
        3. goal : 5
        4. farther : -0.1
        5. closer : 0.1
    """

    if math.fabs(reward + 0.1) < 0.01:
        reward = -1
    elif reward == -1:
        reward = -5
    elif reward == 1:
        reward = 5
    elif reward == 0:
        reward = -0.1
    elif reward == 2:
        reward = 0.1

    return reward

# Define's the agent's Processor
class ReaverProcessor(Processor):
    def __init__(self):
        self.last_action = None
        self.success_cnt = 0
        self.cumulate_reward = 0

    def process_action(self, action):
        self.last_action = action
        return action

    def process_step(self, observation, reward, done, info):
        state_array = self.process_observation(observation)
        reward = reward_reshape(reward)
        self.cumulate_reward += reward

        if reward == 10:
            if self.cumulate_reward > 0:
                self.success_cnt += 1

            self.cumulate_reward = 0
            print("success_cnt = ", self.success_cnt)

        return state_array, reward, done, info

    def process_observation(self, observation, **kwargs):
        """ Pre-process observation

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            processed observation

        """
        if len(observation.my_unit) > 0:
            s = np.zeros(STATE_SIZE)
            me = observation.my_unit[0]
            # Observation for Dropship
            s[0] = scale_pos2(me.pos_x)             # X of coordinates
            s[1] = scale_pos2(me.pos_y)             # Y of coordinates
            s[2] = scale_pos2(me.pos_x - 320)       # relative X of coordinates from goal
            s[3] = scale_pos2(me.pos_y - 320)       # relative Y of coordinates from goal
            s[4] = scale_velocity(me.velocity_x)    # X of velocity
            s[5] = scale_velocity(me.velocity_y)    # Y of velocity
            s[6] = scale_angle(me.angle)            # Angle of head of Dropship
            s[7] = 1 if me.accelerating else 0      # True if Dropship is accelerating

            # Observation for Reavers
            for ind, ob in enumerate(observation.en_unit):
                s[ind * 8 + 8] = scale_pos2(ob.pos_x - me.pos_x)    # X of coordinates
                s[ind * 8 + 9] = scale_pos2(ob.pos_y - me.pos_y)    # Y of coordinates
                s[ind * 8 + 10] = scale_pos2(ob.pos_x - 320)        # X of relative coordinates from goal
                s[ind * 8 + 11] = scale_pos2(ob.pos_y - 320)        # Y of relative coordinates from goal
                s[ind * 8 + 12] = scale_velocity(ob.velocity_x)     # X of velocity
                s[ind * 8 + 13] = scale_velocity(ob.velocity_y)     # Y of velocity
                s[ind * 8 + 14] = scale_angle(ob.angle)             # Angle of head of Reavers
                s[ind * 8 + 15] = 1 if ob.accelerating else 0       # True if Reaver is accelerating

        return s


def build_actor(state_size, action_size, advantage, old_prediction):
    state_input = Input(shape=(state_size,))

    x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
    for _ in range(NUM_LAYERS - 1):
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)

    out_actions = Dense(action_size, activation='softmax', name='output')(x)

    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])

    return model


def build_actor_continuous(state_size, action_size, advantage, old_prediction):

    state_input = Input(shape=(state_size,))
    x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)

    for _ in range(NUM_LAYERS - 1):
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)

    out_actions = Dense(action_size, name='output', activation='tanh')(x)

    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])

    return model


def build_critic(state_size):

    state_input = Input(shape=(state_size,))
    x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
    for _ in range(NUM_LAYERS - 1):
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)

    out_value = Dense(1)(x)

    model = Model(inputs=[state_input], outputs=[out_value])

    return model


if __name__ == '__main__':
    training_mode = True
    load_model = False
    FILE_NAME = os.path.basename(__file__).split('.')[0] + "-" + datetime.now().strftime("%m%d%H%M%S")
    action_type = 0

    env = AvoidReavers(move_angle=dict_args[OPS.MOVE_ANG()], move_dist=dict_args[OPS.MOVE_DIST()], frames_per_step=12
                       , verbose=0, action_type=action_type, no_gui=NO_GUI)

    ACTION_SIZE = env.action_space.n

    continuous = False if action_type == 0 else True

    # Build models
    actor = None
    ADVANTAGE = Input(shape=(1,))
    OLD_PREDICTION = Input(shape=(ACTION_SIZE,))

    if continuous:
        actor = build_actor_continuous(STATE_SIZE, ACTION_SIZE, ADVANTAGE, OLD_PREDICTION)
    else:
        actor = build_actor(STATE_SIZE, ACTION_SIZE, ADVANTAGE, OLD_PREDICTION)

    critic = build_critic(STATE_SIZE)

    agent = PPOAgent(STATE_SIZE, ACTION_SIZE, continuous, actor, critic, GAMMA, LOSS_CLIPPING, EPOCHS, NOISE, ENTROPY_LOSS,
                     BUFFER_SIZE,BATCH_SIZE, processor=ReaverProcessor())

    agent.compile(optimizer=[Adam(lr=LR), Adam(lr=LR)], metrics=[ADVANTAGE, OLD_PREDICTION])

    if training_mode == False:
        agent.load_weights(os.path.realpath("C:/Senior_Project_Repository/python/saida_agent_example/avoidReaver/save_model/"))   
    
    cb_plot = DrawTrainMovingAvgPlotCallback(os.path.realpath("C:/Senior_Project_Repository/python/saida_agent_example/avoidReaver/save_graph/" + FILE_NAME + '_'+post_fix + '.png'), 50, 5, l_label=['episode_reward'])
    agent.run(env, NB_STEPS, train_mode=training_mode, verbose=1, callbacks=[cb_plot], action_repetition=1, nb_episodes=100000)

    if training_mode == True:
        agent.save_weights(os.path.realpath("C:/Senior_Project_Repository/python/saida_agent_example/avoidReaver/save_model/"),"avoid_Reavers_PPO"+post_fix)

    env.close()
