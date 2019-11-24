import numpy as np
import tensorflow as tf

class Network():

    def initialize(self):

        pass

    def generator(self):
        """
        tf.data.Dataset generator

        """
        while True:

            pass

    def init_batch(self):

        pass

    def init_graph(self):

        pass

    def init_training(self):

        pass

    def init_saver(self, save_dir):

        pass

    def create_session(self, memory_fraction=0.85):

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = memory_fraction 
        self.sess = tf.Session(config=config)

    def train(self):

        pass

    def run_block(self):

        pass
