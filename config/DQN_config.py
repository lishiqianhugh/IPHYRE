
class BaseConfig(object):
    def __init__(self):
        self.n_states = 12 * 9
        self.use_images = False
        self.n_actions = 7
        self.hidden_dim = 256
        self.game_time = 15.
        self.max_iter = 150
        self.memory_capacity = 3000
        self.epsilon = 0.9
        self.epsilon_increment = 0
        self.gamma = 0.9
        self.target_replace_iter = 300
        self.learn_freq = 5
        self.batch_size = 32
        self.epoch = 1000
        self.lr = 0.01

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)