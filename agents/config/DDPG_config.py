class BaseConfig(object):
    def __init__(self):
        self.batch_size = 128
        self.memory_capacity = 3000
        self.use_images = False
        self.game_time = 15.
        self.n_states = 12 * 9
        self.n_actions = 7
        self.gamma = 0.99
        self.tau = 1e-2
        self.hidden_dim = 256
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.episode = 10000#10000
        self.max_iter = 150
        self.game_time = 15.

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)