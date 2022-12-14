class BaseConfig(object):
    def __init__(self):
        self.n_states = 12 * 9
        self.use_images = False
        self.n_actions = 7
        self.hidden_dim = 256
        self.game_time = 15.
        self.max_iter = 150
        self.lr = 0.00001
        self.max_frames = 5000#100#10000
        self.num_steps = 10
        self.fps = 10

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)