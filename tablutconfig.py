class TablutConfig:

    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        # Game
        # Dimensions of the game observation
        self.observation_shape = (1, 9, 9, 4)
        #(nb_channels, nb_rows, nb_cols)
        self.network_input_shape = (4, 9, 9)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(6561))
        # List of players. You should only edit the length
        self.players = list(range(2))
        self.moves_for_draw = 10

        # Network
        self.num_filters = 32

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 4
        self.threads_per_worker = 2
        self.max_moves = 60  # Maximum number of moves if game is not finished before
        self.max_time = 10

        # Exploration noise
        self.enable_noise_on_training = True
        self.noise_mean = 0.0
        self.noise_deviation = 0.1

        # Training
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 1000
        # Number of parts of games to train on at each training step
        self.batch_size = 4096  
        self.min_batch_size = 2048
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 100
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        # checkpoint_interval % epochs == 0!
        self.epochs = 10
        self.new_games_per_epoch = 100

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate

        # Save
        self.folder = "checkpoint"
        self.checkpoint_name = "tablut.ckpt"
        self.checkpoint_metadata = "tablut.metadata"
        self.action_buffer_name = "actionbuffer.bin"
        self.tflite_model = "tablut.tflite"