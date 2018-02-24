from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    def __init__(self, state, action):
        self.state_size = state
        #lr
        self.lr = 0.0001
        #action_data: size, low, and high
        self.action_size = action[0]
        self.action_low = action[1]
        self.action_hi = action[2]
        self.model = None
        self.create_network()

    def create_network(self):
        """Build the NN """
        states = layers.Input(shape=(self.state_size,), name='states')
        layer = layers.Dense(units=16, activation='relu')(states)
        layer = layers.Dense(units=32, activation='relu')(layer)
        layer = layers.Dense(units=16, activation='relu')(layer)

        layer_out = layers.Dense(units=self.action_size, activation='sigmoid',
            name='layers_out')(layer)

        range = self.action_hi - self.action_low
        actions = layers.Lambda(lambda x: (x * range) + self.action_low,
                               name='actions')(layer_out)

        self.model = models.Model(inputs=states, outputs=actions)
        action_grad = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_grad * actions)

	#optimizer
        optimizer = optimizers.Adam(lr=self.lr)
        updates = optimizer.get_updates(params=self.model.trainable_weights,loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_grad, K.learning_phase()],
            outputs=[],
            updates=updates)

    def set_weights(self, src):
        self.model.set_weights(src.get_weights())

    def get_weights(self):
        return self.model.get_weights()

    def predict(self, states):
        return self.model.predict(states)

    def update_weights(self, weights):
        self.model.set_weights(weights)
