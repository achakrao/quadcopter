from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    def __init__(self, state, action):
        self.state_size = state
        self.action_size = action

        self.lr = 0.001
        self.model = None
        self.Q_values = None
        self.create_network()

    def create_network(self):
        '''Create the NN for critic'''

        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        state_layer = layers.Dense(units=8, activation='relu')(states)
        state_layer = layers.Dense(units=16, activation='relu')(state_layer)

        action_layer = layers.Dense(units=8, activation='relu')(actions)
        action_layer = layers.Dense(units=16, activation='relu')(action_layer)

        layer = layers.Add()([state_layer, action_layer])
        layer = layers.Activation('relu')(layer)

        self.Q_values = layers.Dense(units=1, name='q_values')(layer)

        self.model = models.Model(inputs=[states, actions], outputs=self.Q_values)

        #optimizer
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        #action gradients dQ_values/dactions
        action_grads = K.gradients(self.Q_values, actions)

        self.get_action_gradients = K.function(
            inputs = [*self.model.input, K.learning_phase()],
            outputs = action_grads)

    def set_weights(self, src):
        self.model.set_weights(src.get_weights())

    def get_weights(self):
        return self.model.get_weights()

    def update_weights(self, weights):
        self.model.set_weights(weights)

    def get_Q_values(self):
        return self.Q_values

