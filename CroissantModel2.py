import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input,  LSTM,  Dense, Lambda, Bidirectional
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import RMSprop


class Sampling(Layer):
    """
    Sampling Layer for the VAE
    """
    def __init__(self,epsilon_std,latent_dim,name='Sampling',**kwargs):
        super(Sampling, self).__init__(name=name, **kwargs)
        self.epsilon_std,self.latent_dim=epsilon_std,latent_dim

    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, self.latent_dim),mean=0., stddev=self.epsilon_std)
        return z_mean +  z_log_sigma * epsilon

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'epsilon_std': self.epsilon_std,
            'latent_dim': self.latent_dim,
            'name': self.name
        })
        return config

class BidirectionalLstmLayer(Layer):
    """
    BiDirectional LSTM Layer for VAE
    """
    def __init__(self, intermediate_dim, timesteps, input_dim, return_sequences,name='BidirectionalLstmLayer',**kwargs):
        super(BidirectionalLstmLayer, self).__init__(name=name, **kwargs)
        self.intermediate_dim,self.timesteps=intermediate_dim,timesteps
        self.input_dim, self.return_sequences= input_dim,return_sequences
        self.layer = Bidirectional(
            LSTM(
                self.intermediate_dim,
                kernel_initializer='random_uniform',
                input_shape=(self.timesteps, self.input_dim),
                return_sequences=self.return_sequences,
                implementation=1)
            ,
            merge_mode='ave')

    def call(self, inputs):
        x = self.layer(inputs)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'intermediate_dim': self.intermediate_dim,
            'input_dim': self.input_dim,
            'name': self.name,
            'return_sequences': self.return_sequences
        })
        return config

class CroissantModel2:

    def __init__(
            self,input_dim,timesteps,batch_size,intermediate_dim,latent_dim,
            epsilon_std=1.,  learning_rate=0.001, strategy=None
    ):
        """
        Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder and Decoder
        :int input_dim:
        :int timesteps: input timestep dimension.
        :int batch_size:
        :int intermediate_dim: output shape of LSTM.
        :int latent_dim:   latent z-layer shape.
        :float epsilon_std:  z-layer sigma.
        :float learning_rate: learning rate
        :strategy: Tensorflow 2.0 distribution Strategy
        """
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.learning_rate = learning_rate
        self.strategy = strategy
        self.sampling = Sampling(epsilon_std,latent_dim,name="Sampling_VAE")
        self.lstm_encoder= BidirectionalLstmLayer(self.intermediate_dim, self.timesteps, self.input_dim, return_sequences=True,name="Encoder_Bi_LSTM")
        self.lstm_decoder_h = BidirectionalLstmLayer(self.intermediate_dim, self.timesteps, self.input_dim, return_sequences=True,name="Decoder_Bi_LSTM_h")
        self.lstm_decoder_mean = BidirectionalLstmLayer(self.intermediate_dim, self.timesteps, self.input_dim, return_sequences=True,name="Encoder_Bi_LSTM_mean")


    def generate_model(self):
        """
        Generate LSTM MOdel
        :param self:
        :return: VAE Model, encoder and decoder
        """
        #with self.strategy.scope():
        x = Input(shape=(self.timesteps, self.input_dim,), name="Main_input_VAE")
        # LSTM encoding
        h1 = self.lstm_encoder(x)
        # Latent Variables and Sampling
        z_mean,z_log_sigma = Dense(self.latent_dim)(h1), Dense(self.latent_dim)(h1)
        z = self.sampling((z_mean, z_log_sigma,))
        #LSTM Decoder
        h_decoded = self.lstm_decoder_h(z)
        x_decoded_mean = self.lstm_decoder_mean(h_decoded)

        #Model Generation
        vae = Model(x, x_decoded_mean)
        encoder = Model(x, z_mean)

        # generator, from latent space to reconstructed inputs
        decoder_input = Input(shape=(self.timesteps,self.latent_dim,))
        _h_decoded = self.lstm_decoder_h(decoder_input)
        _x_decoded_mean = self.lstm_decoder_mean(_h_decoded)

        generator = Model(decoder_input, _x_decoded_mean)

        #Optimisation Function and Loss Function
        opt_rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-4, decay=0) #opt_rmsprop = Adam(lr=self.learning_rate)
        mse_loss_fn = MeanSquaredError()
        kl_loss = - 0.5 * tf.reduce_mean(z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1)
        vae.add_loss(kl_loss)

        vae.compile(optimizer=opt_rmsprop, loss=mse_loss_fn)

        return vae, encoder, generator
