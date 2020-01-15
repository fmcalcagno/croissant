from keras import backend as K
from keras.models import Model
from keras.layers import Input, RepeatVector, CuDNNLSTM, Bidirectional, Concatenate
from keras.layers.core import Dense, Lambda
from keras import objectives
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
import tensorflow as tf

class CroissantModel:

    def __init__(self,
            input_dim,
            timesteps,
            batch_size,
            intermediate_dim,
            latent_dim,
            epsilon_std=1.,
            gpus=1,
            learning_rate=0.001):
        """
        Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder and Decoder
        :int input_dim:
        :int timesteps: input timestep dimension.
        :int batch_size:
        :int intermediate_dim: output shape of LSTM.
        :int latent_dim:   latent z-layer shape.
        :float epsilon_std:  z-layer sigma.
        :int gpus: which GPU's to use in the model
        :float learning_rate: learning rate
        """

        self.input_dim = input_dim
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.gpus = gpus
        self.learning_rate = learning_rate

    def generate_model(self):
        """
        Generate LSTM MOdel
        :param self:
        :return: VAE Model, encoder and decoder
        """
        x = Input(shape=(self.timesteps, self.input_dim,), name="Main_input_VAE")
        # LSTM encoding
        h1 = Bidirectional(
                CuDNNLSTM(self.intermediate_dim,
                    kernel_initializer='random_uniform',
                    input_shape=(self.timesteps,self.input_dim,)
                ),
                merge_mode='ave')(x)
        # VAE Z layer
        z_mean = Dense(self.latent_dim)(h1)
        z_log_sigma = Dense(self.latent_dim)(h1)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(
                shape=(self.batch_size, self.latent_dim),
                mean=0.,
                stddev=self.epsilon_std
            )
            return z_mean + z_log_sigma * epsilon

        z = Lambda(sampling,output_shape=(sef.latent_dim,)
                   )([z_mean, z_log_sigma])

        # decoded LSTM layer
        decoder_h = Bidirectional(
            CuDNNLSTM(
                self.intermediate_dim,
                kernel_initializer='random_uniform',
                input_shape=(self.timesteps,self.latent_dim,),
                return_sequences=True
            ),
            merge_mode='ave'
        )

        decoder_mean = Bidirectional(
            CuDNNLSTM(
                self.input_dim,
                kernel_initializer='random_uniform',
                input_shape=(self.timesteps, self.latent_dim,),
                return_sequences=True
            ),
            merge_mode='ave'
        )

        h_decoded = RepeatVector(self.timesteps)(z)
        h_decoded = decoder_h(h_decoded)

        # decoded layer
        x_decoded_mean = decoder_mean(h_decoded)

        # end-to-end autoencoder
        vae = Model(x, x_decoded_mean)

        # encoder, from inputs to latent space
        encoder = Model(x, z_mean)

        # generator, from latent space to reconstructed inputs
        decoder_input = Input(shape=(self.latent_dim,))

        _h_decoded = RepeatVector(self.timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)

        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        def vae_loss(x_loss, x_decoded_mean_loss):
            """
            Loss function for the Variational AUto-Encoder
            :param x_loss:
            :param x_decoded_mean_loss:
            :return:
            """
            xent_loss = objectives.mse(x_loss, x_decoded_mean_loss)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
            loss = xent_loss + kl_loss
            return loss

        opt_rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-4, decay=0)

        if self.gpus > 1:
            try:
                vae = multi_gpu_model(vae, gpus=self.gpus)
            except:
                print("Error in Multi GPU")

        vae.compile(optimizer=opt_rmsprop, loss=vae_loss)

        return vae, encoder, generator
