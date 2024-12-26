# Import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
    @staticmethod
    def build(width=128, height=128, depth=3, filters=(32, 64, 128), latent_dim=64):
        """
        Build a convolutional autoencoder.

        Args:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Number of channels in the input image.
            filters (tuple): Number of filters in each Conv2D layer.
            latent_dim (int): Dimension of the latent vector.

        Returns:
            Model: The constructed autoencoder model.
        """

        print("=="*50)
        print(f"Printing filters : {filters}")
        print(f"Printing latent dim  : {latent_dim}")
        print("=="*50)
        # Initialize the input shape as "channels last"
        input_shape = (height, width, depth)
        chan_dim = -1

        # Define the input to the encoder
        inputs = Input(shape=input_shape, name="encoder_input")
        x = inputs

        # Build the encoder
        for f in filters:
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(negative_slope=0.2)(x)
            x = BatchNormalization(axis=chan_dim)(x)

        # Flatten and construct the latent vector
        volume_size = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latent_dim, name="latent_vector")(x)

        # Build the decoder
        x = Dense(np.prod(volume_size[1:]), name="decoder_dense")(latent)
        x = Reshape((volume_size[1], volume_size[2], volume_size[3]), name="decoder_reshape")(x)

        for f in filters[::-1]:
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(negative_slope=0.2)(x)
            x = BatchNormalization(axis=chan_dim)(x)

        # Final Conv2DTranspose layer to recover the original depth
        x = Conv2DTranspose(depth, (3, 3), padding="same", name="decoder_output_conv")(x)
        outputs = Activation("sigmoid", name="decoder_output")(x)

        # Construct the autoencoder model
        autoencoder = Model(inputs, outputs, name="autoencoder")
        return autoencoder
