from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate

import datetime
import os


class cDCGAN():
    def __init__(self,latent_dim=100):

        self.latent_dim = latent_dim
        self.n_classes = 10
        self.input_shape = (28, 28, 1)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.adversarial = self._build_adversarial()

        self.gen_opt = Adam(learning_rate=0.0008, beta_1=0.9,decay=8e-8)
        self.dis_opt = Adam(learning_rate=0.0008, beta_1=0.9, decay=8e-8)
        self.gan_opt = Adam(learning_rate=0.0008, beta_1=0.9, decay=8e-8)

        self.d_loss = []
        self.g_loss = []

        self.model_path = rf'./cDCGAN/cDCGAN_{datetime.datetime.now().strftime("%Y%%d_%H%M%S")}'

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    # define the standalone discriminator model
    def _build_discriminator(self):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.n_classes, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = self.in_shape[0] * self.in_shape[1]
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((self.in_shape[0], self.in_shape[1], 1))(li)
        # image input
        in_image = Input(shape=self.in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        # downsample
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        # downsample
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(1, activation='sigmoid')(fe)
        # define model
        model = Model([in_image, in_label], out_layer)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=self.dis_opt, metrics=['accuracy'])
        return model
