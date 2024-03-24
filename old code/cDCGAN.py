import matplotlib.pyplot as plt

import os
import pickle
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Activation, UpSampling2D, Input, \
    Flatten
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as keras_utils
from tensorflow.keras import optimizers
from tensorflow.keras import datasets
# from swiss_army_tensorboard import tfboard_loggers
from tqdm import tqdm
# from cdcgan import cdcgan_models, cdcgan_utils
import cdcgan_utils
from sklearn import preprocessing
from sklearn.utils import shuffle, gen_batches
from tensorflow.keras.utils import to_categorical

ACTIVATION = layers.Activation("tanh")


class cDCGAN():
    def __init__(self):
        '''
        https://github.com/gaborvecsei/CDCGAN-Keras/blob/master/cdcgan/cdcgan_models.py
        '''
        self.gernerator_lr = 0.0002
        self.beta_1 = 0.5
        self.decay = 8e-8
        self.latence_dim = (100,)
        self.data_dim = (1000, 3, 1)
        self.condition_dim = (10,)

        self.generator = self._build_gernerator()
        self.discriminator = self._build_discriminator()

        self.adversarial = self._build_adversarial()

        self.d_losses = []
        self.g_losses = []

        self.prep = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        self.model_path = rf"./cDCGAN/cDCGAN_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    def load_data(self):

        print('loading data...', end='')

        with open(r'D:\Gary\pydata\AIMS\cDCGAN\cDCGAN-1DCNN\data\dataset_reduce_1000hz_3sensor.pkl', 'rb')as f:
            dataset = pickle.load(f)

        x_train = dataset['fft_data']  # [:,1:,1:4]

        x_temp = self.prep.fit_transform(x_train.reshape((x_train.shape[0], -1)))
        # X_train =x_temp.reshape(-1,8,8,1)
        X_train = x_temp.reshape(-1, 1000, 3, 1)

        y_train = dataset['cat_y']
        self.y_classes = dataset['cat_classes']

        print(f'X_train : {X_train.shape} ; y_train:{y_train.shape}')

        return X_train, y_train

    def _build_gernerator(self):

        self.gen_opt = Adam(0.0002, 0.5)
        # Prepare noise input
        input_z = Input(self.latence_dim)
        dense_z_1 = Dense(1024)(input_z)
        act_z_1 = ACTIVATION(dense_z_1)
        dense_z_2 = Dense(1000 * 3 * 8)(act_z_1)
        bn_z_1 = BatchNormalization()(dense_z_2)
        reshape_z = layers.Reshape((1000, 3, 8), input_shape=(1000 * 3 * 8,))(bn_z_1)

        # Prepare Conditional (label) input
        input_c = Input(self.condition_dim)
        dense_c_1 = Dense(1024)(input_c)
        act_c_1 = ACTIVATION(dense_c_1)
        dense_c_2 = Dense(1000 * 3 * 8)(act_c_1)
        bn_c_1 = BatchNormalization()(dense_c_2)
        reshape_c = layers.Reshape((1000, 3, 8), input_shape=(1000 * 3 * 8,))(bn_c_1)

        # Combine input source
        concat_z_c = layers.Concatenate()([reshape_z, reshape_c])

        # Image generation with the concatenated inputs
        up_1 = UpSampling2D(size=(2, 2))(concat_z_c)
        conv_1 = Conv2D(64, (5, 5), padding='same')(up_1)
        act_1 = ACTIVATION(conv_1)

        up_2 = UpSampling2D(size=(2, 2))(act_1)
        conv_2 = Conv2D(32, (5, 5), padding='same')(up_2)
        act_2 = Activation("tanh")(conv_2)

        conv_3 = Conv2D(32, (3, 3), strides=2, padding='same')(act_2)
        act_3 = Activation("tanh")(conv_3)

        conv_4 = Conv2D(1, (3, 3), strides=2, padding='same')(act_3)
        act_4 = Activation("tanh")(conv_4)

        model = Model(inputs=[input_z, input_c], outputs=act_4)

        model.compile(loss='binary_crossentropy', optimizer=self.gen_opt)
        return model

    def _build_discriminator(self):

        self.dis_opt = Adam(0.0002, 0.5)
        input_gen_image = Input(self.data_dim)
        conv_1_image = Conv2D(64, (5, 5), padding='same')(input_gen_image)
        act_1_image = ACTIVATION(conv_1_image)
        pool_1_image = MaxPooling2D(pool_size=(2, 2))(act_1_image)
        conv_2_image = Conv2D(128, (2, 2), padding='same')(pool_1_image)
        act_2_image = ACTIVATION(conv_2_image)
        pool_2_image = MaxPooling2D(pool_size=(2, 2), padding='same')(act_2_image)

        input_c = Input(self.condition_dim)
        dense_1_c = Dense(1024)(input_c)
        act_1_c = ACTIVATION(dense_1_c)
        dense_2_c = Dense(250 * 1 * 128)(act_1_c)
        bn_c = BatchNormalization()(dense_2_c)
        reshaped_c = layers.Reshape((250, 1, 128))(bn_c)

        concat = layers.Concatenate()([pool_2_image, reshaped_c])

        flat = Flatten()(concat)
        dense_1 = Dense(1024)(flat)
        act_1 = ACTIVATION(dense_1)
        dense_2 = Dense(1)(act_1)
        act_2 = Activation('sigmoid')(dense_2)
        model = Model(inputs=[input_gen_image, input_c], outputs=act_2)

        model.compile(loss='binary_crossentropy', optimizer=self.dis_opt)
        return model

    def _build_adversarial(self):

        self.adv_opt = Adam(0.0002, 0.5)
        input_z = Input(self.latence_dim)
        input_c = Input(self.condition_dim)
        gen_image = self.generator([input_z, input_c])
        self.discriminator.trainable = False
        is_real = self.discriminator([gen_image, input_c])
        model = Model(inputs=[input_z, input_c], outputs=is_real)
        model.compile(loss='binary_crossentropy', optimizer=self.adv_opt)
        return model

    def train(self, epochs=1000, batch_size=64):

        X_train, y_train = self.load_data()

        y_train = keras_utils.to_categorical(y_train, self.condition_dim[0])

        # tfboard_loggers.TFBoardModelGraphLogger.log_graph(os.path.join(self.model_path,"/models/logs"), K.get_session())
        # loss_logger = tfboard_loggers.TFBoardScalarLogger(os.path.join(self.model_path,"/models/logs/loss"))
        # image_logger = tfboard_loggers.TFBoardImageLogger(os.path.join(self.model_path,"/models/logs/generated_images"))
        #

        # Model Training

        iteration = 0

        nb_of_iterations_per_epoch = int(X_train.shape[0] / batch_size)
        print("Number of iterations per epoch: {0}".format(nb_of_iterations_per_epoch))

        for epoch in range(epochs):
            pbar = tqdm(desc="Epoch: {0}".format(epoch), total=X_train.shape[0])

            g_losses_for_epoch = []
            d_losses_for_epoch = []

            slices = shuffle(list(gen_batches(X_train.shape[0], batch_size)))

            # for i in range(nb_of_iterations_per_epoch):
            # for i in range(2):
            for i, slice_ in enumerate(slices):
                image_batch = X_train[slice_]
                label_batch = y_train[slice_]

                noise = cdcgan_utils.generate_noise((image_batch.shape[0], 100))

                # noise = cdcgan_utils.generate_noise((batch_size, 100))

                # image_batch = X_train[i * batch_size:(i + 1) * batch_size]
                # label_batch = y_train[i * batch_size:(i + 1) * batch_size]

                generated_images = self.generator.predict([noise, label_batch], verbose=0)

                # if i % 20 == 0:
                #     image_grid = cdcgan_utils.save_generate_mnist_image_grid(self.generator,
                #                                                         "Epoch {0}, iteration {1}".format(epoch,
                #                                                                                                 iteration),
                #                                                          epoch, i, os.path.join(self.model_path,r"images/generated_mnist_images_per_iteration"))
                # cdcgan_utils.save_generated_image(image_grid, epoch, i, os.path.join(self.model_path,"images/generated_mnist_images_per_iteration"))
                # image_logger.log_images("generated_mnist_images_per_iteration", [image_grid], iteration)

                X = np.concatenate((image_batch, generated_images))
                y = [1] * image_batch.shape[0] + [0] * image_batch.shape[0]
                label_batches_for_discriminator = np.concatenate((label_batch, label_batch))

                D_loss = self.discriminator.train_on_batch([X, label_batches_for_discriminator], y)
                d_losses_for_epoch.append(D_loss)
                # loss_logger.log_scalar("discriminator_loss", D_loss, iteration)

                noise = cdcgan_utils.generate_noise((image_batch.shape[0], 100))
                self.discriminator.trainable = False
                G_loss = self.adversarial.train_on_batch([noise, label_batch], [1] * image_batch.shape[0])
                self.discriminator.trainable = True
                g_losses_for_epoch.append(G_loss)
                # loss_logger.log_scalar("generator_loss", G_loss, iteration)

                pbar.update(image_batch.shape[0])

                iteration += 1

            # Save a generated image for every epoch
            image_grid = cdcgan_utils.save_generate_mnist_image_grid(self.generator, "Epoch {0}".format(epoch), epoch,
                                                                     0, os.path.join(self.model_path,
                                                                                     r"images/generated_mnist_images_per_epoch"))
            # cdcgan_utils.save_generated_image(image_grid, epoch, 0, os.path.join(self.model_path,"images/generated_mnist_images_per_epoch"))
            # image_logger.log_images("generated_mnist_images_per_epoch", [image_grid], epoch)

            pbar.close()
            print("D loss: {0}, G loss: {1}".format(np.mean(d_losses_for_epoch), np.mean(g_losses_for_epoch)))

            self.g_losses.append(np.mean(g_losses_for_epoch))
            self.d_losses.append(np.mean(d_losses_for_epoch))

            self.generator.save_weights(os.path.join(self.model_path, "generator.h5"))
            self.discriminator.save_weights(os.path.join(self.model_path, "discriminator.h5"))

    def predict(self, condition):
        condition_cat = to_categorical(y=condition, num_classes=10)
        gen_noise = np.random.normal(0, 1, (condition.shape[0], self.latence_dim[0]))
        pred = self.generator.predict([gen_noise, condition_cat])
        return pred

    def save_data(self):
        np.save(file=os.path.join(self.model_path, 'generator_loss.npy'), arr=np.array(self.g_losses))
        np.save(file=os.path.join(self.model_path, 'discriminator_loss.npy'), arr=np.array(self.d_losses))
        save_model(model=self.generator, filepath=os.path.join(self.model_path, 'Generator.h5'))
        save_model(model=self.discriminator, filepath=os.path.join(self.model_path, 'Discriminator.h5'))
        save_model(model=self.adversarial, filepath=os.path.join(self.model_path, 'Adversarial.h5'))

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes.plot(self.g_losses, label='g loss')
        axes.plot(self.d_losses, label='d loss')
        fig.legend()

        fig.savefig(os.path.join(self.model_path, rf'loss.png'))

    def sample(self):

        control = np.arange(10)
        pr = self.predict(control)

        fig, axes = plt.subplots(10, 3, figsize=(8, 50))
        for c_ in range(10):
            for s_ in range(3):
                ax = axes[c_, s_]
                ax.plot(pr[c_, :, s_, 0])
        fig.tight_layout()

        plt.savefig(os.path.join(self.model_path, 'pred.png'))


if __name__ == '__main__':
    cdcgan = cDCGAN()
    cdcgan.train(epochs=2500, batch_size=64)
    cdcgan.sample()











