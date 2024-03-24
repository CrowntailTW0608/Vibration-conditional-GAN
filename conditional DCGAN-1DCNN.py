from sklearn import preprocessing
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Activation, Input, Concatenate, LeakyReLU
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle, gen_batches
from sklearn.preprocessing import LabelEncoder
import os
import datetime
import matplotlib.pyplot as plt

import pickle

from tqdm import tqdm


class cDCGAN():
    def __init__(self):

        self.model_type = "cDCGAN-1D_NORMAL_IMBALANCE_HMIS_VMIS"
        self.data_path = r'./data/ary_NORMAL_IMBALANCE_50x_rpmGT50.pkl'


        self.gernerator_lr = 0.0002
        self.beta_1 = 0.5
        self.decay = 8e-8
        self.latence_dim = (100,)
        self.data_dim = (5000, 3)
        self.condition_dim = (2,)
        self.num_classes = 2
        self.discriminator_output = 2*self.num_classes

        self.generator = self._build_gernerator()
        self.discriminator = self._build_discriminator()

        self.adversarial = self._build_adversarial()



        self.d_losses = []
        self.g_losses = []

        self.prep = preprocessing.MinMaxScaler(feature_range=(0, 1))

        self.le = LabelEncoder()
        self.model_path = rf"./exp/{self.model_type}/{self.model_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.isdir(self.model_path + r'\model'):
            os.makedirs(self.model_path + r'\model')

        with open(os.path.join(self.model_path, 'loss.txt'), 'a')as f:
            f.write('epoch,d_loss,g_loss\n')

        with open(os.path.join(self.model_path, 'model_summary.txt'), 'a') as f:

            self.generator .summary(print_fn=lambda x: f.write(x + '\n'))
            self.discriminator .summary(print_fn=lambda x: f.write(x + '\n'))
            self.adversarial .summary(print_fn=lambda x: f.write(x + '\n'))

            f.write(f'data_path {self.data_path} \n')
            f.write(f'data_dim {self.data_dim} \n')
            f.write(f'latence_dim {self.latence_dim} \n')
            f.write(f'condition_dim {self.condition_dim} \n')


    def load_data(self):

        print('loading data...', end='')
        with open(self.data_path,'rb')as f:
            dataset = pickle.load(f)

        x_train = dataset['data'][:, :, :5000]

        x_temp = self.prep.fit_transform(x_train.reshape((x_train.shape[0], -1)))
        X_train = x_temp.reshape(-1, self.data_dim[1], self.data_dim[0])
        X_train = np.moveaxis(X_train, 1, -1)

        y_train = self.le.fit_transform(dataset['label']['cats'])

        self.y_classes = self.le.classes_

        print(f'X_train : {X_train.shape} ; y_train:{y_train.shape}')
        print(y_train)
        print(self.y_classes)
        return X_train, y_train

    def _build_gernerator(self, ):

        self.gen_opt = Adam(learning_rate=0.0002, beta_1=0.5, decay=8e-8)

        self.gen_input_shape = self.latence_dim
        self.condition_input_shape = self.condition_dim

        noise_input_layer = Input(shape=self.gen_input_shape)
        condition_input_layer = Input(shape=self.condition_input_shape)

        # noise
        hid = Dense(self.data_dim[0] * self.data_dim[1])(noise_input_layer)
        hid = BatchNormalization(momentum=0.9)(hid)
        noise_hid = LeakyReLU(alpha=0.1)(hid)

        #condition
        hid = Dense(self.gen_input_shape[0])(condition_input_layer)
        hid = BatchNormalization(momentum=0.9)(hid)
        condition_hid = LeakyReLU(alpha=0.1)(hid)

        hid = Concatenate()([noise_hid, condition_hid])

        hid = Dense(self.data_dim[0]* self.data_dim[1])(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        hid = Reshape((self.data_dim[0], self.data_dim[1]))(hid)

        hid = Conv1DTranspose(16, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        hid = Conv1DTranspose(32, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        hid = Conv1DTranspose(32, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        hid = Conv1DTranspose(64, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)        #

        hid = Conv1DTranspose(3, kernel_size=1, strides=1, padding="same")(hid)
        out = Activation("sigmoid")(hid)

        generator = Model(inputs=[noise_input_layer,condition_input_layer], outputs=out ,name='Generator')
        generator.compile(loss='categorical_crossentropy', optimizer=self.gen_opt)

        generator.summary()

        return generator

    def _build_discriminator(self):

        self.dis_opt = Adam(learning_rate=0.0002, beta_1=0.5, decay=8e-8)

        self.dis_img_input_shape = self.data_dim
        self.condition_input_shape = self.condition_dim

        img_input_layer = Input(shape=self.dis_img_input_shape)
        condition_input_layer = Input(shape=self.condition_input_shape)


        hid = Conv1D(64, kernel_size=3, strides=1, padding='same')(img_input_layer)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        hid = Conv1D(32, kernel_size=3, strides=1, padding='same')(img_input_layer)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        hid = Conv1D(16, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        img_hid = Flatten()(hid)

        hid = Concatenate()([img_hid, condition_input_layer])

        hid = Dense(20)(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)

        out = Dense(2*self.num_classes, activation='softmax')(hid)
        discriminator = Model(inputs=[img_input_layer,condition_input_layer], outputs=out,name='Discriminator')

        discriminator.compile(optimizer=self.dis_opt, loss='categorical_crossentropy' ,metrics=['accuracy'])
        discriminator.summary()

        return discriminator

    def _build_adversarial(self):

        print('buliding asvsersarial...')

        self.adv_opt = Adam(learning_rate=0.0002, beta_1=0.5, decay=8e-8)
        self.discriminator.trainable = False

        gan_input = Input(shape=self.latence_dim)
        condition_input = Input(shape=self.condition_dim)

        x = self.generator([gan_input,condition_input])
        out = self.discriminator([x,condition_input])

        adversarial = Model(inputs=[gan_input,condition_input], outputs=out, name='Adversarial')
        adversarial.compile(loss='categorical_crossentropy', optimizer=self.adv_opt)
        adversarial.summary()

        return adversarial

    # 低依次train (未儲存) 250epoch 最低 d_loss : 8.x ; g_loss : 0.000002
    def train(self, epochs=1000, batch_size=4):

        X_train, y_train = self.load_data()

        # nb_of_iterations_per_epoch = int(X_train.shape[0] / batch_size)

        for epoch in range(epochs):
            print(f'epoch {epoch}\t', end='')
            g_loss, d_loss, d_acc = 0, 0,0

            slices = shuffle(list(gen_batches(X_train.shape[0], batch_size)))

            with tqdm(total=len(slices), position=0, leave=True)as pbar:
                for i, slice_ in enumerate(slices):

                    self.discriminator.trainable = True
                    # train discriminator
                    X_train_batch = X_train[slice_]
                    y_train_batch = y_train[slice_]

                    #合法
                    legit_images = X_train_batch
                    legit_y_1 = to_categorical(y=y_train_batch,num_classes=  self.num_classes)#np.ones((len(X_train_batch)),)
                    legit_y_2 = to_categorical(y=y_train_batch,num_classes=2*self.num_classes)#np.ones((len(X_train_batch)),)

                    # 合成
                    gen_noise = np.random.normal(0, 1, (X_train_batch.shape[0], self.latence_dim[0]))
                    synthetic_y_2 = to_categorical(y=y_train_batch+self.num_classes,num_classes=2*self.num_classes)
                    synthetic_images = self.generator.predict( [gen_noise,legit_y_1])


                    images = np.vstack( ( legit_images, synthetic_images ) )
                    labels_1 = np.vstack( ( legit_y_1, legit_y_1 ) )
                    labels_2= np.vstack( ( legit_y_2, synthetic_y_2 ) )

                    d_loss_tmp, d_acc_tmp = self.discriminator.train_on_batch([images,labels_1],labels_2)
                    d_loss += d_loss_tmp
                    d_acc += d_acc_tmp

                    # train generator
                    self.discriminator.trainable = False
                    for j in range(2):
                        gen_noise = np.random.normal(0, 1, (X_train_batch.shape[0], self.latence_dim[0]))
                        g_loss += self.adversarial.train_on_batch([gen_noise,to_categorical(y=y_train_batch,num_classes=self.num_classes)],
                                                                  to_categorical(y=y_train_batch,num_classes=2*self.num_classes))

                    pbar.update(1)

            d_loss = d_loss / (i + 1)
            d_acc = d_acc / (i+1)
            g_loss = g_loss / (i + 1) / 2

            print('epoch: %d  ,[Discriminator :: d_loss: %f  d_acc :%f], [ Generator :: loss: %f]' % (epoch, d_loss,d_acc, g_loss))

            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)

            np.save(file=os.path.join(self.model_path, r'model\generator_loss.npy'), arr=np.array(self.g_losses))
            np.save(file=os.path.join(self.model_path, r'model\discriminator_loss.npy'), arr=np.array(self.d_losses))

            with open(os.path.join(self.model_path, 'loss.txt'), 'a')as f:
                f.write(f'{epoch},{d_loss},{g_loss}\n')

            if epoch % 5 == 0:

                sample_per_class = 6



                fig, axes = plt.subplots(self.num_classes, sample_per_class, figsize=(30, 10))

                for class_ in range(self.num_classes):

                    noise = np.random.normal(0, 1, (sample_per_class, 100))
                    gen_imgs = self.generator.predict([noise,to_categorical(y=[class_]*sample_per_class,num_classes=self.num_classes) ])
                    for c in range(sample_per_class):
                        ax = axes[class_][c]
                        ax.plot(gen_imgs[c, ...])
                        ax.set_title(f'generated data {self.y_classes[class_]}')

                fig.tight_layout()

                fig.savefig("{0}/{1}_{2}_gen_img.png".format(self.model_path, epoch, epoch))

                plt.close()

                fig, axes = plt.subplots(1, 1)

                axes.plot(self.g_losses, label='g_loss')
                axes.plot(self.d_losses, label='d_loss')
                fig.legend()
                fig.suptitle('loss')
                fig.tight_layout()
                fig.savefig("{0}/{1}_loss.png".format(self.model_path, epoch))

                plt.close()

                self.generator.save_weights(os.path.join(self.model_path, rf"model\{epoch}_generator_weights.ckpt"))
                self.discriminator.save_weights(
                    os.path.join(self.model_path, rf"model\{epoch}_discriminator_weights.ckpt"))

        self.save_data()

    def predict(self, condition):
        condition_cat = to_categorical(y=condition, num_classes=self.num_classes)
        gen_noise = np.random.normal(0, 1, (condition.shape[0], self.latence_dim[0]))
        pred = self.generator.predict(gen_noise)
        return pred

    def save_data(self):
        np.save(file=os.path.join(self.model_path, r'model\generator_loss.npy'), arr=np.array(self.g_losses))
        np.save(file=os.path.join(self.model_path, r'model\discriminator_loss.npy'), arr=np.array(self.d_losses))

        self.generator.save_weights(os.path.join(self.model_path, r"model\last_generator_weights.ckpt"))
        self.discriminator.save_weights(os.path.join(self.model_path, r"model\last_discriminator_weights.ckpt"))
        self.adversarial.save_weights(os.path.join(self.model_path, r"model\last_adversarial_weights.ckpt"))

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes.plot(self.g_losses, label='g loss')
        axes.plot(self.d_losses, label='d loss')
        fig.legend()

        fig.savefig(os.path.join(self.model_path, r'loss.png'))
        plt.close()

    def sample(self, filename: str = ''):

        control = np.arange(self.num_classes)
        pred = self.predict(control)

        fig, axes = plt.subplots(4, 3, figsize=(8, 15))
        for c_ in range(4):
            for s_ in range(3):
                ax = axes[c_, s_]
                ax.plot(pred[c_, s_, :, 0])
        fig.tight_layout()

        if filename != '':
            plt.savefig(os.path.join(self.model_path, f'{filename}.png'))
        else:
            plt.savefig(os.path.join(self.model_path, 'pred.png'))

        plt.close

    def load_generator(self, path):

        self.generator.load_weights(path)
        self.generator.summary()

    def load_discriminator(self, path):

        self.discriminator.load_weights(path)
        self.discriminator.summary()


if __name__ == '__main__':
    cdcgan = cDCGAN()


    cdcgan.train(epochs=1000, batch_size=64)
    # cdcgan.sample()

    # cdcgan.save_data()

























