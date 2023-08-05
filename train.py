import os
import pandas as pd
import numpy as np
import tensorflow as tf

from VibrationDataset import VibrationDataset
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# 設定隨機種子以便重複結果
np.random.seed(42)
tf.random.set_seed(42)



# 定義生成器模型
def build_generator(latent_dim, image_dim):

    condition_dim = 3

    latent_input = keras.Input(shape=(latent_dim,))
    condition_input = keras.Input(shape=(condition_dim,))

    # 將隨機噪聲與條件向量串聯起來
    combined_input = layers.Concatenate()([latent_input, condition_input])

    x = layers.Dense(256)(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    output = layers.Dense(image_dim, activation='tanh')(x)

    model = keras.Model(inputs=[latent_input, condition_input], outputs=output, name='generator')
    return model

# 定義判別器模型
def build_discriminator(image_dim):
    condition_dim = 3

    input_image = keras.Input(shape=(image_dim,))
    condition_input = keras.Input(shape=(condition_dim,))

    # 將圖像與條件向量串聯起來
    combined_input = layers.Concatenate()([input_image, condition_input])

    x = layers.Dense(1024)(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[input_image, condition_input], outputs=output, name='discriminator')
    return model



# 定義生成器模型
def build_generator(latent_dim, condition_dim):

    latent_input = keras.Input(shape=(latent_dim,))
    condition_input = keras.Input(shape=(condition_dim,))

    # label_onehot = tf.one_hot(condition_input, depth=condition_dim)
    model_in = tf.concat((latent_input, condition_input), axis=1)

    # x = layers.Dense(50_000 * 6 * 128)(model_in)
    x = layers.Dense(1250*64)(model_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((1250, 64))(x)
    # [None, 1250, 64 ]-> [n, 2500, 64]
    x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # [n, 2500, 64] -> [n, 5000, 64]
    x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # -> [n, 28, 28, 1]
    output = layers.Conv1D(1, kernel_size=3, padding='same', activation=keras.activations.tanh)(x)


    model = keras.Model([latent_input, condition_input], output, name="generator")
    model.summary()

    return model

# 定義判別器模型
def build_discriminator(image_dim=(5_000,1),condition_dim=3,use_bn=True):

    input_image = layers.Input(shape=image_dim)
    input_label = layers.Input(shape=condition_dim, dtype=tf.float32)
    # label_emb = layers.Embedding(3, 32)(input_label)
    emb_img = layers.Reshape((image_dim[0], image_dim[1]))(layers.Dense(image_dim[0] * image_dim[1], activation=keras.activations.relu)(input_label))
    concat_img = tf.concat((input_image, emb_img), axis=2)


    # [None, 5000, 2 -> [None, 2500, 64]
    x= layers.Conv1D(64, kernel_size=3, strides=2, padding='same')(concat_img)
    if use_bn:
        x = layers.BatchNormalization()(x)

    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.3)(x)

    #  [None, 2500, 64] -> [None, 1250, 64]
    x= layers.Conv1D(64, kernel_size=3, strides=2, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)


    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Flatten()(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[input_image, input_label], outputs=output, name='discriminator')
    model.summary()
    return model


# 建立CGAN模型
def build_cgan(generator, discriminator):
    latent_dim = 100
    condition_dim = 3

    latent_input = keras.Input(shape=(latent_dim,))
    condition_input = keras.Input(shape=(condition_dim,))

    # 生成器生成圖像
    generated_image = generator([latent_input, condition_input])

    # CGAN的判別器接受生成的圖像和條件向量作為輸入
    validity = discriminator([generated_image, condition_input])

    model = keras.Model(inputs=[latent_input, condition_input], outputs=validity, name='cgan')
    return model


class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, conditions = data

        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float64)

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator([random_latent_vectors, conditions]) # output_shape = (batch_size, 5_000, 1)

        # Assemble labels discriminating real from fake images.
        combined_labels = tf.concat( [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0 )
        combined_images = tf.concat( [generated_images, tf.cast(real_images, tf.float32)], axis=0 )
        combined_conditions = tf.concat( [conditions, conditions], axis=0 )

        # 訓練判別器
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_conditions])
            d_loss = self.loss_fn(combined_labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float64)
        # Assemble labels that say "all real images".
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, conditions])
            predictions = self.discriminator([fake_images, conditions])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

data_dir =r'D:\dataset\MAFAULDA'
use_columns= ['bearing1_axi','bearing1_rad','bearing1_tan', 'bearing2_axi','bearing2_rad','bearing2_tan']
use_columns= ['bearing1_rad']
batch_size = 128
shuffle = True
epochs = 10000

split_length=5_000
channels = len(use_columns)#6

image_dim = (split_length,channels)
condition_dim = 3
latent_dim = 100

# 建立VibrationDataset實例
myVib = VibrationDataset(data_dir, use_columns=use_columns, batch_size=batch_size,
                         shuffle=shuffle, split_length=split_length)

dataset = myVib.get_dataset()



# 建立和編譯模型
generator = build_generator(latent_dim,condition_dim)
discriminator = build_discriminator(image_dim, condition_dim=condition_dim)


# 構建和編譯CGAN模型
# cgan = build_cgan(generator, discriminator)
cgan = ConditionalGAN( discriminator=discriminator, generator=generator, latent_dim=latent_dim)

discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
discriminator.trainable = False
# cgan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

cgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)


# 定義回調函數
checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                monitor='g_loss',
                                                                mode='min',
                                                                save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping( monitor='g_loss',
                                                            patience=1000,
                                                            restore_best_weights=True)

csv_logger_callback = tf.keras.callbacks.CSVLogger('training_log.csv')

# 訓練CGAN模型
epochs = 10_000
history = cgan.fit( dataset,
                    epochs=epochs,
                    callbacks=[model_checkpoint_callback, csv_logger_callback])

# 保存生成器的權重
generator.save_weights('cgan_generator.h5')