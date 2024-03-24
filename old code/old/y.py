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
def build_generator():
    latent_dim = 100
    condition_dim = 9

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

    output = layers.Dense(5_000, activation='tanh')(x)

    model = keras.Model(inputs=[latent_input, condition_input], outputs=output, name='generator')
    return model

# 定義判別器模型
def build_discriminator():
    condition_dim = 9

    input_image = keras.Input(shape=(5_000,1))
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


data_dir =r'D:\dataset\MAFAULDA'
use_columns= ['bearing1_axi','bearing1_rad','bearing1_tan', 'bearing2_axi','bearing2_rad','bearing2_tan']
use_columns= ['bearing1_rad']
batch_size = 128
shuffle = True
epochs = 10000

split_length=5_000
channels = len(use_columns) #1

image_dim = (split_length,channels)
condition_dim = 7+2
latent_dim = 100

# 建立VibrationDataset實例
myVib = VibrationDataset(data_dir, use_columns=use_columns, batch_size=batch_size,
                         shuffle=shuffle, split_length=split_length)
dataset= myVib.get_dataset()




# 建立和編譯模型
generator = build_generator()
discriminator = build_discriminator()


# 構建和編譯CGAN模型
cgan = build_cgan(generator, discriminator)
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
discriminator.trainable = False
cgan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))


# 加入EarlyStopping回調函數
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)

# 訓練CGAN模型
epochs = 10000
for epoch in range(epochs):
    print('epoch : ',epoch)
    for batch,(x_batch, y_batch) in enumerate(dataset):

        # 產生隨機噪聲和條件向量
        noise = np.random.normal(0, 1, (batch_size, 100))
        condition = y_batch#np.random.random((batch_size, 3))

        # 生成器生成圖像
        generated_images = generator.predict([noise, condition])

        # 選擇真實圖像
        real_images = x_batch

        # 訓練判別器
        d_loss_real = discriminator.train_on_batch([real_images, condition], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([generated_images, condition], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 訓練生成器
        noise = np.random.normal(0, 1, (batch_size, 100))
        condition = np.random.random((batch_size, 3))
        valid_labels = np.ones((batch_size, 1))
        g_loss = cgan.train_on_batch([noise, condition], valid_labels)


    # 在每個epoch輸出loss
    if epoch % 50 == 0:
        print(f"train           [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        generator.save_weights('cgan_generator_{}.h5'.format(epoch))
    # 使用EarlyStopping判斷是否提前停止訓練（可選）
    # if epoch > 0 and epoch % 1000 == 0:
    #     val_loss = g_loss_val  # 這裡使用生成器的損失作為EarlyStopping的指標，你也可以使用其他指標
    #     if early_stopping.on_epoch_end(epoch, logs={'val_loss': val_loss}):
    #         print("Early stopping. Training stopped.")
    #         break


# 保存生成器的權重
generator.save_weights('cgan_generator.h5')

