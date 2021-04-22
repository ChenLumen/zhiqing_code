from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
import numpy as np
import mne
from sklearn.model_selection import train_test_split


# batch_size nb_epoch可以按照需求修改成自己合适的数值
batch_size = 128
nb_epoch = 100

# Parameters for mne dataset
input_channels_num = 20
output_channel_num = 12

# 第一个隐藏层和第二个隐藏层的神经元数量由于没有像原文那样数据展开，
# 所以有所不同。可以酌情修改
encoding_dim = 30
decoding_dim = 10

raw = mne.io.read_raw_brainvision("data/resting_state/zavrin_open_eyes_eeg_15021500.vhdr", preload=True)
data = raw.get_data().T
print(data.shape)

# Build autoencoder model

# 由于原文中对激活函数的描述很不详细，只知道用了tanh函数，
# 而不知道在哪一层用了哪一层没用，所以我都加上了，可以根据效果作出修改
input_ = Input(shape=(input_channels_num,))
hidden_1 = Dense(encoding_dim, activation='tanh')(input_)
encoded = Dense(output_channel_num, activation='tanh')(hidden_1)

input_encoded = Input(shape=(output_channel_num,))
hidden_2 = Dense(decoding_dim, activation='tanh')(input_encoded)
decoded = Dense(output_channel_num, activation='tanh')(hidden_2)

encoder = Model(input_, encoded, name="encoder")
decoder = Model(input_encoded, decoded, name="decoder")

autoencoder = Model(input_, decoder(encoder(input_)), name="autoencoder")

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

X_train, X_test, _, _ = train_test_split(data, data, test_size=0.3)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print(X_train.shape)
print(X_test.shape)

# 这是这个代码作者加的他的标签数据，看不懂的话就算了
X_train_noise = X_train + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noise = X_test + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noise = np.clip(X_train_noise, 0., 1.)
X_test_noise = np.clip(X_test_noise, 0., 1.)
print(X_train_noise.shape)
print(X_test_noise.shape)


# Train
# 把相应的训练数据和target改为自己的数据即可，
# 根据我的理解，训练数据就是你的EEG数据，标签数据就是sEEG数据
autoencoder.fit(X_train_noise, X_train, verbose=1,
                validation_data=(X_test_noise, X_test))