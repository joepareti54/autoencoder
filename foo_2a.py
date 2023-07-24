# check it out : https://d1wqtxts1xzle7.cloudfront.net/56975530/Building_Autoencoders_in_Keras-libre.pdf?1531314557=&response-content-disposition=inline%3B+filename%3DBuilding_Autoencoders_in_Keras.pdf&Expires=1690145517&Signature=Sm-QsnRDecpsNudmSOovdyUJPXzOZLkxnEmeBk4xin2uWDpmmzAgGq6RtPVX~IezPlcwaSSAoSCCbNCj-j2cUVrLP-4d8jtOtluh8dHHgCbPc24fLeH0X21shMdkqtLcdg~a-Zr9m8GtSLR5GRnH07BhiK8uxnP84BvH0cj4LT9t3rdmnvz1mC0FlGltHNZoSV8XheH16X-wTXuCHPGy31BxkOmYSWSK1osbTRUyoFWno27fiipwayyYpGhTt~gMVm06rDlKGDfJ9lQVzU6KiOwbW4o1sjhPXAX7zjxv77hF7qtmr4KLL4WiHOYCll78wEfeP7~K8EnR6t5QUIzByA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
#
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
data = np.random.rand(1000, 64)  # Assuming 1000 data samples with 64 features each

input_dim = data.shape[1]
latent_dim = 32  # Adjust this according to your desired latent space size

INPUT = Input(shape=(input_dim,))
encoded = Dense(60, activation='relu')(INPUT)
encoded = Dense(58, activation='relu')(encoded)
encoded = Dense(56, activation='relu')(encoded)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(46, activation='relu')(encoded)
encoded = Dense(40, activation='relu')(encoded)
encoded = Dense(latent_dim, activation='relu')(encoded)

decoded = Dense(40, activation='relu')(encoded)
decoded = Dense(46, activation='relu')(decoded)
decoded = Dense(50, activation='relu')(decoded)
decoded = Dense(56, activation='relu')(decoded)
decoded = Dense(58, activation='relu')(decoded)
decoded = Dense(60, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)  # Use 'linear' activation here

autoencoder = Model(INPUT, decoded)
encoder = Model(INPUT, encoded) 
decoder_input = Input(shape=(latent_dim,))
decoder_layer = decoder_input
for i in range(len(autoencoder.layers) - 7, len(autoencoder.layers)):
    decoder_layer = autoencoder.layers[i](decoder_layer)
decoder = Model(decoder_input, decoder_layer)

autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.summary()
encoder.summary()
decoder.summary()
autoencoder.fit(data, data, epochs=500, batch_size=32)

latent_representation = encoder.predict(data)
predicted_data = decoder.predict(latent_representation)
AE_predicted_data = autoencoder.predict(data)
norm_A = np.linalg.norm(data)
norm_B = np.linalg.norm(predicted_data)
norm_C = np.linalg.norm(AE_predicted_data)

# Calculate the difference in norms
norm_difference_using_decoder = abs(norm_A - norm_B)
norm_difference_using_AE =      abs(norm_A - norm_C)

print('norm_difference_using_decoder ', norm_difference_using_decoder)
print('norm_difference_using_AE',       norm_difference_using_AE)
