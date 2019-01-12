from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape,Multiply
from keras.models import Model
import keras.backend as K


def subblock(x, filter, block, num, **kwargs):
    #y = BatchNormalization()(x)
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(x)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = BatchNormalization()(y)

    spatial_attention = Conv2D(K.int_shape(y)[-1] // 2, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                               name=block + '_' + str(num) + 'sa_conv1')(y)
    spatial_attention = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', name=block + '_' + str(num) + 'sa_conv2')(spatial_attention)

    channel_attention = GlobalMaxPooling2D(name=block + '_' + str(num) + 'ca_gmp')(y)
    channel_attention = Reshape(target_shape=(-1, K.int_shape(channel_attention)[-1]), name=block + '_' + str(num) + 'ca_reshape1')(channel_attention)
    channel_attention = Dense(K.int_shape(channel_attention)[-1], activation='relu', name=block + '_' + str(num) + 'ca_dense1')(channel_attention)
    channel_attention = Dense(K.int_shape(channel_attention)[-1], activation='softmax', name=block + '_' + str(num) + 'ca_dense2')(channel_attention)
    channel_attention = Reshape(target_shape=(-1, 1, K.int_shape(channel_attention)[-1]), name=block + '_' + str(num) + 'ca_reshape2')(channel_attention)

    y = Multiply(name=block + '_' + str(num) + 'ml1')([y, channel_attention])
    y = Multiply(name=block + '_' + str(num) + 'ml2')([y, spatial_attention])

    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_model(lr, l2, img_shape=(384, 384, 3),activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 64, '1', i, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 64, '2', i,  **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 96, '3', i,  **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 128, '4', i,  **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model
