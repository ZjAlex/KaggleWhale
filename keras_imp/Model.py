from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
import keras.backend as K


def subblock(x, filter, block, num, **kwargs):
    x = BatchNormalization(name='bn_' + block + '_' + str(num) + '_1')(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', name='conv_' + block + '_' + str(num) + '_1', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization(name='bn_' + block + '_' + str(num) + '_2')(y)
    y = Conv2D(filter, (3, 3), activation='relu', name='conv_' + block + '_' + str(num) + '_2', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization(name='bn_' + block + '_' + str(num) + '_3')(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), name='conv_' + block + '_' + str(num) + '_3', **kwargs)(y)  # no activation # Restore the number of original features
    y = Add(name='add_' + block + '_' + str(num) + '_1')([x, y])  # Add the bypass connection
    y = Activation('relu', name='activation_' + block + '_' + str(num) + '_1')(y)
    return y


def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', name='conv1', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='mp1')(x)  # 96x96x64

    x = BatchNormalization(name='bn1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv2',**kwargs)(x)
    x = BatchNormalization(name='bn2')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv3',**kwargs)(x)


    x = MaxPooling2D((2, 2), strides=(2, 2), name='mp2')(x)  # 48x48x64
    x = BatchNormalization(name='bn3')(x)
    x = Conv2D(128, (1, 1), activation='relu', name='conv4', **kwargs)(x)  # 48x48x128
    for i in range(4):
        x = subblock(x, 64, '1', i, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='mp3')(x)  # 24x24x128
    x = BatchNormalization(name='bn4')(x)
    x = Conv2D(256, (1, 1), activation='relu', name='conv5', **kwargs)(x)  # 24x24x256
    for i in range(4):
        x = subblock(x, 64, '2', i, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='mp4')(x)  # 12x12x256
    x = BatchNormalization(name='bn5')(x)
    x = Conv2D(384, (1, 1), activation='relu', name='conv6', **kwargs)(x)  # 12x12x384
    for i in range(4):
        x = subblock(x, 96, '3', i, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='mp5')(x)  # 6x6x384
    x = BatchNormalization(name='bn6')(x)
    x = Conv2D(512, (1, 1), activation='relu', name='conv7', **kwargs)(x)  # 6x6x512
    for i in range(4):
        x = subblock(x, 128, '4', i, **kwargs)

    x = GlobalMaxPooling2D(name='gmp')(x)  # 512
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
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid', name='head_conv1')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid', name='head_conv2')(x)
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