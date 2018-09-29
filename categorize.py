import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.datasets import cifar10


if __name__ == '__main__' :

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


    nclasses = 10

    pos = 1

    for targetClass in range(nclasses) :
        targetIdx = []

        for i in range(len(y_train)) :
            if y_train[i][0] == targetClass :
                targetIdx.append(i)

        np.random.shuffle(targetIdx)

        for idx in targetIdx[:10]:
            img = toimage(X_train[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.show()




    img_rows, img_cols = 32, 32

    img_channels = 3

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    X_train /= 255.0
    X_test  /= 255.0

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test  = np_utils.to_categorical(y_test, nb_classes)

    # CNNを構築
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # モデルのサマリを表示
    model.summary()
    plot(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))

    # 訓練
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_split=0.1)

    # 学習履歴をプロット
    plot_history(history, result_dir)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test acc:', acc)
