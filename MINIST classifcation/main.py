import tensorflow as tf
import matplotlib.pyplot as plt

# 資料預處理: 引入mnist資料集
(x_train_image, y_train_label), (x_test_image, y_test_label) = tf.keras.datasets.mnist.load_data()

# Build model
# CNN: mnist shape(28x28x1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# dense layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Compile and train the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(x_train_image, y_train_label, epochs=10,
                    validation_data=(x_test_image, y_test_label))

model.summary()

# 使用matplotlib.pyplot製作表格
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# history
history.history

"""
    print('train data=',len(x_train_image))
    print('test data=',len(x_test_image))
    print('x_train_image :', x_train_image.shape)
    print('y_train_label :', y_train_label.shape)
"""

# 資料預處理: features(數字影像的特徵值)--參考網址: https://waternotetw.blogspot.com/2018/03/keras-mnist.html
# tensorflow support callback: https://www.tensorflow.org/tutorials/images/cnn
