import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.001
training_epochs = 6
batch_size = 600

# Import the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train.shape
x_test.shape

# Plot sample images
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')

plt.show()

# Prepare the dataset
train_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.reshape(x_train, [-1, 784]), y_train))
    .batch(batch_size)
    .shuffle(1000)
)

train_dataset = (
    train_dataset.map(lambda x, y:
                      (tf.divide(tf.cast(x, tf.float32), 255.0),
                       tf.reshape(tf.one_hot(y, 10), (-1, 10)))
                      )
)

# Initialize weights and bias
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define model and loss
model = lambda x: tf.nn.softmax(tf.add(tf.matmul(x, w), b))
compute_loss = lambda true, pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(true, pred))
compute_accuracy = lambda true, pred: tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true, pred))

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate)

# Training loop
for epoch in range(training_epochs):
    for i, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = compute_loss(y, pred)

        acc = compute_accuracy(y, pred)
        grads = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(grads, [w, b]))
        print('=> Epoch %d | Batch %d | Loss: %.2f | Accuracy: %.2f' % (epoch + 1, i + 1, loss.numpy(), acc.numpy()))

# Prepare test data
x_test = tf.cast(x_test, tf.float32)
x_test = tf.reshape(x_test, [-1, 28 * 28])
x_test = tf.divide(x_test, 255.0)

# Prediction
prediction = model(x_test)

# Display results
plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(np.reshape(x_test[i].numpy(), [28, 28]), cmap='gray')
    plt.xlabel(np.argmax(prediction[i].numpy()))
    plt.xticks([])
    plt.yticks([])
plt.show()
