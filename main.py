# programming language Python can be extended by various packets
import numpy as np                      # Math stuff
import matplotlib.pyplot as plt         # Helps plotting charts
from keras.layers import Dense, Flatten # Functionality for ANNs
from keras.models import Sequential
from keras.utils import to_categorical

# Load MNIST handwritten digit data
# MNIST stands for Modified National Institute of Standards and Technology database
# See http://yann.lecun.com/exdb/mnist/

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Display some images from the training set to get an idea of it
fig, axes = plt.subplots(ncols=15, sharex=False,
			 sharey=True, figsize=(10, 4))

for i in range(15):
	axes[i].set_title(f"{y_train[i]}")
	axes[i].imshow(X_train[i], cmap='gray')
	axes[i].get_xaxis().set_visible(False)
	axes[i].get_yaxis().set_visible(False)
plt.show()

# Convert training set into one-hot encoding
# Label       0  |  1  |  2  |  3  |  4  |  5  | ... |  9
#   5         0     0     0     0     0     1     0     0
#   0     =>  1     0     0     0     0     0     0     0
#   4         0     0     0     0     1     0     0     0
# ...

# convert training set
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)

from keras.layers import Dropout

# Create simple Neural Network model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.05))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


# configure optimization method; print accuracy while learning
model.compile(loss='categorical_crossentropy',
	      optimizer='adam',
	      metrics=['acc'])

# Start the actual learning process (feedforward - backpropagation)
# TODO: hier kann man mit der Anzahl Epochen herumspielen
res = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Print loss over epochs to indicate learning progress
plt.figure(figsize=(10,8))
plt.plot(res.history['loss'],label="Training set loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# do a prediction of the test data, so that we can compare actuals vs. predictions
yt_pred_raw = model.predict(X_test)
yt_pred=np.argmax(yt_pred_raw,axis=1)
print(y_test)
print(yt_pred)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn

# Calculate confusion matrix
cm = confusion_matrix(y_test, yt_pred)

# Plot confusion matrix using sklearn's library
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# Print accuracy score
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, yt_pred)*100}%")

# Display some predictions on test data
fig, axes = plt.subplots(ncols=10, sharex=False,
			 sharey=True, figsize=(20, 4))
for i in range(10):
	axes[i].set_title(f"{y_test[i]} / {yt_pred[i]}")
	axes[i].imshow(X_test[i], cmap='gray')
	axes[i].get_xaxis().set_visible(False)
	axes[i].get_yaxis().set_visible(False)
plt.show()