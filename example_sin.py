import numpy as np
import neural_network
import optimizer
import preprocess
import utils
import matplotlib.pyplot as plt

train_x = np.random.rand(1000, 1) * 10. - 5.
train_y = np.sin(train_x)

val_x = np.random.rand(100, 1) * 10. - 5.
val_y = np.sin(val_x)
#Develop the model
model = neural_network.NeuralNetwork()
model.add_layer(input_size=1, units=5, activation_fn='relu', initializer='glorot')
model.add_layer(units=3, activation_fn='relu')
model.add_layer(units=1)
criterion = neural_network.MSELoss()
optim = optimizer.AdamOptimizer
learning_rate = 0.001
opt = optim(model = model, criterion=criterion, learning_rate=learning_rate)

#Train the model
NUM_EPOCHS = 10000
opt.fit_batch(train_x, train_y, epochs=NUM_EPOCHS)

#Calculate metics on validation set
y_hat = model.forward(val_x.T)
val_error = criterion(y_hat, val_y.T)

print('Validation error: ', val_error)


plt.scatter(val_x.squeeze(), val_y.squeeze(), label='Orig')
plt.scatter(val_x.squeeze(), y_hat.squeeze(), label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()