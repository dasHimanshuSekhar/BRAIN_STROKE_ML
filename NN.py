# to load the file and convert to matxir
from numpy import loadtxt
# to choose model for NN
from tensorflow.keras.models import Sequential
# to define hidden layers and weight matrixes
from tensorflow.keras.layers import Dense
# to plot the graph
import matplotlib.pyplot as plt
# to create confusion matrix
# mport sklearn.metrics import

# Confusion Matrix
confusion_matrix = [
					["exp/prd", "   stroke_Y", "stroke_N"],
					["stroke_Y ",               0,               0],
					["stroke_N",               0,               0]
				   ]


# load the dataset TO ARRAY CONVERTION
dataset = loadtxt('trained_and_test_dataset.csv', delimiter=',')
# 80% to train
train_X = dataset[:int(len(dataset) * 0.8), 0:13] 
train_y = dataset[:int(len(dataset) * 0.8), 13]

#20% to test
test_X = dataset[-int(len(dataset) * 0.2):, 0:13] 
test_y = dataset[-int(len(dataset) * 0.2):, 13]


# define the keras model and add layers
model = Sequential()
model.add(Dense(3, input_shape=(13,), activation='tanh'))
model.add(Dense(3, activation='tanh'))
model.add(Dense(1, activation='tanh'))



# compile the keras model
model.compile(loss='mean_squared_error', metrics=['accuracy'])


# fit the keras model on the dataset
history = model.fit(train_X, train_y, epochs=100, batch_size=500)


# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------TEST---------------------------------------------
# -----------------------------------------------------------------------------------------------------


# make class predictions with the model
predictions = (model.predict(test_X) > 0.5).astype(int)
# summarize test cases for prediction
correct_prediction = 0
for i in range(len(test_X)):
	if(predictions[i] == 1):
		print('%s => %d (expected %d)' % (test_X[i].tolist(), predictions[i], test_y[i]))
	if(test_y[i] == predictions[i]):
		correct_prediction += 1
	#CONFUSION MATRIX UPDATE
	# if(predictions[i] == 0):
	# 	print('0')
	# if(predictions[i] == 1):
	# 	print('1')
	if(test_y[i] == 1 and predictions[i] == 1):
		confusion_matrix[1][1] += 1
	if(test_y[i] == 0 and predictions[i] == 1):
		confusion_matrix[1][2] += 1
	if(test_y[i] == 0 and predictions[i] == 0):
		confusion_matrix[2][2] += 1
	if(test_y[i] == 1 and predictions[i] == 0):
		confusion_matrix[2][1] += 1
print(len(test_X))
print("PREDICTION ACCURACY: ", (correct_prediction / len(test_X)) * 100)
print("ERROR RATE: ", (1 - (correct_prediction / len(test_X))) * 100)

# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------CONFUSION METRIX AND ACCURACY--------------------
# -----------------------------------------------------------------------------------------------------

print('\n\n')

for i in range(3):
	for j in range(3):
		print(confusion_matrix[i][j], end = "\t\t")
	print('\n')

# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------GRAPH--------------------------------------------
# -----------------------------------------------------------------------------------------------------



print(history.history.keys())

# HISTORY SUMMARIZE FOR MODEL [ ACCURACY V/S EPOC ]
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

# HISTORY SUMMARIZE FOR MODEL [ LOSS V/S EPOC ]
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()