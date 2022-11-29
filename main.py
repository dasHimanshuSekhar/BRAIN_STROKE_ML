import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

#SIGMOID FUNCTION



# load csv file <by passing path>
data = pd.read_csv('untrained_dataset.csv')

input_data = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type_0', 'work_type_1', 'work_type_2', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status_0', 'smoking_status_1']

# WEIGHTS AND BIAS MATRIX
# Between InputLayer to HiddenLayer_1
InputLayer_to_HiddenLayer1 = [[0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.82, 9], [0.9, 0.8, 0.6, 0.96, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.8, 0.9, 0.8, 9], [0.99, 0.74, 0.082, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.87, 0.88, 0.91, 0.880, 9]]
# n1H1, n2H1, n3H1 at 1st hidden layer
nH1 = [0, 0, 0]

#Between HiddenLayer_1 to HiddenLayer_2
HiddenLayer_1_to_HiddenLayer_2 = [[0.9, 0.91, 0.92, 9], [0.9, 0.91, 0.82, 9], [0.9, 0.81, 0.92, 9]];
# n1H2, n2H2, n3H2 at 2nd hidden layer
nH2 = [0, 0, 0]

#Between HiddenLayer_2 to OutputLayer
HiddenLayer_2_to_OutputLayer = [0, 0.91, 0.92, 9]
# n1H3 at 3rd output layer
nH3 = 0

# Ploting Variable: (x, y)
x = []
y = []

# ----------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------CALCULATION--------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------

# mue value
mue = 1

# CSV[COLUMN][ROW]
for times in range(0, 30000):
  for row in range(0, 1000):
    # Calculate first_layer
    for i in range(0, 3):
      nH1[i] = 0
      for col in range(0, len(input_data) - 1):
        nH1[i] += data[input_data[col]][row] * InputLayer_to_HiddenLayer1[i][col]
      nH1[i] += InputLayer_to_HiddenLayer1[i][len(InputLayer_to_HiddenLayer1) - 1]
      nH1[i] = np.tanh(nH1[i])
      
    # Calculate second_layer
    for i in range(0, 3):
      nH2[i] = 0
      for j in range(0, 3):
        nH2[i] += nH1[j] * HiddenLayer_1_to_HiddenLayer_2[i][j]
      nH2[i] += HiddenLayer_1_to_HiddenLayer_2[i][len(HiddenLayer_1_to_HiddenLayer_2) - 1]
      nH2[i] = np.tanh(nH2[i])

    # Calculate output_layer
    nH3 = 0
    for i in range(0, 3):
      nH3 += nH2[i] * HiddenLayer_2_to_OutputLayer[i]
    nH3 += HiddenLayer_2_to_OutputLayer[len(HiddenLayer_2_to_OutputLayer) - 1]
    
    output = np.tanh(nH3)
    # -------------------------------------------------ERROR CALCULATION---------------------------------------------------------------
    
    
    # error_sqr = (data['stroke'][row] - output) * (data['stroke'][row] - output)
    # error = math.sqrt(error_sqr)
    
    desired_output = data['stroke'][row]
    error = 0.5 * (desired_output - output) * (desired_output - output)

    # print("error ", error)
    
    # -----------------------------------------------BACK PROPAGATION START------------------------------------------------------------
    # -----------------------------------------------UPDATE WEIGHT MATRIX--------------------------------------------------------------
    
    
    for i in range(0, 3):
      HiddenLayer_2_to_OutputLayer[i] += 2 * mue * error * (1 - output * output) * nH2[i]
    # update bais  
    HiddenLayer_2_to_OutputLayer[3] += 2 * mue * error * (1 - output * output)
    
    # HiddenLayer_1_to_HiddenLayer_2[row] += 
    for i in range(0, 3):
      for j in range(0, 3):
        HiddenLayer_1_to_HiddenLayer_2[i][j] += 2 * mue * error * (1 - output * output) * (1 - nH2[j] * nH2[j]) * HiddenLayer_2_to_OutputLayer[j] * nH1[j]
    # update bais
      for i in range(0, 3):
        HiddenLayer_1_to_HiddenLayer_2[i][3] += 2 * mue * error * (1 - output * output) * (1 - nH2[j] * nH2[j]) * HiddenLayer_2_to_OutputLayer[j]

    # InputLayer_to_HiddenLayer1[row] += 
    for i in range(0, 3):
      for j in range(0, len(input_data) - 1):
        InputLayer_to_HiddenLayer1[i][j] += 2 * mue * error * (1 - output * output) * (1 - nH2[i] * nH2[i]) * HiddenLayer_2_to_OutputLayer[i] * (1 - nH1[i] * nH1[i]) * HiddenLayer_1_to_HiddenLayer_2[i][i] * data[input_data[j]][row]
    # update bais
    for i in range(0, 3):
      InputLayer_to_HiddenLayer1[i][13] += 2 * mue * error * (1 - output * output) * (1 - nH2[i] * nH2[i]) * HiddenLayer_2_to_OutputLayer[i] * (1 - nH1[i] * nH1[i]) * HiddenLayer_1_to_HiddenLayer_2[i][i]

    # -----------------------------------------------BACK PROPAGATION ENDED-------------------------------------------------------------
  sum = 0
  x.append(times)
  for i in range(0, 4):
    sum += HiddenLayer_2_to_OutputLayer[i]
  y.append(sum)
  plt.plot(x, y)
  print(times, "-> ", sum)
  plt.plot(x, y)

plt.show()