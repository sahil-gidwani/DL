# 1. Importing Libraries

import math
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Reading the dataset

df = pd.read_csv("https://raw.githubusercontent.com/sahil-gidwani/DL/main/data/GOOGL.csv")
df

# 3. EDA & Preproccesing

# Shape of the dataset (Rows, Columns)
df.shape

# Dropping 'symble' column
# df.drop(columns =['symbol'] , inplace=True)

# Searching for missing values
df.isna().sum()

df['Date'] = pd.to_datetime(df['Date'], utc=True)
df

df.info()

# 4. Visualization

sns.lineplot(data=df, x='Date', y='Open')

# 5. Prepering data for trainning and testing the model

DF = df.copy()
DF['Date'] = pd.to_datetime(DF['Date'])
# Set 'Date' column as index
DF = DF.set_index('Date')
DF

training_set = DF[:'2020'].iloc[:,0:1].values
test_set = DF['2020':].iloc[:,0:1].values

# Normalization is very important in all deep learning in general. Normalization makes the properties more consistent.
# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

timesteps = 60

"""
Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output.
For each element of training set, we have 60 previous training set elements.

"""

X_train = []
y_train = []
for i in range(timesteps,1147):
    X_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

"""
We get the test set ready in a similar way as the training set.
The following has been done so first 60 entires of test set have 60 previous values
which is impossible to get unless we take the whole 'High' attribute data for processing

"""

dataset_total = pd.concat((DF['Close'][:'2020'], DF['Close']['2020':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

# Preparing X_test
X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# 6. Creating the model

# Define the LSTM architecture
Model = Sequential()

# First LSTM layer with Dropout regularization
Model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
Model.add(Dropout(0.2))

# Second LSTM layer with Dropout regularization
Model.add(LSTM(units=100, return_sequences=True))
Model.add(Dropout(0.2))

# Third LSTM layer with Dropout regularization
Model.add(LSTM(units=100, return_sequences=True))
Model.add(Dropout(0.2))

# Fourth LSTM layer without return_sequences since it's the last LSTM layer
Model.add(LSTM(units=100))
Model.add(Dropout(0.2))

# Fully connected layers
Model.add(Dense(units=25))
Model.add(Dense(units=1))  # Output layer

# Summary of the model architecture
Model.summary()

# Compiling the model
Model.compile(optimizer= 'adam', loss = 'mean_squared_error', metrics =['accuracy'])

# Epochs and Batch Size
epochs = 15
batch_size = 32

#from keras import callbacks
#earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 2, restore_best_weights = True)

# Fitting the model
history =  Model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')

plt.title('Training loss', size=15, weight='bold')
plt.legend(loc=0)
plt.figure()

plt.show()

predicted_stock_price = Model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real Google Stock Price')
    plt.plot(predicted, color='blue',label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

# Visualizing the results for LSTM
plot_predictions(test_set, predicted_stock_price)

"""
### What is an RNN?

A Recurrent Neural Network (RNN) is a type of neural network architecture specifically designed to work with sequential data. It's called "recurrent" because it performs the same task for every element of a sequence, with the output being dependent on previous computations. This makes it well-suited for tasks such as time series prediction, natural language processing (NLP), speech recognition, and more.

### Anatomy of an RNN:

1. **Input**: At each time step \( t \), the RNN receives an input \( x_t \). This input could be a single element, such as a word in a sentence or a data point in a time series.

2. **Hidden State**: The RNN maintains a hidden state \( h_t \) that captures information about the sequence up to time \( t \). This hidden state serves as the memory of the network and is updated at each time step based on the current input and the previous hidden state. Mathematically, \( h_t = f(x_t, h_{t-1}) \), where \( f \) is a function that combines the input and the previous hidden state.

3. **Output**: At each time step, the RNN produces an output \( y_t \) based on the current input and the hidden state. This output can be used for various tasks, such as predicting the next element in the sequence or classifying the sequence.

### Training an RNN:

1. **Backpropagation Through Time (BPTT)**: Training an RNN involves optimizing its parameters (weights and biases) to minimize a loss function. Since RNNs are unfolded through time, backpropagation through time (BPTT) is used to compute gradients and update parameters by propagating the error back through the network.

2. **Vanishing and Exploding Gradients**: RNNs are prone to the vanishing and exploding gradient problems, which can make training difficult. The vanishing gradient problem occurs when gradients become extremely small as they are propagated back through time, leading to slow or ineffective learning. Conversely, the exploding gradient problem occurs when gradients become extremely large, causing unstable training. Techniques like gradient clipping and using activation functions like ReLU can mitigate these issues.

### Types of RNNs:

1. **One-to-One**: This is the simplest form of RNN, where there is a one-to-one mapping between input and output, similar to traditional feedforward neural networks.

2. **One-to-Many**: In this configuration, the RNN receives a single input and produces a sequence of outputs. An example could be generating a sequence of words given a single image.

3. **Many-to-One**: Here, the RNN processes a sequence of inputs and produces a single output. This is often used for tasks like sentiment analysis, where the goal is to classify the sentiment of a sentence.

4. **Many-to-Many (Sequence-to-Sequence)**: This type of RNN takes a sequence of inputs and produces a sequence of outputs. It's used in tasks like machine translation, where the input is a sentence in one language and the output is the same sentence translated into another language.

### Challenges and Limitations:

1. **Short-term Memory**: Traditional RNNs have difficulty learning long-term dependencies due to the vanishing gradient problem. This led to the development of more sophisticated architectures like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) to address this issue.

2. **Computational Complexity**: Training RNNs can be computationally expensive, especially with long sequences. Techniques like batching and parallelization are used to improve efficiency.

3. **Choice of Architecture**: Choosing the right architecture and hyperparameters for an RNN depends on the specific task and dataset, and often requires experimentation and tuning.

In summary, RNNs are powerful tools for working with sequential data, but they come with their own set of challenges and limitations that need to be addressed for effective use in various applications.


Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem encountered in traditional RNNs. LSTMs are particularly effective in capturing long-term dependencies in sequential data, making them well-suited for tasks such as speech recognition, language modeling, and machine translation. Let's dive into the details of LSTM:

### Components of an LSTM:

1. **Cell State (\(C_t\))**: The cell state is like a conveyor belt that runs through the entire sequence, allowing information to flow along it with minimal interference. It's regulated by three main gates:

   - **Forget Gate (\(f_t\))**: Determines what information should be discarded from the cell state. It takes as input the previous hidden state \(h_{t-1}\) and the current input \(x_t\), and outputs a number between 0 and 1 for each number in the cell state. A 1 represents "keep this information" while a 0 represents "get rid of this information".

   - **Input Gate (\(i_t\))**: Determines what new information should be stored in the cell state. It also takes \(h_{t-1}\) and \(x_t\) as input, and produces a vector of new candidate values (\(C_t'\)). 

   - **Output Gate (\(o_t\))**: Determines what information from the cell state should be output as the hidden state. It considers the current input \(x_t\) and the previous hidden state \(h_{t-1}\), and then passes the cell state through a tanh function to squish values between -1 and 1. This output is then filtered by the output gate to determine the final hidden state \(h_t\).

2. **Hidden State (\(h_t\))**: The hidden state serves as the memory of the LSTM cell. It's calculated based on the current input \(x_t\), the previous hidden state \(h_{t-1}\), and the current cell state \(C_t\). The hidden state captures relevant information from the sequence and is passed to the next time step.

### LSTM Operation:

At each time step \(t\), an LSTM cell performs the following operations:

1. **Forget Gate Operation**:
   - Compute the forget gate activation: \(f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)\).
   - Element-wise multiplication with the previous cell state: \(C_{t-1} \times f_t\).

2. **Input Gate Operation**:
   - Compute the input gate activation: \(i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)\).
   - Compute the candidate cell state: \(\tilde{C}_t = \text{tanh}(W_C \cdot [h_{t-1}, x_t] + b_C)\).

3. **Updating the Cell State**:
   - Compute the new cell state: \(C_t = C_{t-1} \times f_t + \tilde{C}_t \times i_t\).

4. **Output Gate Operation**:
   - Compute the output gate activation: \(o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)\).
   - Compute the new hidden state: \(h_t = \text{tanh}(C_t) \times o_t\).

### Training and Learning:

During training, the parameters (weights and biases) of the LSTM network are optimized using techniques like gradient descent and backpropagation through time (BPTT). The gradients are computed with respect to the loss function, and the parameters are updated to minimize this loss.

### Advantages of LSTMs:

1. **Long-term Dependencies**: LSTMs are capable of capturing long-term dependencies in sequential data, which is essential for tasks like language modeling and speech recognition.

2. **Gradient Flow**: The design of LSTM cells enables better gradient flow during training, mitigating the vanishing gradient problem encountered in traditional RNNs.

3. **Versatility**: LSTMs can be applied to a wide range of sequential data tasks and have been successful in various domains, including natural language processing, time series prediction, and more.

In summary, LSTMs are a powerful variant of recurrent neural networks that address the limitations of traditional RNNs, making them highly effective for modeling sequential data with long-range dependencies.
"""
