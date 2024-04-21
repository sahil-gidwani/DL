import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/sahil-gidwani/DL/main/data/letter-recognition.csv")
df.head()

df.isnull().sum()

df.info()

x=df.iloc[:,1:].values
y=df.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

# model=SVC()
# model

from tensorflow.keras import models, layers

num_classes = len(np.unique(y))
input_shape = x_train.shape[1:]

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

y_pred=model.predict(x_test)

y_test

y_pred

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predict on test data
y_pred = model.predict(x_test)

# Convert predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

# Calculate macro-averaged precision
precision = precision_score(y_test, y_pred_classes, average='macro')
print("Precision (macro-averaged):", precision)

# Calculate macro-averaged recall
recall = recall_score(y_test, y_pred_classes, average='macro')
print("Recall (macro-averaged):", recall)

# Calculate macro-averaged F1-score
f1 = f1_score(y_test, y_pred_classes, average='macro')
print("F1-Score (macro-averaged):", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

"""
Multiclass classification is a type of machine learning task where the goal is to classify input instances into one of three or more possible classes. In other words, the model must learn to predict the correct class label from a set of mutually exclusive classes.

Here's a detailed explanation of multiclass classification:

1. **Definition**: In a multiclass classification problem, each instance belongs to one and only one class out of three or more possible classes. For example, classifying images of animals into categories such as "cat," "dog," "bird," "horse," etc., is a multiclass classification problem.

2. **Output**: The output of a multiclass classification model is a single predicted class label for each input instance. The model assigns a probability or confidence score to each class, indicating the likelihood that the instance belongs to that class. The class with the highest probability or score is chosen as the predicted class label.

3. **Evaluation Metrics**: Common evaluation metrics for multiclass classification include accuracy, precision, recall, F1 score, and confusion matrix. These metrics help assess the performance of the model in correctly identifying instances belonging to different classes.

4. **Algorithms**: Various machine learning algorithms can be used for multiclass classification, including but not limited to:
   - Logistic Regression
   - Decision Trees
   - Random Forests
   - Support Vector Machines (SVM)
   - Neural Networks (including deep learning models)

5. **One-vs-All (OvA) vs. One-vs-One (OvO)**: There are different strategies for handling multiclass classification with binary classification algorithms. In the One-vs-All (OvA) approach, a separate binary classifier is trained for each class, where the instances of that class are treated as positive examples, and instances of all other classes are treated as negative examples. In the One-vs-One (OvO) approach, a binary classifier is trained for every pair of classes to distinguish between them.

6. **Handling Imbalance**: Imbalance in the distribution of classes can affect the performance of multiclass classification models. Techniques such as class weighting, oversampling, undersampling, and using ensemble methods can help address class imbalance issues.

7. **Applications**: Multiclass classification is widely used in various real-world applications, including image classification, natural language processing (e.g., sentiment analysis, topic classification), medical diagnosis, and fraud detection, among others.

Overall, multiclass classification is a fundamental task in machine learning, and understanding its principles and techniques is essential for building effective predictive models for diverse applications.

Classification evaluation metrics are used to assess the performance of machine learning models in classification tasks, where the goal is to predict the class label of instances based on their features. These metrics provide insights into how well the model is performing and help in comparing different models or tuning their parameters. Here are some common classification evaluation metrics:

1. **Accuracy**: Accuracy measures the proportion of correctly classified instances out of the total number of instances. It is calculated as the ratio of the number of correct predictions to the total number of predictions.

   \[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \]

   Accuracy is a widely used metric when classes are balanced, meaning that each class occurs with similar frequency in the dataset. However, it can be misleading in imbalanced datasets, where one class dominates the others.

2. **Precision**: Precision measures the proportion of true positive predictions out of all positive predictions. It quantifies the model's ability to avoid false positives.

   \[ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \]

3. **Recall (Sensitivity)**: Recall, also known as sensitivity or true positive rate, measures the proportion of true positive predictions out of all actual positive instances. It quantifies the model's ability to capture all positive instances.

   \[ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \]

4. **F1 Score**: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance, considering both false positives and false negatives.

   \[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]

5. **Specificity**: Specificity measures the proportion of true negative predictions out of all actual negative instances. It quantifies the model's ability to correctly identify negative instances.

   \[ \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}} \]

6. **False Positive Rate (FPR)**: FPR measures the proportion of false positive predictions out of all actual negative instances. It is complementary to specificity.

   \[ \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}} \]

7. **Receiver Operating Characteristic (ROC) Curve**: The ROC curve is a graphical representation of the trade-off between true positive rate (sensitivity) and false positive rate (1 - specificity) for different threshold values. It helps in assessing the model's performance across various thresholds.

8. **Area Under the ROC Curve (AUC-ROC)**: AUC-ROC measures the area under the ROC curve. It provides a single value to summarize the model's performance across all possible threshold values. A higher AUC-ROC indicates better overall performance.

These are some of the key classification evaluation metrics used to assess the performance of machine learning models. The choice of metrics depends on the specific characteristics of the dataset and the objectives of the classification task.
"""
