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

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

model=SVC()
model

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

score=accuracy_score(y_test,y_pred)
print(score)

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

SVC stands for Support Vector Classifier, which is a type of supervised learning algorithm used for classification tasks. It is a variant of Support Vector Machine (SVM) specifically designed for classification problems. Here's a detailed explanation of SVC:

1. **Support Vector Machine (SVM)**: Before delving into SVC, it's essential to understand SVM. SVM is a powerful and versatile algorithm used for both classification and regression tasks. Its primary objective is to find the hyperplane that best separates the classes in the feature space. SVM works by maximizing the margin, which is the distance between the hyperplane and the nearest data points (support vectors) of each class.

2. **Classification with SVC**: SVC extends the principles of SVM to solve classification problems. Given a set of labeled training data, where each instance belongs to one of two classes, SVC learns to build an optimal hyperplane that separates the classes in the feature space. If the classes are linearly separable, SVC finds a linear decision boundary. However, it can also handle non-linear decision boundaries using techniques like the kernel trick.

3. **Kernel Trick**: The kernel trick is a fundamental concept in SVM and SVC. It allows SVC to implicitly map the input features into a higher-dimensional space, where the classes may be linearly separable. Common kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid. These kernels enable SVC to capture complex relationships between features and achieve better classification performance.

4. **Hyperparameters**: SVC has several hyperparameters that can be tuned to optimize its performance, including:
   - C: Penalty parameter that controls the trade-off between maximizing the margin and minimizing classification errors. A smaller C value leads to a softer margin, allowing more misclassifications, while a larger C value leads to a harder margin, minimizing misclassifications.
   - Kernel: Specifies the type of kernel function used for mapping the data into a higher-dimensional space.
   - Gamma: Parameter for RBF kernel, which defines the influence of a single training example. Higher gamma values lead to more complex decision boundaries.

5. **Binary and Multiclass Classification**: SVC is primarily used for binary classification tasks, where the goal is to classify instances into one of two classes. However, it can also be extended to handle multiclass classification using strategies like one-vs-one (OvO) or one-vs-all (OvA), where multiple binary classifiers are trained to distinguish between pairs of classes.

6. **Scalability and Performance**: SVC can be computationally intensive, especially for large datasets, as it involves solving a quadratic optimization problem. However, with efficient optimization algorithms and kernel approximations, it can handle reasonably large datasets. Additionally, SVC tends to perform well in practice, especially in scenarios with high-dimensional feature spaces and complex decision boundaries.

Overall, SVC is a versatile and powerful classification algorithm that is widely used in various machine learning applications, including text classification, image recognition, bioinformatics, and more. Understanding its principles and parameters is crucial for effectively applying it to different classification tasks.

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
