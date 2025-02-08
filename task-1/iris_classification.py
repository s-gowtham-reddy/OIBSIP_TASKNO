import kagglehub as hub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# download iris dataset from kagglehub
iris_data_path = hub.dataset_download("uciml/iris")
# print(iris_data_path)

iris_data = pd.read_csv('iris.csv')
# print(iris_data.sample(10))
iris_inputs = iris_data.iloc[:, :-1]
iris_outputs = iris_data.iloc[:, -1]
# print(iris_inputs.sample(10))
# print(iris_outputs.sample(5))

graphs = sns.pairplot(iris_data, hue="Species")
# plt.show()

x_train, x_test, y_train , y_test = train_test_split(iris_inputs, iris_outputs, random_state=42, test_size=0.4)

model = SVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = accuracy_score( y_test, y_pred)
class_report = classification_report( y_test, y_pred)

print(f"Accuracy of the model is {acc}")
print(f"Classification report ::\n {class_report}")