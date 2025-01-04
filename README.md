# Predicting_Customer_Behaviour
We are using Logistics Regression approach here where the dependent variable predicts a categorical outcome whether a customer will purchase a product or not
Note: Logistics regression is a Linear Classifier

## Dataset
_The First 15 rows of about 100 rows of data is displayed in the table_

|Age|	Estimated Salary|	Purchased|
|----|-----------------|---------|
|19|	19000|	0|
|35	|20000	|0|
|26|	43000|	0|
|27	|57000	|0|
|19|	76000|	0|
|27	|58000	|0|
|27|	84000|	0|
|32	|150000	|1|
|25|	33000|	0|
|35	|65000	|0|
|26|	80000|	0|
|26	|52000	|0|
|20|	86000|	0|
|32	|18000	|0|

## About the Dataset
_The dataset is used to predict a categorical variable of Yes and No where the categories are encoded to be 1 (for Yes) and 0 (for No). This is to use the variable Age and Estimated Salary as the predictors._

## Case Scenario
Your company where you work as a data scientist just released a new brand of furniture, and your role is to predict which of your customers can purchased the product. The result will help the advertising team target thr right customers that will buy the new product.

## Model used
This model uses the Logistics Regression algorithm to carryout the prediction.

## Python Codes

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1] .values
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)
```

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

```python
from sklearn.linear_model import LogisticRegression
#adding random state is for learning purpose to get same result on terminal
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```

## Predicting a new result
```python
# You have to transform to be in same scale as the ones already transformed
# [[takes one record, all coulumn for that record]]
y_one = classifier.predict(sc.transform([[30, 87000]]))

# in the original set, the customer did not purchase, lets see if it
# predicts [0]
print(y_one)
```

## Complete Code
+ [To view and run codes in terminal, click to View](https://colab.research.google.com/drive/12sMqLDdWv2mAWSvH_HQXgMfto6W3P4IO#scrollTo=fGpFR5pIET0L)

## Confusion Matrix in Machine Learning
A Confusion Matrix is a performance evaluation tool for classification models. It is a tabular representation that compares the actual target values with the predicted values from the model. This matrix helps measure how well a classification model performs in distinguishing between different classes.
