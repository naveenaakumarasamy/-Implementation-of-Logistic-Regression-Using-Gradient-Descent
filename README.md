# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries: Load essential libraries (pandas numpy, and matplotlib).
2. Load the dataset: Read the Placement_Data.csv file using pandas.
3. Drop unnecessary columns: Remove the s1_no and salary columns from the dataset.
4. Convert categorical columns: Change data types of categorical columns to category.
5. Encode categorical variables: Use.cat.codes to convert categorical columns into numeric codes.
6. Define features and target: Separate features (x) and target (Y) for the model.
7. Initialize model parameters: Randomly initialize theta values for logistic regression.
8. Define sigmoid function: Create a function to compute the sigmoid of input values.
9. Define loss function: Create a function to calculate the logistic regression loss.
10. Implement gradient descent: Develop a function to update theta iteratively using gradients.
11. Train the model: Optimize theta using gradient descent with specified alpha and iterations.
12. Predict function: Create a function to predict binary outcomes using the learned theta.
13. Make predictions: Use the trained model to predict outcomes for x.
14. Evaluate accuracy: Compute model accuracy by comparing predictions to actual values.
15. Predict for new data: Use the model to make predictions for a new input example (xnew).

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Naveenaa A K
RegisterNumber:  212222230094
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Placement_Data.csv")
df

df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df.dtypes

df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y

theta = np.random.random(X.shape[1]) # intitialise the model parameter
y=Y
# define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

#define the gradient descent algorithm
def gradient_descent(theta, X,y, alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-= alpha*gradient
    return theta

#train the model
theta = gradient_descent(theta,X,y,alpha = 0.01, num_iterations = 1000)
# Make predictions
def predict(theta, X):
    h= sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
y_pred

# evaluate the model
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy",accuracy)

print(y_pred)

xnew= np.array([[0,87,0,95,0,2,0,0,1,0,0,0]])
y_prednew=predict(theta,xnew)
y_prednew
```

## Output:
![image](https://github.com/user-attachments/assets/3f41bb6f-1511-49a2-ba22-b48ab41c63f9)
![image](https://github.com/user-attachments/assets/a44d35f3-4348-4cbd-bcec-45477d49d05f)
![image](https://github.com/user-attachments/assets/fbb0b103-d4dd-4c36-a7c9-56e45611212f)
![image](https://github.com/user-attachments/assets/780190c0-3c5a-4533-b7e7-7edb2e27a148)
![image](https://github.com/user-attachments/assets/104b80a6-835a-429f-a11f-670e08773f07)
![image](https://github.com/user-attachments/assets/f599a316-dfeb-4b3b-a84d-5f18a3ef1cdc)
![image](https://github.com/user-attachments/assets/5ffba6ae-51d6-40b0-ac38-67fa3c3ab299)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

