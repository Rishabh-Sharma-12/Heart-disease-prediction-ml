# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

# %%
dataset = pd.read_csv("heart.csv")

# %%
type(dataset)

# %%
dataset.shape

# %%
dataset.head(5)

# %%
dataset.sample(5)

# %%
dataset.describe()

# %%
dataset.info()

# %%
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

# %%
dataset["target"].describe()

# %%
dataset["target"].unique()

# %%
print(dataset.corr()["target"].abs().sort_values(ascending=False))

# %%
#Exploratory Data Analysis (EDA)
#First, analysing the target variable:

# %%
y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)

# %%
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

#Alternatively,
# print("Percentage of patience with heart problems: "+str(y.where(y==1).count()*100/303))
# print("Percentage of patience with heart problems: "+str(y.where(y==0).count()*100/303))

# #Or,
# countNoDisease = len(df[df.target == 0])
# countHaveDisease = len(df[df.target == 1])

# %%
dataset["sex"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="sex", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["cp"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="cp", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["fbs"].describe()

# %%
dataset["fbs"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="fbs", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["restecg"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="restecg", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["exang"].unique()


# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="exang", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["slope"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="slope", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["ca"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="ca", y="target", data=dataset)

# Show the plot
plt.show()

# %%
dataset["thal"].unique()

# %%
y = dataset["target"]

# Create the bar plot
sns.barplot(x="thal", y="target", data=dataset)

# Show the plot
plt.show()

# %%
sns.distplot(dataset["thal"])

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='target', y='chol', data=dataset)
plt.title('Box Plot of Cholesterol Levels by Target')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='chol', y='trestbps', hue='target', data=dataset)
plt.title('Scatter Plot of Cholesterol vs. Resting Blood Pressure')
plt.show()

# %%
plt.figure(figsize=(5,5))
sns.pairplot(dataset)
plt.show()

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)


# %%
X_test.describe
X_test.shape

# %%
Y_train.describe
Y_train.shape

# %%
from sklearn.metrics import accuracy_score

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

# %%
Y_pred_lr.shape

# %%
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score is: "+str(score_lr)+" %")

# %%
from keras.models import Sequential
from keras.layers import Dense

# %%
# https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# %%
model.fit(X_train,Y_train,epochs=300)

# %%
Y_pred_nn = model.predict(X_test)

# %%
Y_pred_nn.shape

# %%
rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

# %%
score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

# %%
import tkinter as tk
from tkinter import messagebox
import joblib  # For loading your trained model
import pandas as pd

# Load the trained model and its metrics (adjust the path as necessary)
model = joblib.load("your_model_file.pkl")  # Replace with your actual model file path
accuracy = 0.85  # Example value, replace with your model's accuracy
precision = 0.87  # Example value, replace with your model's precision

# Function for prediction
def predict_heart_disease():
    try:
        # Collect inputs from the GUI
        age = int(age_entry.get())
        sex = int(sex_entry.get())
        cp = int(cp_entry.get())
        trestbps = float(trestbps_entry.get())
        chol = float(chol_entry.get())
        fbs = int(fbs_entry.get())
        restecg = int(restecg_entry.get())
        thalach = float(thalach_entry.get())
        exang = int(exang_entry.get())
        oldpeak = float(oldpeak_entry.get())
        slope = int(slope_entry.get())
        ca = int(ca_entry.get())
        thal = int(thal_entry.get())
        
        # Create a dataframe for the model
        data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
        
        # Predict
        prediction = model.predict(data)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("Heart Disease Prediction")

# Accuracy and Precision display
metrics_label = tk.Label(root, text=f"Model Accuracy: {accuracy*100:.2f}% | Precision: {precision*100:.2f}%", font=("Arial", 10), fg="blue")
metrics_label.grid(row=0, column=0, columnspan=2, pady=10)

# Input fields
tk.Label(root, text="Age: (Years)").grid(row=1, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=1, column=1)

tk.Label(root, text="Sex: (1=Male, 0=Female)").grid(row=2, column=0)
sex_entry = tk.Entry(root)
sex_entry.grid(row=2, column=1)

tk.Label(root, text="Chest Pain Type (0-3):\n0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic").grid(row=3, column=0)
cp_entry = tk.Entry(root)
cp_entry.grid(row=3, column=1)

tk.Label(root, text="Resting Blood Pressure (mmHg):").grid(row=4, column=0)
trestbps_entry = tk.Entry(root)
trestbps_entry.grid(row=4, column=1)

tk.Label(root, text="Cholesterol (mg/dl):").grid(row=5, column=0)
chol_entry = tk.Entry(root)
chol_entry.grid(row=5, column=1)

tk.Label(root, text="Fasting Blood Sugar (1=True, 0=False):").grid(row=6, column=0)
fbs_entry = tk.Entry(root)
fbs_entry.grid(row=6, column=1)

tk.Label(root, text="Resting ECG Results (0-2):\n0=Normal, 1=Abnormal, 2=Hypertrophy").grid(row=7, column=0)
restecg_entry = tk.Entry(root)
restecg_entry.grid(row=7, column=1)

tk.Label(root, text="Max Heart Rate Achieved:").grid(row=8, column=0)
thalach_entry = tk.Entry(root)
thalach_entry.grid(row=8, column=1)

tk.Label(root, text="Exercise-Induced Angina (1=Yes, 0=No):").grid(row=9, column=0)
exang_entry = tk.Entry(root)
exang_entry.grid(row=9, column=1)

tk.Label(root, text="ST Depression Induced by Exercise (oldpeak):").grid(row=10, column=0)
oldpeak_entry = tk.Entry(root)
oldpeak_entry.grid(row=10, column=1)

tk.Label(root, text="Slope of Peak Exercise ST Segment (0-2):\n0=Upsloping, 1=Flat, 2=Downsloping").grid(row=11, column=0)
slope_entry = tk.Entry(root)
slope_entry.grid(row=11, column=1)

tk.Label(root, text="Number of Major Vessels (0-4):").grid(row=12, column=0)
ca_entry = tk.Entry(root)
ca_entry.grid(row=12, column=1)

tk.Label(root, text="Thalassemia (1-3):\n1=Normal, 2=Fixed Defect, 3=Reversible Defect").grid(row=13, column=0)
thal_entry = tk.Entry(root)
thal_entry.grid(row=13, column=1)

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_heart_disease, bg="lightgreen", font=("Arial", 10))
predict_button.grid(row=14, column=0, columnspan=2, pady=10)

# Run the GUI
root.mainloop()


# %%


# %%



