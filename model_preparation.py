import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThinkness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcom"
]

data = pd.read_csv(url, names=columns_names)

x = data.drop('Outcom', axis=1)
y = data['Outcom']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
accuracy_score1 = accuracy_score(Y_test,y_pred)
print(f"Accurancy: {accuracy_score1 * 100:.2f}%")

with open('diabetes_model.pkl', 'wb') as new_model:
    pickle.dump(model, new_model)

with open("scaler.pkl", 'wb') as scaler_f:
    pickle.dump(scaler, scaler_f)