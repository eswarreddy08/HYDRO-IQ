#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from lazypredict.Supervised import LazyClassifier

# Load dataset
data = pd.read_csv('water_potability.csv')

print(data.columns)

print(data.info())

print(data.isna().sum()) 

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Outlier removal
z_scores = np.abs((data - data.mean()) / data.std())
data = data[(z_scores < 3).all(axis=1)]

# Feature Selection
x = data.drop(columns=['Potability'])
y = data['Potability']
skb = SelectKBest(score_func=f_classif, k=9)  # Selecting top 9 features
x_selected = skb.fit_transform(x, y)
selected_features = x.columns[skb.get_support()]
x = data[selected_features]


# Balancing the dataset
smote = SMOTETomek()
x, y = smote.fit_resample(x, y)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# LazyClassifier for quick model comparison
print("\nRunning LazyClassifier to compare models...")
lazy = LazyClassifier()
models, predictions = lazy.fit(x_train, x_test, y_train, y_test)
print(models)

# Get top 3 models from LazyClassifier
top_3_models = models.index[:3]
print(f"\nTop 3 models based on LazyClassifier: {top_3_models.tolist()}")

# Model Training: Stacking Classifier
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('et', ExtraTreesClassifier(n_estimators=100)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier())
]
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier())
stacking_clf.fit(x_train, y_train)


# Predictions
y_pred = stacking_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             precision_score, recall_score, f1_score, matthews_corrcoef)

modls={"Extra tree classifier ":ExtraTreesClassifier(n_estimators=500),
        "Random forest classifier ":RandomForestClassifier(max_depth=9),
        "Light GBM": LGBMClassifier(),
        "XGB classifier":XGBClassifier()}

results={}
for name, model in modls.items():
    print(f"Evaluating {name}")
    model.fit(x_train, y_train)
    print(f"The score of {name} model is ", model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    print('confusion matrix')
    print('------------------------------------------------')
    print(classification_report(y_test,y_pred))


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    results[name] = {
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'MCC': mcc
    }

    print(results[name])
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# ROC Curve for XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train, y_train)
y_proba_xgb = xgb_model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"XGB Classifier (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Classifier")
plt.legend(loc="lower right")
plt.show()


# Precision-Recall Curve for XGBClassifier
precision, recall, _ = precision_recall_curve(y_test, y_proba_xgb)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="XGB Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - XGBoost Classifier")
plt.legend()
plt.show()

# Feature Importance for XGBClassifier
feature_importances = xgb_model.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
plt.barh(selected_features[sorted_idx], feature_importances[sorted_idx], color=colors)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - XGBoost Classifier")
plt.show()

# User Input for Prediction
print("\nEnter values for the following features:")
input_values = [float(input(f"{feature}: ")) for feature in selected_features]

input_data = np.array([input_values])
input_data_scaled = scaler.transform(input_data)
prediction = stacking_clf.predict(input_data_scaled)

print("\nPredicted Potability:", "Potable" if prediction[0] == 1 else "Not Potable")


import pickle

# Save the scaler and stacking model
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("stacking_model.pkl", "wb") as f:
    pickle.dump(stacking_clf, f)
