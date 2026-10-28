import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix

# Load the trained model
saved = joblib.load("model.pkl")

model = saved["model"]
feature_columns = saved["feature_columns"]

print("MODEL CHECK")
print("-" * 40)

print("Model type:", type(model).__name__)
print("Number of input features:", len(feature_columns))

print("\nFeatures used by the model:")
for feature in feature_columns:
    print("-", feature)

print("\nModel classes:", model.classes_)

print("\nModel feature importances:")
importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print(importance.head(15))