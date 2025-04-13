
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("emotional_impact_dataset.csv")

# Features and target
X = df.drop('stable_emotional_state', axis=1)
y = df['stable_emotional_state']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance plot
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feat_importance.plot(kind='bar', title='Feature Importance')
plt.tight_layout()
plt.show()

# Comparison of stable vs unstable emotional states
grouped = df.groupby('stable_emotional_state')[['work_productivity', 'mental_health', 'social_interaction', 'physical_health']].mean()
print(grouped)

grouped.plot(kind='bar', figsize=(10,6), title="Stable vs Unstable Emotional Health Comparison")
plt.xticks([0,1], labels=["Unstable", "Stable"], rotation=0)
plt.ylabel("Average Score")
plt.tight_layout()
plt.show()
