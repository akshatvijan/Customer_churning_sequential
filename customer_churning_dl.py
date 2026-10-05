"""## Supervise Models"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix,
classification_report
)

import time

print("Train:", X_train_final.shape)
print("Validation:", X_val_final.shape)
print("Test:", X_test_final.shape)

print("y_train:", y_train_final.shape)
print("y_val:", y_val_final.shape)
print("y_test:", y_test_final.shape)

"""### Logistic Regression"""

logistic_pipeline = Pipeline([
(
"model",
LogisticRegression(
max_iter=1000,
random_state=42
)
)
])

logistic_param_grid = {
"model__C": [0.1, 1, 10]
}

logistic_grid = GridSearchCV(
estimator=logistic_pipeline,
param_grid=logistic_param_grid,
cv=3,
scoring="accuracy",
n_jobs=-1
)

start_time = time.time()

logistic_grid.fit(
X_train_final,
y_train_final
)

logistic_train_time = time.time() - start_time

print("Best Parameters:")
print(logistic_grid.best_params_)

print("\nBest CV Accuracy:")
print(logistic_grid.best_score_)

print("\nTraining Time:")
print(logistic_train_time)

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
average_precision_score,
confusion_matrix,
classification_report
)

best_logistic = logistic_grid.best_estimator_

start_time = time.time()

y_val_pred = best_logistic.predict(X_val_final)
y_val_prob = best_logistic.predict_proba(X_val_final)[:, 1]

logistic_inference_time = time.time() - start_time

logistic_val_accuracy = accuracy_score(
y_val_final,
y_val_pred
)

logistic_val_precision = precision_score(
y_val_final,
y_val_pred
)

logistic_val_recall = recall_score(
y_val_final,
y_val_pred
)

logistic_val_f1 = f1_score(
y_val_final,
y_val_pred
)

logistic_val_roc_auc = roc_auc_score(
y_val_final,
y_val_prob
)

logistic_val_pr_auc = average_precision_score(
y_val_final,
y_val_prob
)


print("LOGISTIC REGRESSION - VALIDATION RESULTS")


print(f"Accuracy  : {logistic_val_accuracy:.4f}")
print(f"Precision : {logistic_val_precision:.4f}")
print(f"Recall    : {logistic_val_recall:.4f}")
print(f"F1 Score  : {logistic_val_f1:.4f}")
print(f"ROC-AUC   : {logistic_val_roc_auc:.4f}")
print(f"PR-AUC    : {logistic_val_pr_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val_final, y_val_pred))

print("\nClassification Report:")
print(
classification_report(
y_val_final,
y_val_pred
)
)

"""### Decision Tree"""

from sklearn.model_selection import train_test_split

X_train_sample, _, y_train_sample, _ = train_test_split(
X_train_final,
y_train_final,
train_size=10000,
random_state=42,
stratify=y_train_final
)

print("Original Training Data:", X_train_final.shape)
print("GridSearch Training Sample:", X_train_sample.shape)

decision_tree_pipeline = Pipeline([
  (
  "model",
  DecisionTreeClassifier(
  random_state=42
  )
  )
  ])

decision_tree_param_grid = {
  "model__criterion": ["gini", "entropy"],
  "model__max_depth": [10, 20, 30],
  "model__min_samples_split": [2, 10],
  "model__min_samples_leaf": [1, 5]
  }

decision_tree_grid = GridSearchCV(
  estimator=decision_tree_pipeline,
  param_grid=decision_tree_param_grid,
  cv=3,
  scoring="accuracy",
  n_jobs=-1,
  verbose=1
  )

  start_time = time.time()

  decision_tree_grid.fit(
  X_train_sample,
  y_train_sample
  )

  decision_tree_train_time = time.time() - start_time

  print("Best Parameters:")
  print(decision_tree_grid.best_params_)

  print("\nBest CV Accuracy:")
  print(decision_tree_grid.best_score_)

  print("\nTraining Time:")
  print(decision_tree_train_time)

best_decision_tree = decision_tree_grid.best_estimator_

  start_time = time.time()

  y_val_pred_dt = best_decision_tree.predict(X_val_final)
  y_val_prob_dt = best_decision_tree.predict_proba(X_val_final)[:, 1]

  decision_tree_inference_time = time.time() - start_time

  decision_tree_val_accuracy = accuracy_score(
  y_val_final,
  y_val_pred_dt
  )

  decision_tree_val_precision = precision_score(
  y_val_final,
  y_val_pred_dt
  )

  decision_tree_val_recall = recall_score(
  y_val_final,
  y_val_pred_dt
  )

  decision_tree_val_f1 = f1_score(
  y_val_final,
  y_val_pred_dt
  )

  decision_tree_val_roc_auc = roc_auc_score(
  y_val_final,
  y_val_prob_dt
  )

  decision_tree_val_pr_auc = average_precision_score(
  y_val_final,
  y_val_prob_dt
  )

  print("Decision Tree Validation Results")

  print(f"Accuracy  : {decision_tree_val_accuracy:.4f}")
  print(f"Precision : {decision_tree_val_precision:.4f}")
  print(f"Recall    : {decision_tree_val_recall:.4f}")
  print(f"F1 Score  : {decision_tree_val_f1:.4f}")
  print(f"ROC-AUC   : {decision_tree_val_roc_auc:.4f}")
  print(f"PR-AUC    : {decision_tree_val_pr_auc:.4f}")

  print("\nConfusion Matrix:")
  print(
  confusion_matrix(
  y_val_final,
  y_val_pred_dt
  )
  )

  print("\nClassification Report:")
  print(
  classification_report(
  y_val_final,
  y_val_pred_dt
  )
  )

  print(
  f"\nInference Time: "
  f"{decision_tree_inference_time:.4f} seconds"
  )

  print(
  f"Training Time: "
  f"{decision_tree_train_time:.4f} seconds"
  )

"""## Random Forest"""

random_forest_pipeline = Pipeline([
(
"model",
RandomForestClassifier(
random_state=42,
n_jobs=-1
)
)
])

random_forest_param_grid = {
"model__n_estimators": [100, 200],
"model__max_depth": [10, 20],
"model__min_samples_split": [2, 5],
"model__min_samples_leaf": [1, 2],
"model__class_weight": [None, "balanced"]
}

random_forest_grid = GridSearchCV(
estimator=random_forest_pipeline,
param_grid=random_forest_param_grid,
cv=3,
scoring="accuracy",
n_jobs=-1,
verbose=1
)

start_time = time.time()

random_forest_grid.fit(
X_train_sample,
y_train_sample
)

random_forest_train_time = time.time() - start_time

print("Best Parameters:")
print(random_forest_grid.best_params_)

print("\nBest CV Accuracy:")
print(random_forest_grid.best_score_)

print("\nTraining Time:")
print(random_forest_train_time)

best_random_forest = random_forest_grid.best_estimator_

start_time = time.time()

y_val_pred_rf = best_random_forest.predict(X_val_final)
y_val_prob_rf = best_random_forest.predict_proba(X_val_final)[:, 1]

random_forest_inference_time = time.time() - start_time

random_forest_val_accuracy = accuracy_score(
y_val_final,
y_val_pred_rf
)

random_forest_val_precision = precision_score(
y_val_final,
y_val_pred_rf
)

random_forest_val_recall = recall_score(
y_val_final,
y_val_pred_rf
)

random_forest_val_f1 = f1_score(
y_val_final,
y_val_pred_rf
)

random_forest_val_roc_auc = roc_auc_score(
y_val_final,
y_val_prob_rf
)

random_forest_val_pr_auc = average_precision_score(
y_val_final,
y_val_prob_rf
)

print("Random Forest Validation Results")

print(f"Accuracy  : {random_forest_val_accuracy:.4f}")
print(f"Precision : {random_forest_val_precision:.4f}")
print(f"Recall    : {random_forest_val_recall:.4f}")
print(f"F1 Score  : {random_forest_val_f1:.4f}")
print(f"ROC-AUC   : {random_forest_val_roc_auc:.4f}")
print(f"PR-AUC    : {random_forest_val_pr_auc:.4f}")

print("\nConfusion Matrix:")
print(
confusion_matrix(
y_val_final,
y_val_pred_rf
)
)

print("\nClassification Report:")
print(
classification_report(
y_val_final,
y_val_pred_rf
)
)

print(
f"\nInference Time: "
f"{random_forest_inference_time:.4f} seconds"
)

print(
f"Training Time: "
f"{random_forest_train_time:.4f} seconds"
)

"""### xgBoost"""

from xgboost import XGBClassifier

xgb_pipeline = Pipeline([
(
"model",
XGBClassifier(
random_state=42,
eval_metric="logloss",
n_jobs=-1
)
)
])

xgb_param_grid = {
"model__n_estimators": [100, 200],
"model__max_depth": [3, 5],
"model__learning_rate": [0.05, 0.1],
"model__subsample": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
estimator=xgb_pipeline,
param_grid=xgb_param_grid,
cv=3,
scoring="accuracy",
n_jobs=-1,
verbose=1
)

start_time = time.time()

xgb_grid.fit(
X_train_sample,
y_train_sample
)

xgb_train_time = time.time() - start_time

print("Best Parameters:")
print(xgb_grid.best_params_)

print("\nBest CV Accuracy:")
print(xgb_grid.best_score_)

print("\nTraining Time:")
print(xgb_train_time)

best_xgb = xgb_grid.best_estimator_

start_time = time.time()

y_val_pred_xgb = best_xgb.predict(X_val_final)
y_val_prob_xgb = best_xgb.predict_proba(X_val_final)[:, 1]

xgb_inference_time = time.time() - start_time

xgb_val_accuracy = accuracy_score(
y_val_final,
y_val_pred_xgb
)

xgb_val_precision = precision_score(
y_val_final,
y_val_pred_xgb
)

xgb_val_recall = recall_score(
y_val_final,
y_val_pred_xgb
)

xgb_val_f1 = f1_score(
y_val_final,
y_val_pred_xgb
)

xgb_val_roc_auc = roc_auc_score(
y_val_final,
y_val_prob_xgb
)

xgb_val_pr_auc = average_precision_score(
y_val_final,
y_val_prob_xgb
)

print("XGBoost Validation Results")

print(f"Accuracy  : {xgb_val_accuracy:.4f}")
print(f"Precision : {xgb_val_precision:.4f}")
print(f"Recall    : {xgb_val_recall:.4f}")
print(f"F1 Score  : {xgb_val_f1:.4f}")
print(f"ROC-AUC   : {xgb_val_roc_auc:.4f}")
print(f"PR-AUC    : {xgb_val_pr_auc:.4f}")

print("\nConfusion Matrix:")
print(
confusion_matrix(
y_val_final,
y_val_pred_xgb
)
)

print("\nClassification Report:")
print(
classification_report(
y_val_final,
y_val_pred_xgb
)
)

print(
f"\nInference Time: "
f"{xgb_inference_time:.4f} seconds"
)

print(
f"Training Time: "
f"{xgb_train_time:.4f} seconds"
)

model_comparison = pd.DataFrame({
"Model": [
"Logistic Regression",
"Decision Tree",
"Random Forest",
"XGBoost"
],
"Accuracy": [
logistic_val_accuracy,
decision_tree_val_accuracy,
random_forest_val_accuracy,
xgb_val_accuracy
],
"Precision": [
logistic_val_precision,
decision_tree_val_precision,
random_forest_val_precision,
xgb_val_precision
],
"Recall": [
logistic_val_recall,
decision_tree_val_recall,
random_forest_val_recall,
xgb_val_recall
],
"F1 Score": [
logistic_val_f1,
decision_tree_val_f1,
random_forest_val_f1,
xgb_val_f1
],
"ROC-AUC": [
logistic_val_roc_auc,
decision_tree_val_roc_auc,
random_forest_val_roc_auc,
xgb_val_roc_auc
],
"PR-AUC": [
logistic_val_pr_auc,
decision_tree_val_pr_auc,
random_forest_val_pr_auc,
xgb_val_pr_auc
],
"Training Time": [
logistic_train_time,
decision_tree_train_time,
random_forest_train_time,
xgb_train_time
],
"Inference Time": [
logistic_inference_time,
decision_tree_inference_time,
random_forest_inference_time,
xgb_inference_time
]
})

display(
model_comparison.round(4)
)

