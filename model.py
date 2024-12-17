import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
data = pd.read_csv(r'C:\dev\ml\project\dataframe2.csv')
label_encoder_cpu = LabelEncoder()
label_encoder_gpu = LabelEncoder()
label_encoder_company = LabelEncoder()
label_encoder_os = LabelEncoder()
data['Cpu_brand'] = label_encoder_cpu.fit_transform(data['Cpu_brand'])
data['Gpu_brand'] = label_encoder_gpu.fit_transform(data['Gpu_brand'])
data['Company'] = label_encoder_company.fit_transform(data['Company'])
data['os'] = label_encoder_os.fit_transform(data['os'])
data.drop(columns=['O', 'S', 'M'], inplace=True)
X = data.iloc[:, :-3]
Y = data.iloc[:, -3:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
forest = RandomForestClassifier(random_state=1, max_depth=10, min_samples_split=5, min_samples_leaf=2)
log_reg = LogisticRegression(random_state=1, max_iter=500)
svc = SVC(random_state=1, kernel='linear')
knn = KNeighborsClassifier(n_neighbors=5)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_log_reg = MultiOutputClassifier(log_reg, n_jobs=-1)
multi_target_svc = MultiOutputClassifier(svc, n_jobs=-1)
multi_target_knn = MultiOutputClassifier(knn, n_jobs=-1)
multi_target_forest.fit(X_train, Y_train)
multi_target_log_reg.fit(X_train, Y_train)
multi_target_svc.fit(X_train, Y_train)
multi_target_knn.fit(X_train, Y_train)
Y_train_pred_forest = multi_target_forest.predict(X_train)
Y_test_pred_forest = multi_target_forest.predict(X_test)
Y_train_pred_log_reg = multi_target_log_reg.predict(X_train)
Y_test_pred_log_reg = multi_target_log_reg.predict(X_test)
Y_train_pred_svc = multi_target_svc.predict(X_train)
Y_test_pred_svc = multi_target_svc.predict(X_test)
Y_train_pred_knn = multi_target_knn.predict(X_train)
Y_test_pred_knn = multi_target_knn.predict(X_test)
train_accuracy_scores = {
    "Random Forest": [],
    "Logistic Regression": [],
    "SVC": [],
    "KNN": []
}
test_accuracy_scores = {
    "Random Forest": [],
    "Logistic Regression": [],
    "SVC": [],
    "KNN": []
}
for i, column in enumerate(Y.columns):
    for clf_name, (Y_test_pred, Y_train_pred) in {
        "Random Forest": (Y_test_pred_forest, Y_train_pred_forest),
        "Logistic Regression": (Y_test_pred_log_reg, Y_train_pred_log_reg),
        "SVC": (Y_test_pred_svc, Y_train_pred_svc),
        "KNN": (Y_test_pred_knn, Y_train_pred_knn)
    }.items():
        train_acc = accuracy_score(Y_train.iloc[:, i], Y_train_pred[:, i])
        test_acc = accuracy_score(Y_test.iloc[:, i], Y_test_pred[:, i])
        train_accuracy_scores[clf_name].append(train_acc)
        test_accuracy_scores[clf_name].append(test_acc)
mean_train_accuracy = {clf_name: np.mean(train_accuracy_scores[clf_name]) for clf_name in train_accuracy_scores}
mean_test_accuracy = {clf_name: np.mean(test_accuracy_scores[clf_name]) for clf_name in test_accuracy_scores}
best_classifier_name = max(mean_test_accuracy, key=mean_test_accuracy.get)
print(f"The best classifier based on mean test accuracy: {best_classifier_name}")
if best_classifier_name == "Random Forest":
    Y_train_pred_best = Y_train_pred_forest
    Y_test_pred_best = Y_test_pred_forest
    multi_target_best = multi_target_forest
elif best_classifier_name == "Logistic Regression":
    Y_train_pred_best = Y_train_pred_log_reg
    Y_test_pred_best = Y_test_pred_log_reg
    multi_target_best = multi_target_log_reg
elif best_classifier_name == "SVC":
    Y_train_pred_best = Y_train_pred_svc
    Y_test_pred_best = Y_test_pred_svc
    multi_target_best = multi_target_svc
elif best_classifier_name == "KNN":
    Y_train_pred_best = Y_train_pred_knn
    Y_test_pred_best = Y_test_pred_knn
    multi_target_best = multi_target_knn
for i in range(Y_test.shape[1]):
    cm = confusion_matrix(Y_test.iloc[:, i], Y_test_pred_best[:, i])
    print(f"Confusion Matrix for output '{Y.columns[i]}':")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test.iloc[:, i]))
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix for {Y.columns[i]} - {best_classifier_name} (Test Data)")
    plt.show()
print("\nAccuracy scores for all classifiers:")
for clf_name in train_accuracy_scores:
    print(f"\n{clf_name}:")
    for i, column in enumerate(Y.columns):
        print(f"Train Accuracy for {column}: {train_accuracy_scores[clf_name][i]:.2f}")
        print(f"Test Accuracy for {column}: {test_accuracy_scores[clf_name][i]:.2f}")
    overall_train_accuracy = np.mean(train_accuracy_scores[clf_name])
    overall_test_accuracy = np.mean(test_accuracy_scores[clf_name])
    print(f"\nOverall Train Accuracy: {overall_train_accuracy:.2f}")
    print(f"Overall Test Accuracy: {overall_test_accuracy:.2f}")  
    x = np.arange(len(Y.columns))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, train_accuracy_scores[clf_name], width, label='Train Accuracy', color='lightblue', edgecolor='black')
    plt.bar(x + width / 2, test_accuracy_scores[clf_name], width, label='Test Accuracy', color='lightgreen', edgecolor='black')
    plt.axhline(y=overall_train_accuracy, color='blue', linestyle='--', label=f"Mean Train Accuracy: {overall_train_accuracy:.2f}")
    plt.axhline(y=overall_test_accuracy, color='green', linestyle='--', label=f"Mean Test Accuracy: {overall_test_accuracy:.2f}")
    plt.title(f'Train and Test Accuracy for {clf_name}', fontsize=14)
    plt.xlabel('Targets', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x, Y.columns, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
clf_names = list(mean_train_accuracy.keys())
train_means = list(mean_train_accuracy.values())
test_means = list(mean_test_accuracy.values())
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(clf_names))
bar1 = ax.bar(index, train_means, bar_width, label='Train Accuracy', color='lightblue', edgecolor='black')
bar2 = ax.bar(index + bar_width, test_means, bar_width, label='Test Accuracy', color='lightgreen', edgecolor='black')
for i, (train_acc, test_acc) in enumerate(zip(train_means, test_means)):
    ax.text(index[i] - bar_width / 2, train_acc + 0.01, f'{train_acc:.2f}', ha='center', va='bottom', fontsize=10, color='blue')
    ax.text(index[i] + bar_width / 2, test_acc + 0.01, f'{test_acc:.2f}', ha='center', va='bottom', fontsize=10, color='green')
ax.set_xlabel('Classifiers', fontsize=12)
ax.set_ylabel('Mean Accuracy', fontsize=12)
ax.set_title('Comparison of Train and Test Accuracy for Each Classifier', fontsize=14)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(clf_names, rotation=45, ha='right', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()
