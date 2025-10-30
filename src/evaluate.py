# Evaluation module for machine learning models

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def Evaluate_model(model, X_test, y_test, name, out_path):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{out_path}/confusion_matrix_{name}.png")
    plt.close()

    print(f"\n{name} Model Report:")
    print(classification_report(y_test, y_pred))

    return classification_report(y_test, y_pred, output_dict=True)
