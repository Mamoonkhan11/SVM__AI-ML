# Main script to load data, train SVM models, evaluate them, and plot decision boundaries

from src.Load_data import Load_data
from src.models import Train_SVM
from src.evaluate import Evaluate_model
from src.utils import Plot_decision_boundary

def main():
    X_train, X_test, y_train, y_test = Load_data(file_path="Data/breast-cancer.csv")

    # Linear SVM
    linear_model = Train_SVM(X_train, y_train, kernel='linear', C=1)
    Evaluate_model(linear_model, X_test, y_test, "Linear", "outputs")
    Plot_decision_boundary(linear_model, X_train, y_train, "linear", "outputs")

    # RBF SVM
    rbf_model = Train_SVM(X_train, y_train, kernel='rbf', C=1, gamma=0.1)
    Evaluate_model(rbf_model, X_test, y_test, "RBF", "outputs")
    Plot_decision_boundary(rbf_model, X_train, y_train, "rbf", "outputs")

if __name__ == "__main__":
    main()
