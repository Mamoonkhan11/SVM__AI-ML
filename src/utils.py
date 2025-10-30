# Utility functions for SVM visualization

import numpy as np
import matplotlib.pyplot as plt

def Plot_decision_boundary(model, X, y, kernel_name, out_path):
    X = X[:, :2]  # use only first 2 features for visualization
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f"SVM Decision Boundary ({kernel_name})")
    plt.savefig(f"{out_path}/svm_{kernel_name}_boundary.png")
    plt.close()
