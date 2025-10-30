# Utility functions for SVM visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

def Plot_decision_boundary(model, X, y, kernel_name, out_path):
    X_2d = X[:, :2]
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Retrain small 2-feature model
    temp_model = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma)
    temp_model.fit(X_2d, y)

    Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(["#583737", "#437E43"])
    cmap_bold = ListedColormap(["#9A2828", "#57A057"])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)
    plt.title(f"SVM Decision Boundary ({kernel_name} kernel)")
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.savefig(f"{out_path}/svm_{kernel_name}_boundary.png")
    plt.close()
