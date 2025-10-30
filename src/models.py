# SVM model training module

from sklearn.svm import SVC

def Train_SVM(X_train, y_train, kernel='linear', C=1.0, gamma='scale'):
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    model.fit(X_train, y_train)
    return model
