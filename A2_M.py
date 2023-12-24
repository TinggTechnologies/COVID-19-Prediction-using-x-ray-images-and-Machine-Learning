# Import required libraries
import os
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

Dataset_dir = "./"

def load_process_imgs(thresh, IMG_SIZE=100):
    Categories = ["Negative", "Positive"]
    features = []
    target = []
    for category in Categories:
        path = os.path.join(Dataset_dir, category)
        target_val = Categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            rez_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            im_bin = (rez_img > thresh) * 255
            features.append(im_bin.flatten())
            target.append(target_val)
    return features, target

Features, Target = load_process_imgs(200, 150)

features = np.array(Features)
target = np.array(Target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=0)

def SVM_Model(kernel, X_train, X_test, y_train, y_test):
    clf_SVM = SVC(C=10, kernel=kernel)
    clf_SVM.fit(X_train, y_train)
    y_pred = clf_SVM.predict(X_test)
    
    accuracy = accuracy_score(y_pred, y_test) * 100
    print(f"The SVM model with {kernel} kernel is {accuracy:.2f}% accurate")
    
    print("SVM Model Summary:")
    print(clf_SVM)
    
    print("SVM Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred)
    print("SVM Model ROC AUC:", roc_auc)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("SVM Model ROC AUC:", roc_auc)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



def NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons):
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=22500))
    model.add(Dense(numOfHidden_neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' activation for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    y_train_binary = y_train.astype(int)
    y_test_binary = y_test.astype(int)
    
    model.fit(X_train, y_train_binary, epochs=20)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.round(y_pred_probs)  # Round to get binary predictions
    
    accuracy = accuracy_score(y_test_binary, y_pred) * 100
    print(f"The NN keras model is {accuracy:.2f}% accurate")
    
    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()




    
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=22500))
    model.add(Dense(numOfHidden_neurons, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The CNN model is {accuracy * 100}% accurate")

# test cases
SVM_Model('rbf', X_train, X_test, y_train, y_test)
SVM_Model('poly', X_train, X_test, y_train, y_test)
SVM_Model('linear', X_train, X_test, y_train, y_test)
SVM_Model('sigmoid', X_train, X_test, y_train, y_test)

#NN_model_MLPClassifier(X_train, X_test, y_train, y_test, 12)
#NN_model_MLPClassifier(X_train, X_test, y_train, y_test, 44)
#NN_model_MLPClassifier(X_train, X_test, y_train, y_test, 86)


#NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons=50)
#NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons=90)
NN_Keras_Model(X_train, X_test, y_train, y_test, numOfHidden_neurons=128)