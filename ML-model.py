import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle

# Import data
path = os.listdir(
    r'C:\Users\Abhishek Patwardhan\Desktop\TE-Project\BTWebsite\DataSet\Training')
classes = {'glioma_tumor': 0, 'meningioma_tumor': 1,
           'no_tumor': 2, 'pituitary_tumor': 3}

X = []
Y = []
for cls in classes:
    # pth = r'C:\Users\Abhishek Patwardhan\Desktop\TE-Project\DataSet\Training'+cls
    if cls == "glioma_tumor":
        pth = r"C:\Users\Abhishek Patwardhan\Desktop\TE-Project\BTWebsite\DataSet\Training\glioma_tumor"
    elif cls == "meningioma_tumor":
        pth = r"C:\Users\Abhishek Patwardhan\Desktop\TE-Project\BTWebsite\DataSet\Training\meningioma_tumor"
    elif cls == "no_tumor":
        pth = r"C:\Users\Abhishek Patwardhan\Desktop\TE-Project\BTWebsite\DataSet\Training\no_tumor"
    elif cls == "pituitary_tumor":
        pth = r"C:\Users\Abhishek Patwardhan\Desktop\TE-Project\BTWebsite\DataSet\Training\pituitary_tumor"

    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(classes[cls])

print(np.unique(Y))

X = np.array(X)
Y = np.array(Y)

print(pd.Series(Y).value_counts())

X.shape

# Visualize Data - To check particular image
# plt.imshow(X[0], cmap='gray')
# plt.show()

# Prepare Data
X_updated = X.reshape(len(X), -1)
print(X_updated.shape)

# Split Data
xtrain, xtest, ytrain, ytest = train_test_split(
    X_updated, Y, random_state=10, test_size=.20)

print(xtrain.shape, xtest.shape)


# Feature Scaling
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# Feature Selection : PCA
print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest


# Train Model
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)


sv = SVC()
sv.fit(pca_train, ytrain)


# Evaluation
# print("Training Score : ", lg.score(pca_train, ytrain))
# print("Testing Score : ", lg.score(pca_test, ytest))

# print("Training Score : ", sv.score(pca_train, ytrain))
# print("Testing Score : ", sv.score(pca_test, ytest))

# Prediction
pred = sv.predict(pca_test)
print(np.where(ytest != pred))

# plt.scatter(xtrain,ytrain,c=ytrain,cmap='rainbow')
# plt.title('Scatter Plot')
# plt.show()

# Test Model
dec = {0: 'glioma_tumor', 1: 'meningioma_tumor',
       2: 'no_tumor', 3: 'pituitary_tumor'}

plt.figure(figsize=(12, 8))
p = os.listdir(
    r'C:/Users/Abhishek Patwardhan/Desktop/TE-Project\BTWebsite/DataSet/Testing/')
c = 1
for i in os.listdir(r'C:/Users/Abhishek Patwardhan/Desktop/TE-Project\BTWebsite/DataSet/Testing/Unseen/')[:16]:
    plt.subplot(4, 4, c)

    img = cv2.imread(
        r'C:/Users/Abhishek Patwardhan/Desktop/TE-Project\BTWebsite/DataSet/Testing/Unseen/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = lg.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c += 1
plt.show()



filename = 'BT-model'
pickle.dump(lg,open(filename,'wb'))