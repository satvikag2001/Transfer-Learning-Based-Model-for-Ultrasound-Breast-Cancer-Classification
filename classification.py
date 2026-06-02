#%%
import numpy as np
import pandas as pd

#%%
df = pd.read_csv("Breakhis400x.csv")
Y = df.iloc[:,0]
X = df.iloc[:,1:]
#%%
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#%%
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#%%
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression().fit(X_train, Y_train)
print(accuracy_score(Y_test, clf.predict(X_test)))

#%%

X_train,X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.2)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
#model.add(Dense(units = 1024, activation = "relu"))
#model.add(Dense(units = 1024, activation = "relu"))
model.add(Dense(units = 512, activation = "relu"))
model.add(Dense(units = 256, activation = "relu"))
model.add(Dense(units = 1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = "accuracy")
model.fit(X_train, Y_train, epochs = 150, validation_data=(X_val, Y_val))

Y_pred = []
for i in model.predict(X_test):
    if i>0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)
print(accuracy_score(Y_test, Y_pred))


#%%
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
print(accuracy_score(Y_test,Y_pred))

#%%
from sklearn.neighbors import KNeighborsClassifier
for i in range(3,30):
    k = KNeighborsClassifier(n_neighbors = i)
    k.fit(X_train, Y_train)
    Y_pred = k.predict(X_test)
    print(i, accuracy_score(Y_test, Y_pred))