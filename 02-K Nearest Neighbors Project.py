


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


df=pd.read_csv('KNN_Project_Data')



df.head()

sns.pairplot(data=df,hue='TARGET CLASS',palette='coolwarm')


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

features=df.drop('TARGET CLASS',axis=1)
scaler.fit(features)

scaler_transform= scaler.transform(df.drop('TARGET CLASS',axis=1))

data=pd.DataFrame(scaler_transform,columns=df.columns[:-1])
data.head()


from sklearn.model_selection import train_test_split
X=data
X_train, X_test, y_train, y_test = train_test_split(data,df['TARGET CLASS'],
                                                    test_size=0.30)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)




from sklearn.metrics import classification_report,confusion_matrix




print(confusion_matrix(y_test,pred))


error_rate=[]

for i in range (1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
    
    

plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('K neighbour vs error')



# NOW WITH K=10
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=10')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


