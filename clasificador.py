import operator
import numpy as np
import pandas as pd
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#-------------LIBROS----------------
libros = ["Artificial Intelligence A Modern Approach.txt",
    "1._100_ways_to_motivate_yourself.txt",
    "A Student's Introduction to English Grammar ( PDFDrive ).txt",
    "4_harry_potter_and_the_goblet_of_fire.txt",
    "36 Signature Training Workout Programs ( PDFDrive ).txt"]
classlibros=[0,1,2,3,4]

testlib= ["Abstract Computing Machines A Lambda Calculus Perspective ( PDFDrive ).txt"]
classtest=[0]
#0=AI 1=BUSINESS 2=EDUCATION 3=FANTASY 4=HEALTH

tl=len(classlibros)

'''l1=open("Artificial Intelligence A Modern Approach.txt")
lib1=(l1.read())
l1.close()
print(len(lib1))'''

#--------CREANDO TRAINSET----------
trainset=np.array(())
for i in range(tl):
    libro=libros[i]
    lib=open(libro)
    aux=np.array((lib.read()))
    trainset=np.append(trainset,aux)
    lib.close()

print(np.shape(trainset))
print(np.shape(classlibros))

vect=CountVectorizer() #inicio de conteo
vect.fit(trainset) #entrenamiento
vect.get_feature_names() #se obtiene el vocabulario
matrixtrain=vect.transform(trainset) #crea la matriz
datatrain=pd.DataFrame(matrixtrain.toarray(),columns=vect.get_feature_names()) #transforma a dataframe
print(datatrain)

#----------CREANDO TESTSET-----------
testset=np.array(())
for i in range(1):
    libro=libros[i]
    lib=open(libro)
    aux=np.array((lib.read()))
    testset=np.append(testset,aux)
    lib.close()

matrixtest=vect.transform(testset)
datatest=pd.DataFrame(matrixtest.toarray(), columns=vect.get_feature_names())
print(datatest)

#------EVAULANDO--------
nb=MultinomialNB()
nb.fit(matrixtrain,classlibros)

classpred=nb.predict(matrixtest)

print("El porcentaje de clasificación es:")
porcentaje=metrics.accuracy_score(classtest,classpred)
print(porcentaje)
print("La matriz de confusion es:")
confmatrix=metrics.confusion_matrix(classtest,classpred)
print(confmatrix)