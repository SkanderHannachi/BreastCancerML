"""Algorithme deployant une regréssion logistique pour prévoir si une tumeur du sein est bénigne ou maligne.

Entête des data
    Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  
  Pour TAILLE_TEST et POURCENTAGE_TRAIN ajustées respectivement à 100 et 0.15, le taux de succés du modèle est estimé à 99%. 
  Données disponibles sur :   https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
  """
import pandas 
from pandas.tools.plotting import scatter_matrix #pour tracer les scatters
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import cross_validation 
import matplotlib.pyplot as plt 
import numpy 
#--------------------------------------------------------------Constantes
TAILLE_TEST         =    100 
POURCENTAGE_TRAIN   =    0.15
#--------------------------------------------------------------PP
binai = lambda x : 4 if x <= 0.5 else 2
#--------------------------------------------------------------

def convert_bin(X) : 
    """Fonction qui sert pour afficher des données à tester avec le modèle construit."""
    for i in range(len(X)) : 
        for j in range(2) :
            X[i][j] = binai(X[i][j])
    return X 
def main() : 
    Brut = pandas.read_table('cancer.txt',sep=",",header=0)
    print(Brut)
    data = Brut.as_matrix()
    X = data[:,1:10]
    Y = data[:,10]
    #Y = numpy.asarray([0 if el == 2 else 1 for el in Y])
    print("Données signifiantes : ")
    print(X)        
    X_app,X_test,y_app,y_test = cross_validation.train_test_split(X,Y,test_size =TAILLE_TEST,train_size=POURCENTAGE_TRAIN,random_state=0)
    print(X_app.shape,X_test.shape,y_app.shape,y_test.shape)
    Ir = LogisticRegression()
    print(X_app)
    #X_app = X_app.astype(float)
    try :
        model = Ir.fit(X_app,y_app)
        y_predict = model.predict(X_test)
        print("Taux de succés du modèle = "+str(model.score(X_test,y_test,sample_weight=None) * 100)+" %")
        print(" les coefficients du modèle sont : " + str(model.coef_))
        cm = metrics.confusion_matrix(y_test,y_predict)
        print("*********")
        #Matrice de confusion = 
        print("Matrice de confusion = ")
        print(cm)
        #taux d'erreur 
        print("Taux d'erreur = "+str(1.0 - metrics.accuracy_score(y_test,y_predict)))
    except ValueError as e:
        print(e)
    probas = Ir.predict_proba(X_test[10:89])
    print("########################################################")
    print("########################################################")
    print("##########################TEST##########################")
    print("Valeures exactes :")
    print(y_test[10:89])
    Val = convert_bin(probas)
    print("Probabilité calculée : ")
    print(Val[:,0])
    score = probas[:,1]
    #print(score)
if __name__=='__main__' :
    main()
       
