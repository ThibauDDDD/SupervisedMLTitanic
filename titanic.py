import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


np.set_printoptions(threshold=np.inf)  # permet d'afficher le print en entier

#je récupère les données du fichier csv
csv_file = pd.read_csv("Titanic-Dataset.csv")
np_data = csv_file.to_numpy()

# convertie la colone sexe en 0 ou 1
np_data[:, 4:5] = np.where(np_data[:, 4:5] == "male", 1, 0)

# cette partie permet de calculer le pourcentage de survie pour un homme et pour une femme.
# dead_male = 0
# survived_male = 0
# dead_female = 0
# survived_female = 0
# for d in range(len(np_data)):
#     if np_data[d][1] == 0 and np_data[d][4] == 1:
#         dead_male += 1
#     if np_data[d][1] == 1 and np_data[d][4] == 1:
#         survived_male += 1
#     if np_data[d][1] == 1 and np_data[d][4] == 0:
#         survived_female += 1
#     if np_data[d][1] == 0 and np_data[d][4] == 0:
#         dead_female += 1
# print("mort male =", dead_male, "survivant male", survived_male, "taux de survie pour un homme = ", 
#       1 - dead_male / (survived_male + dead_male), " % \nmort femme =", dead_female, "survivant femme", 
#       survived_female, "taux de survie pour une femme = ", 1 - dead_female / (dead_female + survived_female), "%\n\n")


#convertie la colonne Cabin en 0 si il n'y a pas de cabine et 1 si il y a une cabine
np_data[:, 10:11] = np_data[:, 10:11].astype(str)
np_data[:, 10:11] = np.where(np_data[:, 10:11] == "nan", 1, 0)

#convertie la colone Age en -1 s'il n'y a pas d'âge et l'âge sinon
np_data[:, 5:6] = np.where(np_data[:, 5:6].astype(str) == "nan", -1, np_data[:, 5:6])

#je supprime la colonne ticket car l'algo ne peut pas gérer les string
np_data = np.delete(np_data, 8, axis=1)
#je supprime la colonne embarked car l'algo ne peut pas gérer les string
np_data = np.delete(np_data, 10, axis=1)

#je divise les données en 2 parties, 80% pour l'entrainement et 20% pour le test
limit = int(0.8*len(np_data))
train_data = np_data[:limit]
test_data = np_data[limit:]

#je supprime la colone survived parce que c'est l'élément X et les noms car l'algo ne peut pas gérer les string
train_data_x = np.delete(np.delete(train_data, 1, axis=1), 2, axis=1)
test_data_x = np.delete(np.delete(test_data, 1, axis=1), 2, axis=1)

#je transforme le tableau 2d en tableau 1d et m'assure que les valeurs sont des int
train_data_y = train_data[:, 1:2].ravel().astype(int)
test_data_y = test_data[:, 1:2].ravel().astype(int)

#algo Machine Learning supervisé de régresison logistique, car c'est un problème de classification binnaire (réponse 1 ou 0)
logistic_reg = LogisticRegression(max_iter=500).fit(train_data_x, train_data_y)
result = logistic_reg.predict(test_data_x)
accuracy = logistic_reg.score(test_data_x, test_data_y)

print("matrice de survie Réel: 1 pour survie, 0 pour mort\n\n", test_data_y)
print("matrice de survie Hypothétique : 1 pour survie, 0 pour mort\n\n",result)
print("pourçentage de chance de réussite: ", int(accuracy * 100), "%")
