import sklearn.ensemble as ske
from sklearn import cross_validation, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43], [153, 54, 35], [154, 56, 36], [154, 60, 37], [155, 70, 36],
     [158, 65, 36], [159, 63, 37], [160, 80, 39], [161, 66, 36], [163, 57, 39],
     [164, 67, 40], [165, 64, 40], [165, 75, 39], [165, 63, 40], [166, 71, 43],
     [167, 72, 42], [167, 74, 43], [168, 80, 43], [168, 77, 40], [170, 80, 44],
     [170, 67, 43], [170, 69, 41], [171, 68, 42], [173, 73, 43], [174, 75, 43],
     [174, 90, 41], [174, 89, 43], [174, 75, 42], [175, 76, 45], [175, 73, 44],
     [176, 78, 42], [177, 76, 43], [178, 82, 43], [178, 88, 44], [178, 98, 45],
     [179, 90, 45], [179, 93, 45], [179, 85, 44], [180, 87, 46], [180, 95, 45],
     [183, 70, 46], [185, 75, 47], [193, 120, 47]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female', 'female', 'female', 'female', 'male',
     'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male',
     'female', 'female', 'female', 'male', 'male', 'female', 'female', 'male',
     'male', 'male', 'male', 'male', 'female', 'male', 'female', 'male', 'male',
     'male', 'male', 'female', 'male', 'male', 'male', 'male', 'male', 'male',
      'male', 'male', 'male']

#Train on the dataset
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y ,test_size=0.2)

#Algorithm comparison
algorithms = {
            "DecissionTree": tree.DecisionTreeClassifier(),
            "KNeighbors": KNeighborsClassifier(),
            "RandomForest": ske.RandomForestClassifier(),
            "GradientBoosting": ske.GradientBoostingClassifier(),
            "AdaBoost": ske.AdaBoostClassifier(),
            "GNB": GaussianNB()
            }

results = {}

print("\nNow testing algotihms")
print()
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

#Print the winner with the highest success on the testing data
winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))
print()

#Put your own data in
data = []

hight = int(input("What is your hight in cm: "))
weight = int(input("What is your weight in kg: "))
size = int(input("What is your shoe size in european: "))

data.append(hight)
data.append(weight)
data.append(size)

clf = algorithms[winner]
predictions = clf.predict([data])

#Print score
print()
print("I predict with",((round(results[winner],4))*100),"%","accuracy that you are a:",''.join(predictions)+".")
