import os
import pickle
from sklearn.externals import joblib

# Load classifier
clf = joblib.load(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'classifier/classifier.pkl'
    ))

score = pickle.loads(open(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'classifier/score.pkl'),
        'rb').read()
    )

#Put your own data in
data = []

hight = int(input("What is your hight in cm: "))
weight = int(input("What is your weight in kg: "))
size = int(input("What is your shoe size in european: "))

data.append(hight)
data.append(weight)
data.append(size)

predictions = clf.predict([data])

#Print score
print()
print("I predict with",((round(score,4))*100),"%","accuracy that you are a:",''.join(predictions)+".")