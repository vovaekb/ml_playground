'''
Source: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

digits = load_digits()

# Print information about images data
print "Image data shape:", digits.data.shape
# Print information about labels data
print "Label data shape:",digits.target.shape

# Visualize data
'''
plt.figure(figsize=(20,4))
for index, (image,label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

plt.show()
'''
# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, 
        digits.target, test_size=0.25, random_state=0)

print "Train image set shape:",x_train.shape
print "Train label set shape:",y_train.shape

# Setup model

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

# Predict for new observation
predict = logisticRegr.predict(x_test[0].reshape(1,-1))
print predict

# Predict for multiple observations
#predictions = logisticRegr.predict(x_test[0:5])
#print predicts

# Predict for entire test set
predictions = logisticRegr.predict(x_test)
#print predicts_all

# Get accuracy of model
score = logisticRegr.score(x_test, y_test)
print 'Model accuracy:',score

cm = metrics.confusion_matrix(y_test, predictions)
#print cm

# Visualize confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidth=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
