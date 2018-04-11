from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.externals.six import StringIO
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydotplus as pydot

mnist = fetch_mldata('MNIST original')
print mnist.data.shape
print mnist.target.shape

# Decision tree classifier
train_x, test_x, train_y, test_y = train_test_split(mnist.data, mnist.target, test_size=1/8.0, random_state=0) # 1/7.0
decision = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
decision.fit(train_x, train_y)
print 'Accuracy:',decision.score(test_x, test_y)

# Visualization
'''
dot_data = StringIO()
tree.export_graphviz(decision, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
'''

