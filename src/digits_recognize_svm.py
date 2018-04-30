#
# Tutorial: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html#svm-opencv
#


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from skimage import feature

SZ=20
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    hog_fd = feature.hog(np.array(img), orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualise=False)
    return hog_fd

img = cv2.imread('../img/digits.png', 0)

plt.title('Digits')
plt.imshow(img)
#plt.show()
print(img.shape)

cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
print(len(cells))
print(np.array(cells).shape)

train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells ]
print(np.array(train_cells).shape)
print(np.array(test_cells).shape)

# Train SVM
deskewed = [map(deskew, row) for row in train_cells]
#print(np.array(deskewed).shape)
hog_features = [] # [map(hog, row) for row in train_cells]

for i,row in enumerate(train_cells):
    for j,cell in enumerate(row):
        hog_fd = hog(cell)
        hog_features.append(hog_fd)

#print(np.array(hog_features).shape)
train_data = np.array(hog_features, 'float64')
responses = np.array(np.repeat(np.arange(10),250)) 
#print(responses.shape)
clf = svm.LinearSVC(C=2.75) # 2.67) 
clf.fit(train_data, responses)

deskewed_test = [map(deskew, row) for row in test_cells]
hog_features = []

for row in test_cells:
    for cell in row:
        hog_fd = hog(cell)
        hog_features.append(hog_fd)

test_data = np.array(hog_features, 'float64')
result = clf.predict(test_data)
print(result[:20])

# Calculate accuracy
score = clf.score(test_data, responses)
print('Score: {}'.format(score))
