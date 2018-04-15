from sklearn.datasets import fetch_mldata
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_mldata('MNIST original')

X = mnist.data
y = mnist.target

X4 = X[y==4,:]
X9 = X[y==9,:]

n = 3000

#finding the optimal parameter
Xtrain = np.concatenate((X4[0:n,:], X9[0:n,:]))
ytrain = np.concatenate((4*np.ones(n), 9*np.ones(n)))

#Cval = np.logspace(-5, 5, num=11) 
#Cval = np.logspace(-9, 3, num=20) 
Cval = np.logspace(-3, 3, num=13) 
gamma = np.logspace(-9, 3, num=13)
Pe = np.zeros(Cval.shape)

for i in range(Cval.size):
	clf = svm.SVC(C=Cval[i],kernel='poly',degree=2)
	clf.fit(Xtrain,ytrain)

	#parameter tuning
	Xholdout = np.concatenate((X4[n:4000,:], X9[n:4000,:]))
	yholdout = np.concatenate((4*np.ones(4000 - n), 9*np.ones(4000 - n)))
	Pe[i] = 1 - clf.score(Xholdout,yholdout)

C_val = Cval[np.argmin(Pe)]

#training
Xtrain = np.concatenate((X4[0:4000,:], X9[0:4000,:]))
ytrain = np.concatenate((4*np.ones(4000), 9*np.ones(4000)))

clf = svm.SVC(C=C_val,kernel='poly',degree=2)
clf.fit(Xtrain,ytrain)

indices = [np.abs(clf.decision_function(clf.support_vectors_)).argsort()[:16]]
SV_hard = clf.support_vectors_[indices]
SV_hard_indices = ytrain[clf.support_[indices]]

Xtest = np.concatenate((X4[4000:,:], X9[4000:,:]))
ytest = np.concatenate((4*np.ones((X4.shape[0] - 4000)), 9*np.ones(X9.shape[0] - 4000)))
test_error = 1 - clf.score(Xtest,ytest)

numSV = clf.support_vectors_.shape[0]

print "C = %0.9f" %C_val
print "Test Error = %f" %test_error
print "Number of support vectors = %d" %numSV

f, axarr = plt.subplots(4, 4)
h=0
k=0
for i in range(16):
    axarr[h, k].imshow(SV_hard[i].reshape((28,28)), cmap='gray')
    axarr[h, k].axis('off')
    axarr[h, k].set_title('{label}'.format(label=int(SV_hard_indices[i])))
    k += 1
    if k==4:
        k = 0
        h += 1
plt.show()