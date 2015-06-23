import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn import naive_bayes
from sklearn import svm

def sigmoid(z):
	return 1 / (1 + np.exp(-z));

def train(x,y,learningRate,maxIter):

	lenWeights = len(x[1,:]);
	weights = np.random.rand(lenWeights);
	bias = np.random.random();
	t = 1;
	converged = False;

	# Perceptron Algorithm

	while not converged and t < maxIter:
		targets = [];
		for i in range(len(x)):

				# Calculate logistic function (sigmoid function)
				# The decision function is given by the line w'x + b = 0;

				z = np.dot(x[i,:],weights) + bias;
				logistic = sigmoid(z);

				# Logistic regression probability estimate

				if (logistic > 0.5):
					target = 1;
				else:
					target = 0;

				# Calculate error and update weights
				# Shifts the decision boundary

				error = y[i] - logistic;
				weights = weights + (x[i,:] * (learningRate * error));
				bias = bias + (learningRate * error);

				targets.append(target);

				t = t + 1;

		if ( list(y) == list(targets) ) == True:
			# As soon as a solution is found break out of the loop
			converged = True;


	return weights,bias

def trainb(x,y,learningRate,maxIter):

	lenWeights = len(x[1,:]);
	weights = np.random.rand(lenWeights);
	bias = np.random.random();
	t = 1;
	converged = False;

	# Perceptron Algorithm

	while not converged and t < maxIter:
		targets = [];
		for i in range(len(x)):

				# Calculate logistic function (sigmoid function)
				# The decision function is given by the line w'x + b = 0;

				z = np.dot(x[i,:],weights) + bias;
				logistic = sigmoid(z);

				# Logistic regression probability estimate

				if (logistic > 0.5):
					target = 1;
				else:
					target = 0;

				# Calculate error and update weights
				# Shifts the decision boundary

				error = y[i] - logistic;
				weights = weights + (x[i,:] * (learningRate * error));
				bias = bias + (learningRate * error);

				targets.append(target);

				t = t + 1;

		if ( list(y) == list(targets) ) == True:
			# As soon as a solution is found break out of the loop
			converged = True;


	return weights,bias

def test(weights, bias, x):

	predictions = [];
	margins = [];
	probabilties = [];

	for i in range(len(x)):
		
		# Calculate w'x + b and sigmoid of output
		z = np.dot(x[i,:],weights) + bias;
		logistic = sigmoid(z);
		
		# Get decision from hardlim function
		if (logistic > 0.5):
			target = 1;
		else:
			target = 0;

		predictions.append(target);
		probabilties.append(logistic)

	return predictions,probabilties

if __name__ == "__main__":
	iris = datasets.load_iris()

	x = iris.data[:-50]
	y = iris.target[:-50]

	#learningRate = 0.02

	perceptron1 = train(x[:-5],y[:-5],0.001,2000)
	perceptron2 = train(x[:-5],y[:-5],0.01,2000)
	perceptron3 = train(x[:-5],y[:-5],0.1,2000)
	perceptron4 = train(x[:-5],y[:-5],1,2000)
	#print x[-5:],y[-5:]
	#print test(perceptron[0],perceptron[1],x[-5:])
	#print np.array( [0,(-1*perceptron[1])/perceptron[0][0]] ) , np.array( [ (-1*perceptron[1])/perceptron[0][1], 0] )

	'''skperceptron = linear_model.Perceptron()
	skperceptron.fit(x[:-5],y[:-5])
	print skperceptron.predict(x[-5:])
	print skperceptron.get_params()'''

	decisionPlot = plt.subplot(121);
	decisionPlot.plot(range(len(x)),test(perceptron1[0],perceptron1[1],x)[1],'r-')
	decisionPlot.plot(range(len(x)),test(perceptron2[0],perceptron2[1],x)[1],'g-')
	decisionPlot.plot(range(len(x)),test(perceptron3[0],perceptron3[1],x)[1],'y-')
	decisionPlot.plot(range(len(x)),test(perceptron4[0],perceptron4[1],x)[1],'m-')
	#decisionPlot.plot(range(len(x)),y,'b-')

	perceptron1b = trainb(x[:-5],y[:-5],0.001,2000)
	perceptron2b = trainb(x[:-5],y[:-5],0.01,2000)
	perceptron3b = trainb(x[:-5],y[:-5],0.1,2000)
	perceptron4b = trainb(x[:-5],y[:-5],1,2000)
	#print x[-5:],y[-5:]
	#print test(perceptron[0],perceptron[1],x[-5:])
	#print np.array( [0,(-1*perceptron[1])/perceptron[0][0]] ) , np.array( [ (-1*perceptron[1])/perceptron[0][1], 0] )

	'''skperceptron = linear_model.Perceptron()
	skperceptron.fit(x[:-5],y[:-5])
	print skperceptron.predict(x[-5:])
	print skperceptron.get_params()'''

	decisionPlotb = plt.subplot(122);
	decisionPlotb.plot(range(len(x)),test(perceptron1b[0],perceptron1b[1],x)[1],'r-')
	decisionPlotb.plot(range(len(x)),test(perceptron2b[0],perceptron2b[1],x)[1],'g-')
	decisionPlotb.plot(range(len(x)),test(perceptron3b[0],perceptron3b[1],x)[1],'y-')
	decisionPlotb.plot(range(len(x)),test(perceptron4b[0],perceptron4b[1],x)[1],'m-')

	plt.show()
