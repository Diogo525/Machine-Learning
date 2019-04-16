import numpy as np;
import matplotlib.pyplot as plt;
import math;

'''
Calculate the cost function result for the theta

J=(h-y)'*(h-y)/(2*m);
'''
def costFunction(X, y, theta):

    # calculate hypothesis values through vector multiplication
    h = np.matmul(X, theta);

    # difference between theoretical result and the true result
    # subtraction of matrices
    dif = h-y;

    difTranspose = dif.transpose();

    # the operation results in a list of lists with one value, that is extracted
    # in the return statement
    J = np.matmul(difTranspose, dif) / (2*len(y));

    return J[0, 0];

'''
Calculates the best values for theta using gradient descent

Returns a tuple with the best theta values found, the history of costs
and the history of the theta values.
'''
def gradientDescent(X, y, theta, alpha, niters):

    # variable setup
    J_history = np.zeros(shape=(niters, 1));
    nTrainSamples = len(y);
    nFeatures = len(X[0]);
    theta_history = np.zeros(shape=(niters, 2));

    for i in range(niters):
        printPercentageDone(i, niters-1);

        h = np.matmul(X, theta);
        dif = h-y;
        difTranspose = dif.transpose();
        grad = np.matmul((1/nTrainSamples)*difTranspose, X);
        gradTranspose = grad.transpose();
        theta=theta-alpha*gradTranspose;

        # save theta values
        for j in range(nFeatures):
            theta_history[i,j] = theta[j];

        # compute and save cost values
        J_history[i] = costFunction(X, y, theta);

    return (theta, J_history, theta_history);

'''
For datasets with 1 feature, show the evolution of the hypothesis function
in relation to the training set
'''
def gradientDescentWithHypothesisGraph(X, y, theta, alpha, niters):

    # turn interactive mode on
    plt.ion();
    fig, ax = plt.subplots();
    l0, = ax.plot(X[:, 1], y[:, 0], marker='x', linestyle='None', color='red', label='Training samples');

    # These are subplot grid parameters encoded as a single integer.
    # For example, "111" means "1x1 grid, first subplot" and "234" means "2x3 grid, 4th subplot".
    # ax = fig.add_subplot(111);
    ax.set_xlabel("Feature x1 - Population in 10000s")
    ax.set_ylabel("Profit in $10000s")
    ax.set_title("Training set and hypothesis function");
    # the trailing comma is used to denote a tuple of undetermined size
    l1, = ax.plot(X[:, 1], np.matmul(X, theta), '-', label='Hypothesis');        # returns a tuple of line2D objects
    ax.legend();

    lines = [l0, l1];

    # draw and show it
    fig.canvas.draw();

    J_history = np.zeros(shape=(niters, 1));
    nTrainSamples = len(y);
    nFeatures = len(X[0]);
    theta_history = np.zeros(shape=(niters, 2));

    for i in range(niters):
        printPercentageDone(i, niters-1);

        h = np.matmul(X, theta);
        dif = h-y;
        difTranspose = dif.transpose();
        grad = np.matmul((1/nTrainSamples)*difTranspose, X);
        gradTranspose = grad.transpose();
        theta=theta-alpha*gradTranspose;

        # save theta values
        for j in range(nFeatures):
            theta_history[i,j] = theta[j];

        # compute and save cost values
        J_history[i] = costFunction(X, y, theta);

        lines[1].set_ydata(np.matmul(X, theta));
        ax.relim();										# recalculate limits
        ax.autoscale_view(scalex=True, scaley=True); 	# autoscale

        #plt.draw();
        fig.canvas.draw();
        #fig.canvas.flush_events();

    plt.clf();

    return (theta, J_history, theta_history);

'''
For datasets with 1 feature, show the evolution of the cost
'''
def gradientDescentWithCostGraph(X, y, theta, alpha, niters):
    # turn interactive mode on
    plt.ion();

    fig = plt.figure();
    # Cost function plot
    ax = fig.add_subplot(1, 1, 1);
    ax.set_xlabel("Number of iterations");
    ax.set_ylabel("Cost function J");
    ax.set_title("Cost function evolution");
    ax.set_xlim((0, niters));
    l1, = ax.plot([], [], '-', label='J(theta)');
    ax.legend();

    # draw and show it
    fig.canvas.draw();

    J_history = np.zeros(shape=(niters, 1));
    nTrainSamples = len(y);
    nFeatures = len(X[0]);
    theta_history = np.zeros(shape=(niters, 2));

    for i in range(niters):
        printPercentageDone(i, niters-1);

        h = np.matmul(X, theta);
        dif = h-y;
        difTranspose = dif.transpose();
        grad = np.matmul((1/nTrainSamples)*difTranspose, X);
        gradTranspose = grad.transpose();
        theta=theta-alpha*gradTranspose;

        # save theta values
        for j in range(nFeatures):
            theta_history[i,j] = theta[j];

        # compute and save cost values
        J_history[i] = costFunction(X, y, theta);

        l1.set_xdata(np.append(l1.get_xdata(), i));
        l1.set_ydata(J_history[:i+1]);
        ax.relim();                                     # recalculate limits
        ax.autoscale_view(scalex=True, scaley=True);    # autoscale

        fig.canvas.draw();
        #fig.canvas.flush_events();

    plt.clf();

    return (theta, J_history, theta_history);

'''
Update and display the percentage of work done
'''
def printPercentageDone(i, niters):
    percentage = '%.2f' % (i/niters*100);
    string = 'Computing theta: ' + percentage + '%';
    print(string, end='\r');
