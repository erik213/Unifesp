""" 
Implementation of a Rosenblatt Perceptron in Python
Adapted from: https://gist.github.com/jaypmorgan/eb0dc831b4cb46a95eaa2a7bcb9382c6
"""
import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, threshold=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def fit(self, X, y, names):
        """
        Our fit function trains on the dataset X and tries to predict vector y,
        Using the learning rate, it will modify it's weight vector to increase
        it's accuracy in predictions.
        It will iterate over the X dataset as defined by the epochs.
        Args:
            X: The input data (numpy array of shape [n_samples * m_features])
            y: Class labels vector (numpy array of shape [n_samples])
        """
        weights = np.zeros(X.shape[1])
        self.betterAccuracy = 0
        self.best_weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            # list of predicted classes for our accuracy calculation
            predicted = []
            print('\nEpoch [{}]'.format(epoch))
            for i_index, sample in enumerate(X):
                y_hat = self.predict(sample, weights)
                predicted.append(y_hat)  # add our new prediction to the array
                for j_index, feature in enumerate(weights):
                    # update our weight values
                    self.updatethreshold(y_hat, y[i_index])
                    delta = self.lr * (y[i_index] - y_hat)
                    delta = delta * sample[j_index-1]
                    weights[j_index-1] = weights[j_index-1] + delta
                acc = self._calculate_acc(X, y, weights)
                if acc >= self.betterAccuracy:
                    self.betterAccuracy = acc
                    self._change_array(weights, self.best_weights)
                print('Name: [{}] Predict: [{}] Class: [{}] Accuracy: [{:.2f}] Weights: {}'.format(names[i_index], y_hat, y[i_index], acc,weights))
        print('\nBest Accuracy: {}'.format(self.betterAccuracy))
        print('Best Weights: {}\n'.format(self.best_weights))
        
    def updatethreshold(self, predicted, actual):
        self.threshold = self.threshold + self.lr * 1 * (actual - predicted)

    def _change_array(self, a, b):
      for index, i in enumerate(a):
        b[index] = i

    def _calculate_acc(self, X, y, w):
        """
        Calculate the accuracy of predictions applying the current Weights vector to all dataset elements
        """
        predicted = []
        for i_index, sample in enumerate(X):
              y_hat = self.predict(sample, w)
              predicted.append(y_hat)
        return self._calculate_accuracy(y, predicted)

    def _calculate_accuracy(self, actual, predicted):
        """
        Calculate the accuracy of predictions for this epoch.
        Args:
            actual: vector of actual class values (the y vector) [n_samples]
            predicted: vector of predicted class values [n_samples]
        """
        return sum(np.array(predicted) == np.array(actual)) / float(len(actual))

    def predict(self, x, w):
        """
        Args:
            x: vector of the data sample - shape [m_features]
            w: vector of the weights - shape [m_features]
        """
        res = self._sum(x, w) + self.threshold
        return 1 if res > 0.0 else -1

    def _sum(self, x, w):
        """
        Multiply our sample and weight vector elements then the sum of the
        result.
        Args:
            x: vector of the data sample - shape [m_features]
            w: vector of the weights - shape [m_features]
        Returns:
            Int of the sum of vector products
        """
        return np.sum(np.dot(x, np.transpose(w)))

    def _test_new(self, new_data):
        """
        Test our new data with our trained weights.
        Args:
            new_data: vector of the data sample - shape [m_features]
        """
        return self.predict(new_data, self.best_weights)

if __name__ == '__main__':
    p = Perceptron(0.5, -0.5, 10)
    names = ['João','Pedro','Maria','José','Ana','Leila']
    x = np.array([[1,1,0,1],[0,0,1,0],[1,1,0,0],[1,0,1,1],[1,0,0,1],[0,0,1,1]])
    y = np.array([1,-1,-1,1,-1,1])
    p.fit(x, y, names)
    print('(Luis, 0, 0, 0, 1) = {}'.format(p._test_new(np.array([0,0,0,1]))))
    print('(Laura, 1, 1, 1, 1) = {}'.format(p._test_new(np.array([1,1,1,1]))))