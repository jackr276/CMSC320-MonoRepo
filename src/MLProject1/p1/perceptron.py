import numpy as np

class Perceptron:
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs
        self.w = None
        self.b = 0

    def train(self, X, y):
        """
        TODO: Implement the Perceptron Update Rule.
        1. Init w and b to zeros (w is a vector and b is a scalar).
        2. Loop epochs.
        3. Loop examples:
           If prediction is wrong:
              w = w + y * x
              b = b + y
        """
        num_rows, num_cols = X.shape  # initialize number of examples and features

        self.w = np.zeros(num_cols) # initialize weights to 0, length is the number of features
        self.b = 0.0 # initialize bias to 0

        # loop over each epoch
        for epoch in range(self.num_epochs):
            # loop over each example (row)
            for i in range(num_rows):
                x_i = X[i]
                y_i = y[i]

                # compute the socre
                score = np.dot(self.w, x_i) + self.b

                # make score +1 or -1
                if score > 0: 
                   pred = 1.0
                else:
                    pred = -1.0

                if pred != y_i:
                    self.w = self.w + y_i * x_i
                    self.b = self.b + y_i


    def predict(self, X):
        """
        TODO: Compute w*x + b. Return +1 or -1.
        """
        scores = np.dot(X, self.w) + self.b
        return np.where(scores > 0, 1.0, -1.0)