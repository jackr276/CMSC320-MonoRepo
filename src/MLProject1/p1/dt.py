"""
In dt.py, you will implement a basic decision tree classifier for
binary classification.  Your implementation should be based on the
minimum classification error heuristic (even though this isn't ideal,
it's easier to code than the information-based metrics).
"""

import numpy as np
import statistics


class DT:
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts, left, right, feature):
        """
        Initialize our internal state.
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "is_leaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)
        
        self.is_leaf = True
        self.label = 1

        self.left = left
        self.right = right
        self.feature = feature

    def online(self):
        """
        Our decision trees are batch
        """
        return False


    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.display_tree(0)


    def display_tree(self, depth):
        # recursively display a tree
        if self.is_leaf:
            return (" " * (depth*2)) + "Leaf " + repr(int(self.label)) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.display_tree(depth+1) + \
                      self.right.display_tree(depth+1)

    """
    predict(node, x):
        if node is leaf:
            return node.label

        threshhold = 0.5

        if(x[feature] < threshhold):
            go left
        else :
            go right

    """
    def predict_singular_value(self, x):
        # Base case
        if(self.is_leaf == True):
            return self.label

        threshold = 0.5

        # If X is below the threshold go left
        if(x[self.feature] < threshold) :
            return self.left.predict_singular_value(x)

        # Else go right
        else :
            return self.right.predict_singular_value(x)


    def predict(self, X):
        """
        Traverse the tree to make predictions. You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch. Notice that a `for` loop is used per data point since different
        data in the batched `X` can end up in different paths down the tree. 
        """
        N = X.shape[0]
        # Empty prediction array
        preds = np.empty(N, dtype=float)

        # For each pr
        for n in range(N):
            # Let the helper predict a singular value
            result = self.predict_singular_value(X[n]) 

            #Store in preds
            preds[n] = result


    def train_dt(self, X, y, remaining_depth, used_features):
        """
        recursively build the decision tree
        """

        # Get the size of the data set (samples x features)
        N, d = X.shape

        # Check to see if we're either out of depth or no longer
        # have any decisions to make
        # Base case - we can't go any further
        if remaining_depth <= 0 or len(np.unique(y)) <= 1:
            # This is a leaf
            self.is_leaf = True
            # Guess is going to be y because y is the same
            self.label = y[0]

        else:
            #Determine who has the highest score
            feature_with_max_score = 0
            best_NO = []
            best_YES = []

            for feature in range(d):
                # We've already used this feature
                if feature in used_features:
                    pass

                # Lower is superior
                best_score = np.Infinity

                #NO <- subset of data on which f = no
                #YES <- subset of data on which f = yes
                #score[f] <- # of majority vote answers in NO + # of majority vote answers in YES


            #endfor
            #Need to manipulate y to only be for best_NO/best_YES
            self.left.train_dt(best_NO, y, remaining_depth - 1, used_features + [feature_with_max_score])
            self.right.train_dt(best_YES, y, remaining_depth -1, used_features + [feature_with_max_score])

            # What was the feature we last used(column index)



            # Examine error on all features (linear search)
            # Find best feature
            # Split on that into left and right subtrees

            ### TODO: YOUR CODE HERE (see Algorithm 1 of your textbook)
            pass


    def train(self, X, y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['max_depth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponting classes,
            say Y(X(:,5)>=0.5) and if you want the correspnding rows of X,
            say X(X(:,5)>=0.5,:)
            
          - i suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - Hint: Use `statistics.mode` to return the most common class. 
        """
        self.train_dt(X, y, self.opts["max_depth"], [])


    def get_representation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        return self
