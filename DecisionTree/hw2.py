import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if len(data) < 2:
        return 0
    labels, counts = np.unique(data[:, -1], return_counts=True)
    total = len(data)
    for i in range(labels.size):
        iVal = counts[i] / total
        gini += iVal ** 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (1 - gini)

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if len(data) < 2:
        return 0
    labels, counts = np.unique(data[:, -1], return_counts=True)
    total = len(data)
    for i in range(labels.size):
        iVal = counts[i] / total
        entropy += iVal * np.log2(iVal)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return -entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        self.labels = {} # true labels dictionary
        self.parent = None


    def add_child(self, node):
        self.children.append(node)
    
    def build_nodes_chi(self, data, impurity, chi_value):
        """
        Building a tree recursively for self node as the root
        with the provided data using the impurity measure
        Chi value = 1 means no pruning

        Input:
        - data: to consider with building the tree
        - impurity: measure to consider for calculations
        - chi_value: chi value to prune accordingly

        Output: void.
        """
        # labels count for current node
        label, count = np.unique(data[:, -1], return_counts=True)
        self.labels = dict(zip(label, count))

        # impurity check
        if impurity(data) == 0:
            return

        # find best feature and value and assign to self
        self.feature, self.value = find_best_attribute(data, impurity)

        # Chi square check
        if chi_value != 1:
            if chi_square(data, self.feature, self.value) <= chi_table.get(chi_value):
                return

        # data partitioning,
        # lt_data = less than threshold
        # gt_data = greater than threshold
        lt_data, gt_data = data_partition(data, self.feature, self.value)

        # initiate descendants and assign parents
        left_node = DecisionNode(None, None)
        right_node = DecisionNode(None, None)
        left_node.parent = self
        right_node.parent = self
        self.add_child(left_node)
        self.add_child(right_node)

        # build subtrees recursively
        left_node.build_nodes_chi(lt_data, impurity, chi_value)
        right_node.build_nodes_chi(gt_data, impurity, chi_value)

    def __repr__(self):
        if not self.children: # check if node is a leaf
            return f"leaf: [{self.labels}]"
        else:
            return f"[X{self.feature} <= {self.value}]"

def build_tree(data, impurity, chi_value = 1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure.
    Chi value default set to 1.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(None, None) # initiate root node as Decision node
    root.build_nodes_chi(data, impurity, chi_value) # build tree using the root recursively, prune according to chi value
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def find_best_attribute(data, impurity):
    """
    Find best attribute and threshold for the next split.
    
    Input: 
    - data: the training dataset.
    - impurity: the chosen impurity measure.
    
    Output: best feature and its threshold for the next split.
    """
    # initiate maximum gain
    max_gain = 0.0

    # iterate data features, dropping the true labels column
    for column in range(data.shape[1] - 1):
        threshold, gain = find_best_threshold(data, column, impurity)

        # assign current best attribute and its threshold yet
        if gain > max_gain:
            best_feature = column
            best_value = threshold
            max_gain = gain

    return best_feature, best_value

def find_best_threshold(data, column, impurity):
    """
    Find the best threshold for the provided data
    considering feature 'column' and the impurity measure

    Input:
    - data: the training dataset.
    - column: the feature column
    - impurity: the chosen impurity measure.

    Output:
    - best_threshold: best threshold for the provided arguments
    - max_gain: goodness of split on the calculated threshold split
    """

    # initiate thresholds list, maximum gain, and sorting all values
    thresholds = []
    max_gain = 0.0
    sorted_values = np.sort(data[:, column])

    # iterate all values to find thresholds
    for i in range(len(sorted_values) - 1):
        thresholds.append((sorted_values[i] + sorted_values[i + 1]) / 2)

    # find best threshold by maximum impurity reduce
    for threshold in thresholds:
        gain = goodness_split(data, column, threshold, impurity)
        if gain > max_gain:
            best_threshold = threshold
            max_gain = gain

    return best_threshold, max_gain

def goodness_split(data, column, threshold, impurity):
    """
    Calculates goodness for the provided data
    to the 'column' feature with its threshold
    using the impurity measure.

    Input:
    - data: the training dataset.
    - column: the feature column
    - threshold: the features best threshold.
    - impurity: the chosen impurity measure.

    Output: goodness of split using the provided impurity measure
    """
    current_impurity = impurity(data)
    rows = len(data)
    lt_data, gt_data = data_partition(data, column, threshold)
    weighted_impurity = (len(lt_data) / rows) * impurity(lt_data) \
                        + (len(gt_data) / rows) * impurity(gt_data)
    
    return current_impurity - weighted_impurity

def data_partition(data, feature, value):
    """
    Partitioning the data for to numpy lists, less than threshold
    and greater than threshold
    Input:
    - data: the training dataset.
    - feature: the feature column
    - value: the features best threshold.

    Output:
    lt_data: numpy array for less than threshold instances
    gt_data: numpy array for greater than threshold instances
    """
    lt_data = []
    gt_data = []

    for row in data: # iterate instance rows in data
        # split instances for less or greater than threshold
        if row[feature] > value:
            gt_data.append(row)
        else:
            lt_data.append(row)

    return np.array(lt_data), np.array(gt_data)

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the class prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    while node.children:
        if instance[node.feature] > node.value:
            node = node.children[1]
        else:
            node = node.children[0]
    pred = max(node.labels, key = node.labels.get)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    count = 0.0
    total = dataset.shape[0]
    for row in dataset:
        if predict(node, row) == row[-1]:
            count += 1
    accuracy = (count / total) * 100
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def print_tree(node):
    """
    Prints tree using the given node as the root, according the the example showed in the notebook.

	Input:
	-node: root node in the decision tree

	This function has no return value
	"""

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    print_preorder(node, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

def print_preorder(node, level):
    """
        Prints the tree recursively
        Input:
        - node: root node in the decision tree
        - level: current level of printing
        This function has no return value
    """
    print('  ' * level + str(node))
    if node.children:
        print_preorder(node.children[1], level + 1)
        print_preorder(node.children[0], level + 1)
    return

def chi_square(data, feature, value):
    """
    Calculates the Chi square value for a specific split

    Input:
    - data: the training dataset.
    - feature: column index of the feature to split with
    - value: threshold of split

    Output:
    - chi_value: Calculated Chi square value
    """

    # data size validation
    total = len(data)
    if total < 2:
        return 0

    # initiate general parameters
    p_y0 = len(data[(data[:, -1] == 0)]) / total
    p_y1 = len(data[(data[:, -1] == 1)]) / total

    # data purity validation
    if p_y0 * p_y1 == 0.0:
        return 0
    chi_value = 0.0

    # greater than threshold
    d_greater = data[(data[:, feature] > value)]
    p_greater = len(d_greater[(d_greater[:, -1] == 0)])
    n_greater = len(d_greater[(d_greater[:, -1] == 1)])
    d_greater = len(d_greater)
    if d_greater != 0:
        E0_greater = d_greater * p_y0
        E1_greater = d_greater * p_y1
        chi_value += (p_greater - E0_greater) ** 2 / E0_greater + (n_greater - E1_greater) ** 2 / E1_greater


    # less than threshold
    d_less = data[(data[:, feature] <= value)]
    p_less = len(d_less[(d_less[:, -1] == 0)])
    n_less = len(d_less[(d_less[:, -1] == 1)])
    d_less = len(d_less)
    if d_less != 0:
        E0_less = d_less * p_y0
        E1_less = d_less * p_y1
        chi_value += (p_less - E0_less) ** 2 / E0_less + (n_less - E1_less) ** 2 / E1_less

    return chi_value

def count_nodes(root):
    """
    Counts the number of internal nodes in a tree for a given root recursively
    Input:
    - root: the root of the tree

    Output:
    Void function
    """
    if not root.children: # if leaf, returns 0
        return 0
    return count_nodes(root.children[0]) + count_nodes(root.children[1]) + 1

def list_parents(root):
    """
    Creates a list of nodes, appending only parents of leaves
    Input:
    -root: decision tree root

    Output:
    -parents: lists of nodes containing only parents of leaves
    """
    # initiate list and queue
    parents = []
    queue = [root]

    # while queue is not empty, appends parents of leaves to parents list
    while queue:
        node = queue.pop(0)
        if node.children:
            queue.append(node.children[0])
            queue.append(node.children[1])
        else: # *try to check for unique later*
            parents.append(node.parent)

    return parents

def calc_accuracy_parent(root, parent, data):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset, dropping the parent's children and returning children after calculation

    Input:
    - node: a node in the decision tree.
    - parent: the parent the should be prune children
    - data: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset
    as if the the parent would have not splitted (%).
    """
    # save children nodes
    temp_left = parent.children[0]
    temp_right = parent.children[1]

    # prune and calculate accuracy
    parent.children.clear()
    acc = calc_accuracy(root, data)

    # return tree to start position
    parent.children.append(temp_left)
    parent.children.append(temp_right)

    return acc * 100

def post_pruning(root_post, X_train, X_test):
    """"
    For each leaf in the tree, calculate the test accuracy of the tree
    assuming no split occurred on the parent of that leaf and find the best such parent
    Updates number of nodes, accuracy of train and test data for each prune

    Input:
    - root_post: the root of the tree to post-prune
    - X_train: training data
    - X_test: testing data

    Output:
    Corollary for each iteration, returns:
    - num_int_nodes: list of number of internal nodes in the tree
    - acc_train: list of accuracy over train data
    - acc_train: list of accuracy over test data
    """

    # initiate parameters
    num_int_nodes = []
    acc_train = []
    acc_test = []

    # iterate and prune tree till only root left
    while root_post.children:
        num_int_nodes.append(count_nodes(root_post))
        acc_train.append(calc_accuracy(root_post, X_train))
        acc_test.append(calc_accuracy(root_post, X_test))

        parents = list_parents(root_post)
        max_acc = 0.0
        cut = None
        for parent in parents:
            acc = calc_accuracy_parent(root_post, parent, X_test)
            if acc > max_acc:
                max_acc = acc
                cut = parent
        cut.children.clear()

    # append last iteration (root only) parameters to lists
    num_int_nodes.append(count_nodes(root_post))
    acc_train.append(calc_accuracy(root_post, X_train))
    acc_test.append(calc_accuracy(root_post, X_test))

    return num_int_nodes, acc_train, acc_test


