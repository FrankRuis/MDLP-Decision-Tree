# MDLP-Decision-Tree
Based on a paper by Quinlan and Rivest, *“Inferring decision trees using the minimum description lenght
principle”. In: Information and computation 80.3 (1989), pp. 227–248.*

The main idea is that trying to transmit a theory for how data is classified and the exceptions for that theory using as few bits as possible will result in a theory that will also accurately classify new data. Because if an attribute’s relationship with the object’s class is significant enough to save bits during transmission, it will most likely be a worthwhile addition to a decision tree. 

Including the theory in the cost of the transmission takes care of the overfitting problem, because a more complex theory that might classify the training data perfectly by assigning a leaf node to each training object will result in a relatively high theory cost. On the other hand, a too simple theory will result in a high cost for encoding all of the exceptions.

### Usage

    from mdlp import Tree

	train = # list of feature arrays with the label as the last element
	test = # list of feature arrays with the label as the last element
	tree = Tree(train)
	predictions  = [tree.classify(t) for  t  in  test]
