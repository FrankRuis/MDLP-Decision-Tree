import math
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Decision:
    def __init__(self, pos, val):
        self.pos = pos
        self.val = val

    def is_continuous(self):
        return isinstance(self.val, int) or isinstance(self.val, float)

    def decide(self, feature):
        if not feature[self.pos]:
            return False

        if self.is_continuous():
            return feature[self.pos] < self.val
        else:
            return feature[self.pos] == self.val

    def __repr__(self):
        if self.is_continuous():
            return '{} < {}'.format(self.pos, self.val)
        else:
            return '{} == {}'.format(self.pos, self.val)


class Node:
    def __init__(self, decision, left, right, tree):
        self.tree = tree
        self.decision = decision
        self.left = left
        self.right = right

    def __repr__(self):
        return repr(self.decision)


class Leaf:
    def __init__(self, tree, elements):
        self.tree = tree
        self.elements = elements
        self.counts = Counter()
        for e in elements:
            self.counts[e[-1]] += 1

        self.class_ = max(self.counts, key=lambda x: self.counts[x])

    def __repr__(self):
        return repr(self.counts)


class Tree:
    def __init__(self, train, test=None):
        self.features = train
        self.features_test = test
        self.classes = set([f[-1] for f in self.features])

        self.attributes = [[] for _ in self.features[0][:-1]]
        for feature in self.features:
            for i, e in enumerate(feature[:-1]):
                if e is None:
                    continue
                self.attributes[i].append(e)

        for i, attr in enumerate(self.attributes):
            self.attributes[i] = list(sorted(set(attr)))

        self.root = self._build_tree(Leaf(self, self.features))
        self._prune_tree(self.root)

    def classify(self, feature):
        cur = self.root
        while isinstance(cur, Node):
            if cur.decision.decide(feature):
                cur = cur.right
            else:
                cur = cur.left

        return cur.class_

    def _build_tree(self, leaf: Leaf):
        mdlp, decision = self._best_split(leaf)
        if not decision:
            return leaf

        left, right = self._split(leaf.elements, decision)

        return Node(decision, self._build_tree(Leaf(self, left)), self._build_tree(Leaf(self, right)), self)

    def _prune_tree(self, root):
        if root is None:
            return

        s1 = []
        s2 = []
        s1.append(root)
        while s1:
            node = s1.pop()
            if isinstance(node, Node):
                s2.append(node)
                node.left.parent = node
                node.left.branch = 'L'
                node.right.parent = node
                node.right.branch = 'R'
                s1.append(node.left)
                s1.append(node.right)

        while s2:
            node = s2.pop()
            if isinstance(node.left, Leaf) and isinstance(node.right, Leaf):
                cur_mdl = 3 + self._mdlp_attribute(node.decision) + (2 * math.log2(len(self.classes))) \
                          + self._mdlp_exceptions([l[-1] for l in node.left.elements]) \
                          + self._mdlp_exceptions([r[-1] for r in node.right.elements])
                new_mdl = 1 + self._mdlp_exceptions([e[-1] for e in node.left.elements + node.right.elements])
                if cur_mdl >= new_mdl:
                    if node.branch == 'L':
                        node.parent.left = Leaf(node.tree, node.left.elements + node.right.elements)
                    elif node.branch == 'R':
                        node.parent.right = Leaf(node.tree, node.left.elements + node.right.elements)

    def _best_split(self, cur_leaf: Leaf):
        features = cur_leaf.elements
        best = (math.inf, None)
        for i, attr in enumerate(self.attributes):
            if len(attr) == 0:
                continue

            if not isinstance(attr[0], str):
                sq = math.sqrt(len(attr))
                attr = [attr[i] for i in range(0, len(attr), math.floor(sq))]

            for val in attr:
                decision = Decision(i, val)
                left, right = self._split(features, decision)

                if not left or not right:
                    continue

                # 2 new nodes, 1 new default class, 1 new attribute, 2 exception costs, subtract previous exception cost
                mdlp = 2 + math.log2(len(self.classes)) + self._mdlp_attribute(decision) \
                    + self._mdlp_exceptions([l[-1] for l in left]) \
                    + self._mdlp_exceptions([r[-1] for r in right]) \
                    - self._mdlp_exceptions([e[-1] for e in cur_leaf.elements])
                if mdlp < best[0]:
                    best = (mdlp, decision)

        return best

    def _split(self, features, decision):
        left, right = [], []
        for feature in features:
            if decision.decide(feature):
                right.append(feature)
            else:
                left.append(feature)

        return left, right

    def _calc_l(self, n, k, b):
        n_choose_k = math.log2(math.factorial(n) // (math.factorial(n - k) * math.factorial(k)))
        return n_choose_k + math.log2(b + 1)

    def _mdlp_exceptions(self, labels, counts=None):
        if not counts:
            counts = {}
            for l in labels:
                if l not in counts:
                    counts[l] = 1
                else:
                    counts[l] += 1

        default = max(counts, key=lambda x: counts[x])
        n = len(labels)
        k = counts[default]
        b = (len(labels) - 1) // 2
        del counts[default]
        result = self._calc_l(n, k, b)

        return 8 * result if len(counts) <= 1 else 8 * (result + self._mdlp_exceptions([l for l in labels if l != default],
                                                                             counts))

    def _count_nodes(self, node, counts=np.array([0, 0, 0])):
        if isinstance(node, Leaf):
            return counts + [1, 1, 0]
        else:
            return counts + [1, 0, self._mdlp_attribute(node.decision)] + self._count_nodes(
                node.left) + self._count_nodes(node.right)

    def _mdlp_tree(self, root):
        n, k, a = self._count_nodes(root)

        return n + (math.log2(len(self.classes)) * k) + a

    def _mdlp_attribute(self, decision):
        attr = self.attributes[decision.pos]
        if isinstance(attr, str):
            return math.log2(len(self.attributes)) + math.log2(len(attr))
        else:
            return math.log2(len(self.attributes)) + math.log2(math.sqrt(len(attr)))


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    train = []
    for i in range(len(X_train)):
        train.append(np.append(X_train[i], y_train[i]))

    test = []
    for i in range(len(X_test)):
        test.append(np.append(X_test[i], y_test[i]))

    tree = Tree(train, test)
    preds = [tree.classify(t) for t in test]
    acc = [a == b for a, b in zip(preds, y_test)]

    print('Accuracy:', sum(acc) / len(acc) * 100)
