'''
Hierarchical Softmax¶
- At this point we have a Huffman Tree with words of the vocabulary as leaves.
- The root will be the start of our 'output layer.'
- After we get the hidden layer from the embedding, we take the inner product with the vector 
of the current root and move left->right based on our target/context word's huffman encodings.
- After taking the inner product, we first activate with sigmoid before continuing. On successive iterations, we multiply these sigmoid values to get our final probability.
'''

import numpy as np
import heapq
from scipy.stats import ortho_group

class Node(object):
    def __init__(self, index = -1, path = '', left = None, right = None, parent = None, prob = None, vector = None, embedding_dim = 10):
        self.index = index 
        self.left = left
        self.right = right
        self.parent = parent
        self.prob = prob
        self.path = path
        self.vector = vector
        if self.vector is None:
            self.vector = np.random.uniform(-1, 1, embedding_dim)
    
    def show_node(self, encoding, vocab_size):
        if self.index < 0 or self.index > vocab_size:
            print(f'Prob: {self.prob}, Path: {self.path}\nVector: {self.vector}\n')
            return
    
    def show_vector(self):
        print(self.vector)
    
    def __lt__(self, other):
        return self.prob < other.prob

class HuffmanTree:
    def __init__(self, root = Node()):
        self.root = root
    
    def build_tree(self, wordcounts, int_encoding, word_encoding):
        leaf_nodes = []
        total_words = sum(wordcounts.values())
        for w in wordcounts:
            leaf_nodes.append(Node(index=int_encoding[w], prob=round(wordcounts[w]/total_words, 4)))

        heapq.heapify(leaf_nodes)

        i = 0
        while len(leaf_nodes) > 1:
            n1 = heapq.heappop(leaf_nodes)
            n2 = heapq.heappop(leaf_nodes)
            parent = Node(left=n1, right=n2, prob=round(n1.prob+n2.prob, 3))
            # set parent for the child nodes so we can efficiently propogate the gradient
            n1.parent = parent
            n2.parent = parent
            heapq.heappush(leaf_nodes, parent)
            
            i += 1

        self.root = leaf_nodes[0]

    def set_paths(self, root):
        if root.left:
            root.left.path = root.path + '0'
            self.set_paths(root.left)
        if root.right:
            root.right.path = root.path + '1'
            self.set_paths(root.right)
    
    def find_path(self, root, word_idx):
        if root.left:
            self.show_tree(root.left)
        if root.index == word_idx:
            return root.path
        if root.right:
            self.show_tree(root.right)
    
    def show_tree(self, root):
        if root.left:
            self.show_tree(root.left)
        root.show_node()
        if root.right:
            self.show_tree(root.right)

class ModelLayer:
    def Embedding(input_dim, output_dim):
        embedding_matrix = ortho_group.rvs(dim=input_dim)
        for i in reversed(range(output_dim, input_dim)):
            embedding_matrix = np.delete(embedding_matrix, i, 1)
        return embedding_matrix

    def SoftmaxTree(count, int_to_word, word_to_int, tree = None):
        tree = HuffmanTree()
        tree.build_tree(count, word_encoding=int_to_word, int_encoding=word_to_int)
        tree.set_paths(tree.root)
        return tree

class SkipGramModel:
    def __init__(self, learning_rate, counts, vocab_size, int_to_word, word_to_int, embedding = None, embedding_dim = 10, tree = None):
        self.learning_rate = learning_rate
        self.tree = ModelLayer.SoftmaxTree(counts, int_to_word, word_to_int)
        self.paths = dict()
        self.get_paths(self.tree.root)
        self.embedding = embedding
        if embedding is None:
            self.embedding = ModelLayer.Embedding(vocab_size, embedding_dim)
            
    def sigmoid(self, output):
        return 1/(1 + np.exp(-output))

    def probability(self, hidden, inner_unit):
        inner_product = np.dot(inner_unit.T, hidden)
        return self.sigmoid(inner_product)
    
    def hierarchical_softmax(self, hidden, path):
        output = 1
        loss = 0
        hidden_grad = 0
        curr_node = self.tree.root
        
        while len(path) > 0:
            inner_unit = curr_node.vector
            curr_node.prob = self.probability(hidden, inner_unit)
            inner_product = curr_node.prob + (int(path[0]) - 1)
            
            inner_grad = np.dot(inner_product, hidden)
            hidden_grad += np.dot(inner_product, inner_unit)
            curr_node.vector -= self.learning_rate * inner_grad
            
            if path[0] == '0':
                output *= curr_node.prob
                loss += np.log(curr_node.prob + 0.000001)
                curr_node = curr_node.left
            else:
                output *= (1-curr_node.prob)
                loss += np.log(1-curr_node.prob + 0.000001)
                curr_node = curr_node.right
            path = path[1:]
            
        return output, -loss, hidden_grad
    
    def training_step(self, target, context):
        loss = 0
        hidden_grad = 0
        outputs, hidden = [], self.embedding[target]
        for word in context:
            path = self.paths[word][0]
            output, curr_loss, curr_grad = self.hierarchical_softmax(hidden, path)
            
            outputs.append(output)
            loss += curr_loss
            hidden_grad += curr_grad
            
        hidden -= self.learning_rate * hidden_grad
            
        return outputs, loss, hidden
    
    def train(self, x_train, y_train, epochs = 100, split = 50, verbose = 1):
        from sklearn.decomposition import TruncatedSVD, PCA

        svd = TruncatedSVD(n_components=2)
        # pca = PCA(n_components=2)
        history = []
        states = []
        for i in range(epochs):
            avg_loss = 0
            for (x, y) in zip(x_train, y_train):
                _, loss, hidden = self.training_step(x, y)
                self.embedding[x] = hidden
                avg_loss += loss

            x = self.embedding.copy()
            svd.fit(x)
            # x = pca.fit_transform(x)
            states.append(x)
            
            avg_loss /= len(x_train)
            history.append(avg_loss)
            if verbose and i % split == 0:
                print(f'Epoch {i}: Loss = {avg_loss}')
        
        return history, states

    def get_outputs(self, hidden, path):
        output = 1
        curr_node = self.tree.root
        while len(path) > 0:
            inner_unit = curr_node.vector
            prob = self.probability(hidden, inner_unit)
            if path[0] == '0':
                output *= prob
                curr_node = curr_node.left
            else:
                output *= (1 - prob)
                curr_node = curr_node.right
            path = path[1:]
        return output
    
    def forward(self, target, context):
        hidden = self.embedding[target]
        outputs = []
        for word in context:
            path = self.paths[word][0]
            output = self.get_outputs(hidden, path)
            outputs.append(output)
        return outputs, hidden
    
    def evaluate(self, x, vocab_size):
        outputs, _ = self.forward(x, [i for i in range(vocab_size)])
        return outputs
    
    def get_paths(self, root):
        if root.left:
            self.get_paths(root.left)
        if root.index > -1:  
            self.paths[root.index] = (root.path, root.prob)
        if root.right:
            self.get_paths(root.right)

    def show_paths(self, encoding):
        for idx in self.paths:
            print(f'Word: {encoding[idx]}, Path: {self.paths[idx]}\n')
