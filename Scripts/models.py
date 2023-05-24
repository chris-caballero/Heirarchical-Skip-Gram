'''
Hierarchical Softmax¶
- Constructs a Huffman Tree with vocabulary words as leaves.
- The root will be the start of our 'output layer.'
- After we get the hidden layer from the embedding, we take the inner product with the vector 
of the current root and move left->right based on our target/context word's huffman encodings.
- After taking the inner product, we first activate with sigmoid before continuing. On successive iterations, we multiply these sigmoid values to get our final probability.
'''

import numpy as np
import heapq
from scipy.stats import ortho_group

class Node:
    """Node class for the Huffman Tree."""
    def __init__(self, index=-1, path='', left=None, right=None, parent=None, prob=None, vector=None, embedding_dim=10):
        """
        Initialize a node in the Huffman Tree.

        Args:
        - index (int): Index of the node.
        - path (str): Huffman path from the root to the node.
        - left (Node): Left child node.
        - right (Node): Right child node.
        - parent (Node): Parent node.
        - prob (float): Probability associated with the node.
        - vector (ndarray): Vector associated with the node.
        - embedding_dim (int): Dimension of the embedding vector.
        """
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
        """
        Display information about the node.

        Args:
        - encoding (dict): Dictionary mapping word indices to their corresponding words.
        - vocab_size (int): Size of the vocabulary.
        """
        if self.index < 0 or self.index > vocab_size:
            print(f'Prob: {self.prob}, Path: {self.path}\nVector: {self.vector}\n')

    def show_vector(self):
        """Display the vector associated with the node."""
        print(self.vector)

    def __lt__(self, other):
        """
        Compare the probability of two nodes.
        """
        return self.prob < other.prob

class HuffmanTree:
    """Huffman Tree class for hierarchical softmax."""
    def __init__(self, root=Node()):
        """
        Initialize the Huffman Tree.

        Args:
        - root (Node): Root node of the tree.
        """
        self.root = root

    def build_tree(self, wordcounts, int_encoding, word_encoding):
        """
        Build the Huffman Tree based on word counts.

        Args:
        - wordcounts (dict): Dictionary mapping words to their counts.
        - int_encoding (dict): Dictionary mapping words to their integer indices.
        - word_encoding (dict): Dictionary mapping integer indices to their corresponding words.
        """
        leaf_nodes = []
        total_words = sum(wordcounts.values())
        for w in wordcounts:
            leaf_nodes.append(Node(index=int_encoding[w], prob=round(wordcounts[w]/total_words, 4)))

        heapq.heapify(leaf_nodes)

        while len(leaf_nodes) > 1:
            n1 = heapq.heappop(leaf_nodes)
            n2 = heapq.heappop(leaf_nodes)
            parent = Node(left=n1, right=n2, prob=round(n1.prob+n2.prob, 3))
            n1.parent = parent
            n2.parent = parent
            heapq.heappush(leaf_nodes, parent)

        self.root = leaf_nodes[0]

    def set_paths(self, root):
        """
        Set Huffman paths for all nodes in the tree.

        Args:
        - root (Node): Root node of the tree.
        """
        if root.left:
            root.left.path = root.path + "0"
            self.set_paths(root.left)
        if root.right:
            root.right.path = root.path + "1"
            self.set_paths(root.right)
    
    def find_path(self, root, word_idx):
        """
        Find the Huffman path for a given word index.

        Args:
        - root (Node): Root node of the Huffman Tree.
        - word_idx (int): Index of the word to find the path for.

        Returns:
        - str: The Huffman path for the word.
        """
        if root.left:
            self.find_path(root.left, word_idx)
        if root.index == word_idx:
            return root.path
        if root.right:
            self.find_path(root.right, word_idx)

    def show_tree(self, root):
        """
        Display the Huffman Tree.

        Args:
        - root (Node): Root node of the Huffman Tree.
        """
        if root.left:
            self.show_tree(root.left)
        root.show_node()
        if root.right:
            self.show_tree(root.right)

class ModelLayer:
    def Embedding(input_dim, output_dim):
        """
        Generates an embedding matrix using an orthogonal matrix and performs dimensionality reduction.

        Args:
            input_dim (int): Input dimension of the embedding matrix.
            output_dim (int): Output dimension of the embedding matrix.

        Returns:
            np.ndarray: Embedding matrix.
        """
        embedding_matrix = ortho_group.rvs(dim=input_dim)
        for i in reversed(range(output_dim, input_dim)):
            embedding_matrix = np.delete(embedding_matrix, i, 1)
        return embedding_matrix

    def SoftmaxTree(count, int_to_word, word_to_int, tree=None):
        """
        Builds a softmax tree based on word counts and assigns Huffman codes to each word.

        Args:
            count (list): List of word frequencies in the vocabulary.
            int_to_word (dict): Mapping from word indices to corresponding words.
            word_to_int (dict): Mapping from words to corresponding indices.
            tree (HuffmanTree, optional): Pre-existing Huffman tree object. Defaults to None.

        Returns:
            HuffmanTree: Constructed softmax tree.
        """
        tree = HuffmanTree()
        tree.build_tree(count, word_encoding=int_to_word, int_encoding=word_to_int)
        tree.set_paths(tree.root)
        return tree


class SkipGramModel:
    """Skip-gram model with hierarchical softmax."""
    def __init__(self, learning_rate, counts, vocab_size, int_to_word, word_to_int, embedding=None, embedding_dim=10, tree=None):
        """
        Initializes the SkipGramModel.

        Args:
            learning_rate (float): Learning rate for model training.
            counts (list): List of word frequencies in the vocabulary.
            vocab_size (int): Size of the vocabulary.
            int_to_word (dict): Mapping from word indices to corresponding words.
            word_to_int (dict): Mapping from words to corresponding indices.
            embedding (np.ndarray, optional): Pre-trained word embedding matrix. Defaults to None.
            embedding_dim (int, optional): Dimensionality of word embeddings. Defaults to 10.
            tree (ModelLayer.SoftmaxTree, optional): Pre-built hierarchical softmax tree. Defaults to None.
        """
        self.learning_rate = learning_rate
        self.tree = ModelLayer.SoftmaxTree(counts, int_to_word, word_to_int)
        self.paths = dict()
        self.get_paths(self.tree.root)
        self.embedding = embedding
        if embedding is None:
            self.embedding = ModelLayer.Embedding(vocab_size, embedding_dim)

    def sigmoid(self, output):
        """
        Applies the sigmoid activation function to the given output.

        Args:
            output (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying sigmoid activation.
        """
        return 1 / (1 + np.exp(-output))

    def probability(self, hidden, inner_unit):
        """
        Computes the probability using the inner product between the hidden layer and the inner unit.

        Args:
            hidden (np.ndarray): Hidden layer vector.
            inner_unit (np.ndarray): Inner unit vector.

        Returns:
            float: Computed probability.
        """
        inner_product = np.dot(inner_unit.T, hidden)
        return self.sigmoid(inner_product)

    def hierarchical_softmax(self, hidden, path):
        """
        Performs hierarchical softmax for a given hidden layer vector and Huffman path.

        Args:
            hidden (np.ndarray): Hidden layer vector.
            path (str): Huffman path.

        Returns:
            float: Output probability.
            float: Negative loss.
            np.ndarray: Gradient with respect to the hidden layer.
        """
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
                output *= (1 - curr_node.prob)
                loss += np.log(1 - curr_node.prob + 0.000001)
                curr_node = curr_node.right
            path = path[1:]

        return output, -loss, hidden_grad

    def training_step(self, target, context):
        """
        Performs a single training step for the Skip-gram model.

        Args:
            target (int): Target word index.
            context (list): List of context word indices.

        Returns:
            list: List of output probabilities for each context word.
            float: Total loss.
            np.ndarray: Gradient with respect to the hidden layer.
        """
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

    def train(self, x_train, y_train, epochs=100, split=50, verbose=1):
        """
        Trains the Skip-gram model.

        Args:
            x_train (list): List of target word indices.
            y_train (list): List of corresponding context word indices.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            split (int, optional): Interval for displaying loss during training. Defaults to 50.
            verbose (int, optional): Verbosity mode. 0 - silent, 1 - display progress. Defaults to 1.

        Returns:
            list: List of average losses per epoch.
            list: List of model states at each epoch.
        """
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
        """
        Computes the output probability for a given hidden layer vector and Huffman path.

        Args:
            hidden (np.ndarray): Hidden layer vector.
            path (str): Huffman path.

        Returns:
            float: Output probability.
        """
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
        """
        Performs forward propagation for a given target word and context words.

        Args:
            target (int): Target word index.
            context (list): List of context word indices.

        Returns:
            list: List of output probabilities for each context word.
            np.ndarray: Hidden layer vector.
        """
        hidden = self.embedding[target]
        outputs = []
        for word in context:
            path = self.paths[word][0]
            output = self.get_outputs(hidden, path)
            outputs.append(output)
        return outputs, hidden

    def evaluate(self, x, vocab_size):
        """
        Evaluates the model for a given target word index and vocabulary size.

        Args:
            x (int): Target word index.
            vocab_size (int): Size of the vocabulary.

        Returns:
            list: List of output probabilities for all words in the vocabulary.
        """
        outputs, _ = self.forward(x, [i for i in range(vocab_size)])
        return outputs

    def get_paths(self, root):
        """
        Recursively retrieves the paths for each node in the hierarchical softmax tree.

        Args:
            root (Node): Root node of the tree.
        """
        if root.left:
            self.get_paths(root.left)
        if root.index > -1:
            self.paths[root.index] = (root.path, root.prob)
        if root.right:
            self.get_paths(root.right)

    def show_paths(self, encoding):
        """
        Displays the paths for each word in the vocabulary.

        Args:
            encoding (dict): Mapping from word indices to corresponding words.
        """
        for idx in self.paths:
            print(f'Word: {encoding[idx]}, Path: {self.paths[idx]}\n')
