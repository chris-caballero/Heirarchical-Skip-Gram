# Hierarchical Softmax

The Hierarchical Softmax is an optimization over traditional softmax which uses a Huffman Tree to reduce the computational burden of performing matrix multiplication across the entire vocabulary. This tree serves as the output layer for the Skip-Gram model (in this case).

I started this project with the intention of deepening my understanding of machine learning model design and gaining insights into the intricacies of backpropagation.

To accomplish this, essentially all components were developed without the use of external libraries. This, in turn, severely limits efficiency.
***
## Overview

The Hierarchical Softmax consists of the following main components:

- **Node**: Represents a node in the Huffman Tree. It contains information such as index, path, left and right child nodes, parent node, probability, and node vector.
- **HuffmanTree**: Represents the Huffman Tree itself, built with `Node` objects. It is responsible for building the tree based on word counts, assigning Huffman paths to each word, and providing methods to navigate and display the tree.
- **ModelLayer**: Contains utility functions for generating an embedding matrix and building a softmax tree based on word counts.
- **SkipGramModel**: Implements the Skip-gram model with hierarchical softmax. It performs training steps, computes probabilities, and evaluates the model.
***
## Usage

To use the Hierarchical Softmax Model, you can follow these steps:

- Create a `SkipGramModel` object with the necessary parameters, including the learning rate, word counts, vocabulary size, integer-to-word and word-to-integer encodings, and the pre-built hierarchical softmax tree and embedding matrix if available.
- Train the model using the `train` method, passing the target and context word indices, number of epochs, and optional parameters for displaying progress.
- Test the model's performance using the `evaluate` method.
- Display the Huffman paths for each word using the `show_paths` method.

**Note:** The code provided in this README is just a simplified implementation of the Hierarchical Softmax and Skip-gram model. You can adapt it to your specific use case and extend it as needed.
***
## Dependencies

The following dependencies are required to run the code:

- numpy
- scipy
- sklearn
- seaborn
- matplotlib

You can install them using `pip`:

```
pip install numpy scipy scikit-learn seaborn matplotlib
```

