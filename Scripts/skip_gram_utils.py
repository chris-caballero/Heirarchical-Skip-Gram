import re
import numpy as np
import seaborn as sns
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from models import *
'''
VISUALIZATION SETUP
'''
sns.set_style('darkgrid') # dark grid, white grid, dark, white, ticks
plt.rc('axes',  titlesize=14, labelsize=12)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('font', size=12)

# process contents of a text file into list of words
def get_data(filename):
    contents = []
    with open(filename) as f:
        contents = f.read()
        words = re.split('[^a-zA-Z0-9]+', contents)
        data = []
        for w in words:
            if len(w) > 0:
                data.append(w.lower())
    return data

# integer encoding so each word gets a unique index
def encode_words(data):      
    # we want unique words
    vocabulary = list(set(data))
    word_to_int = dict((word, i) for i, word in enumerate(vocabulary))
    int_to_word = dict((i, word) for i, word in enumerate(vocabulary))
    return word_to_int, int_to_word, vocabulary

# prints the training data in string representation
def print_training_data(training_data, int_to_word):
  for pair in training_data:
    print('Target: ', int_to_word[pair[0][1]])
    print('Context: ', pair[1][1])
    print('\n')

# prints the training data in vector representation
def print_training_data_raw(training_data):
  for pair in training_data:
    print('Target: ', pair[0], '\nContext: ', pair[1])
    print('\n')

def print_info(testset, embedding_matrix, weights_out, word_to_int, int_to_word):
    from skip_gram import forward_pass, cos_similarity_dict
    num_similar = 10
    for word in testset:
        _, y, _ = forward_pass(word_to_int[word], embedding_matrix, weights_out)
        similarities_in, similarities_out = cos_similarity_dict(word, embedding_matrix, weights_out.T, word_to_int, int_to_word)
        # print target word, context prediction with certainty, and a few of the most similar word vectors 
        print('{:<15}'.format('Target:'), word)
        print('{:<15}'.format('Prediction:'), int_to_word[np.argmax(y)])
        print('{:<15}'.format('Probability:'), y[np.argmax(y)].round(4))
        print('{:<15}'.format('Most similar Input:'), list(similarities_in)[:num_similar])
        print('{:<15}'.format('Most similar Output:'), list(similarities_out)[:num_similar], '\n')

# visualize data as vectors (mode = 0) or predictions (mode != 0) over time
def visualize_data(vectors_over_time=None, x=None, y_true=None, pred_over_time=None, epochs=30, int_to_word=None, mode = 0, savefile='./Animations/anim.gif'):
    fig = plt.figure()
    if vectors_over_time is not None:
        colors = sns.color_palette("hls", len(vectors_over_time[0]))
    else:
        colors = sns.color_palette("hls", 8)[:2]
    def visualize_data_1(i=int):
        if i < len(vectors_over_time):
            x1, x2 = [], []
            for v in vectors_over_time[i]:
                x1.append(v[0])
                x2.append(v[1])
            fig.clear()
            # print(len(vectors_over_time[i][1]))
            plt.scatter(x1, x2, c=colors)
            for j in range(len(vectors_over_time[i])):
                plt.annotate(int_to_word[j], xy=(vectors_over_time[i][j][0], vectors_over_time[i][j][1]), xytext=(0,7), textcoords='offset points', ha='left', va='center')

    def visualize_data_2(i=int):
        if i < len(pred_over_time):
            fig.clear()
            plt.plot(x, y_true, color=colors[0])
            plt.plot(x, pred_over_time[i], color=colors[1])
    if mode == 0:
        anim = ani.FuncAnimation(fig, visualize_data_1, interval=epochs/10)
    else:
        anim = ani.FuncAnimation(fig, visualize_data_2, interval=epochs/10)

    gif = ani.PillowWriter(fps=50) 
    anim.save(savefile, writer=gif)
    plt.show()

def visualize_data_discrete(vectors_over_time, x, epochs, int_to_word):
    import matplotlib
    import tkinter as Tk
    from matplotlib.widgets import Slider
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    matplotlib.use('TkAgg')
    root = Tk.Tk()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.1)

    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    
    ax_time = fig.add_axes([0.12, 0.95, 0.78, 0.03])
    slider = Slider(ax_time, 'Epoch', 0, epochs/10, valinit=0, valstep=1, dragging=True)

    colors = [[random(), random(), random()] for _ in range(len(x))]

    def visualize_data_1(i):
        if i < len(vectors_over_time):
            ax.clear()
            ax.scatter(vectors_over_time[i][0], vectors_over_time[i][1], c=colors)
            for j in range(len(vectors_over_time[0][0])):
                ax.annotate(  int_to_word[j], 
                            xy=(vectors_over_time[i][0][j], vectors_over_time[i][1][j]), 
                            xytext=(0,7), textcoords='offset points', 
                            ha='left', va='center', c=colors[j])
            fig.canvas.draw_idle()

    def visualize_data_2(i):
        if i < len(pred_over_time):
            ax.clear()
            ax.plot(x, y_true, color='green')
            ax.plot(x, pred_over_time[i], color='red')

    slider.on_changed(visualize_data_1)
    Tk.mainloop()

def get_true_context(entry, window, int_to_word):
    entry = sorted(entry.items(), key=lambda item: item[1], reverse=True)
    true_probs, true_context = [], []
    for i, y in enumerate(entry):
        if i == 2*window:
            break
        true_probs.append(y[1])
        true_context.append(int_to_word[y[0]])
    return true_probs, true_context

def plot_context_predictions(target, contexts, certainties, gt_table, window, word_to_int, int_to_word):
    colors = sns.color_palette('pastel')
    fig, (ax1,  ax2) = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    plt.suptitle('Target: {}'.format(target), weight='bold')
    
    ax1.pie(certainties, labels=contexts, autopct='%.0f %%', pctdistance=.7, colors=colors, shadow=True)
    
    entry = gt_table[word_to_int[target]]
    true_probs, true_context = get_true_context(entry, window, int_to_word)
    
    ax2.pie(true_probs, labels=true_context, autopct='%.0f %%', pctdistance=.7, colors=colors, shadow=True)

    plt.show()

def show_predictions(model, gt_table, window=2, random = False, num_samples = 3, count = None, vocab = None, word_to_int = None, int_to_word = None):
    for itr, target in enumerate(vocab):
        if random:
            new_model = SkipGramModel(0.05, count, len(vocab), int_to_word, word_to_int)
            outputs = new_model.evaluate(word_to_int[target],  len(vocab))
        else:
            outputs = model.evaluate(word_to_int[target],  len(vocab))

        output_array = np.array(outputs)
        output_array = np.argsort(output_array)

        start = len(vocab) - 2*window
        contexts, certainties = [], []
        for i in reversed(range(start, len(vocab))):
            idx = output_array[i]
            contexts.append(int_to_word[idx])
            certainties.append(outputs[idx])
        
        plot_context_predictions(target, contexts, certainties, gt_table, window, word_to_int, int_to_word)
        
        if itr == num_samples - 1:
            break

def dense_predictions(embedding, output_weights, gt_table, embedding_dim=10, window=2, num_samples=3, vocab = None, word_to_int = None, int_to_word = None, random=False):
    from skip_gram import create_model_params, forward_pass
    
    for itr, target in enumerate(vocab):
        # print(target)
        if random:
            embedding, output_weights = create_model_params(embedding_dim, len(vocab))
            _, outputs, _ = forward_pass(itr, embedding, output_weights)
        else:
            _, outputs, _ = forward_pass(itr, embedding, output_weights)

        output_array = np.array(outputs)
        output_array = np.argsort(output_array)

        start = len(vocab) - 2*window
        contexts, certainties = [], []
        for i in reversed(range(start, len(vocab))):
            idx = output_array[i]
            contexts.append(int_to_word[idx])
            certainties.append(outputs[idx])
        
        plot_context_predictions(target, contexts, certainties, gt_table, window, word_to_int, int_to_word)
        
        if itr == num_samples - 1:
            break

def setup_model(data, window_size = 2, start_size = 50):
    from collections import Counter
    from skip_gram import generate_data

    size = start_size
    V = 0
    curr_data = None
    counts = None
    w_to_i, i_to_w = None, None
    vocab = None
    while V < size and start_size < len(data):
        curr_data = data[:start_size]
        counts = Counter(curr_data)
        w_to_i, i_to_w, vocab = encode_words(curr_data)
        V = len(vocab)
        start_size += 1

    if curr_data == None:
        curr_data = data[:] 
    x, y, _ = generate_data(curr_data, window_size, w_to_i)
    model = SkipGramModel(0.01, counts, V, i_to_w, w_to_i)

    return model, x, y, V

def setup_dense(data, window_size, embedding_dim, start_size=50):
    from skip_gram import generate_data, create_model_params

    size = start_size
    V = 0
    curr_data, w_to_i, vocab = None, None, None
    while V < size and start_size < len(data):
        curr_data = data[:start_size]
        w_to_i, _, vocab = encode_words(curr_data)
        start_size += 1
        V = len(vocab)

    if curr_data == None:
        curr_data = data[:] 
    x, y, _ = generate_data(curr_data, window_size, w_to_i)

    embedding, output_weights = create_model_params(embedding_dim, len(vocab))

    return embedding, output_weights, x, y, len(vocab)


def time_model(model, x, y):
    from time import time
    start_time = time()
    model.train(x, y, epochs=1, verbose=0)
    elapsed_time = time() - start_time
    return elapsed_time

def time_dense(embedding, output_weights, x, y):
    from skip_gram import train
    from time import time
    start_time = time()
    train(x, y, embedding, output_weights, epochs=1, verbose=0)
    elapsed_time = time() - start_time
    return round(elapsed_time, 2)

def train_time_scaling(file, window = 2, itrs = 40, step_size=20, mode=0, dim=10):
    data = get_data(file)
    step_times = []
    sizes = []
    start_size = 50
    
    for i in range(itrs):
        size = start_size + i * step_size
        if i % 10 == 0:
            print(f'{round(i/itrs,  2)*100}% complete')
        if mode == 0:
            model, x, y, size = setup_model(data, window_size=window, start_size=size)
            step_times.append(time_model(model, x, y))
        else:
            embedding, output_weights, x, y, size = setup_dense(data, window, dim, start_size=size)
            step_times.append(time_dense(embedding, output_weights, x, y))
        sizes.append(size)

    print(f'100.0% complete')
    return step_times, sizes