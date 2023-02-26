utilities.py

def generate_vocab(dir, min_count, max_files):

Parameters:
dir: Directory containing documents to build vocabulary from.
min_count: Minimum # of times word appears in documents processed to be included in vocabulary.

max_files: # of documents to process. Half will be from positive class and half from negative. If -1 will process all documents.

Returns:
A list of all words found in the documents excluding those that didn't appear min_count # of times.

Uses regular expression to separate words from punctuation.  

Counts up the # of times words appear in the requested documents and builds a dictionary.

Then deletes out any words that don't appear the minimum amount of times.



def create_word_vector(fname, vocab):

Parameters:
fname: The file to have feature vector made from.

vocab: A list of words in the vocabulary.

Returns:
A single numpy arrray feature vector for the file, fname, using the passed in vocabulary, vocab.



def load_data(dir, vocab, max_files):

Parameters:
dir: Directory of files you wish to be processed.

vocab: A list of words in the vocabulary.

max_files: # of documents to process. Half will be from positive class and half from negative. If -1 will process all documents.

Returns:
X, Y
Numpy array X, a set of feature vectors.  Each row a sample, each column a feature.
Numpy array Y, a set of labels corresponding to the feature vector.




ml.py:

Functions that end in _train, train and return a model based on the algorithm in the first part of the function name.

pca_transform(X,pca):
Runs PCA on the data, and returns the principle components desired.

model_test(X,model):
Uses whatever model is passed in and predicts labels based on the X data.

compute_F1:
Computes the F1 score.  Higher is better.