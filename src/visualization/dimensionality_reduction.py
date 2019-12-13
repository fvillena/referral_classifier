import gensim.models
import sklearn.manifold
import pandas as pd

class DimensionalityReducer:
    def __init__(self,embedding_location):
        self.embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_location, binary=False)
    
    def fit(self,reduced_embedding_location,dimensions=2):
        tisni = sklearn.manifold.TSNE(verbose=3, random_state=11, n_components=dimensions)
        self.reduced_embedding = tisni.fit_transform(self.embedding.vectors)
        self.reduced_embedding = pd.DataFrame(self.reduced_embedding,columns=['x','y'])
        self.reduced_embedding['word'] = pd.Series(list(self.embedding.vocab.keys()))
        self.reduced_embedding.to_csv(reduced_embedding_location,index=False)