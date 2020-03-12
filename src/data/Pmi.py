import numpy as np
import scipy
import json
import logging
import nltk
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class NpEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



def PMI(Pwc,Pw,Pc):
    try:
        return np.log((Pwc)/(Pw*Pc))
    except ZeroDivisionError:
        return 0

class Pmi:
    def __init__(self,word_count_matrix,vocab,labels):
        self.word_count_matrix = scipy.sparse.load_npz(word_count_matrix)
        with open(vocab, 'r', encoding='utf-8') as json_file:
            self.vocab = json.load(json_file)
        self.labels = {
            "True": [],
            "False": []
        }
        with open(labels, encoding='utf-8') as file:
            for i,line in enumerate(file):
                line = line.rstrip()
                self.labels[line].append(i)
        
    def calculate(self,pmi_location):
        self.PMI_values = {
            "True": {},
            "False": {}
        }
        n = self.word_count_matrix.shape[0]
        i = 0
        sw = nltk.corpus.stopwords.words("spanish")
        for word,idx in self.vocab.items():
            if word in sw:
                continue
            #
            c_rows = self.labels["True"]
            elements = np.ix_(c_rows,[idx])
            Pwc_t = self.word_count_matrix[elements].count_nonzero() / n
            #
            c_rows = self.labels["False"]
            elements = np.ix_(c_rows,[idx])
            Pwc_f = self.word_count_matrix[elements].count_nonzero() / n
            #
            Pw = self.word_count_matrix[:,idx].count_nonzero() / n
            Pc = 0.5
            self.PMI_values["True"][word] = PMI(Pwc_t,Pw,Pc)
            self.PMI_values["False"][word] = PMI(Pwc_f,Pw,Pc)
            logger.info("{} {} - true: {}, false: {}".format(i/len(self.vocab),word, Pwc_t,Pwc_f))
            i+=1
        self.PMI_values["True"] = {k: v for k, v in sorted(self.PMI_values["True"].items(), reverse = True, key=lambda item: item[1])}
        self.PMI_values["False"] = {k: v for k, v in sorted(self.PMI_values["True"].items(), reverse = True, key=lambda item: item[1])}
        with open(pmi_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.PMI_values, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)
            