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
        bool_matrix = self.word_count_matrix.astype(bool)
        Fw = np.asarray(bool_matrix.sum(axis=0)).reshape(-1)
        Fd = bool_matrix.shape[0]
        Pw = Fw / Fd
        Fc_t = bool_matrix[self.labels["True"],:].shape[0]
        Fc_f = bool_matrix[self.labels["False"],:].shape[0]
        Pc_t = Fc_t / Fd
        Pc_f = Fc_f / Fd
        Fw_t = np.asarray(bool_matrix[self.labels["True"],:].sum(axis=0)).reshape(-1)
        Pwc_t = Fw_t / Fd
        Fw_f = np.asarray(bool_matrix[self.labels["False"],:].sum(axis=0)).reshape(-1)
        Pwc_f = Fw_f / Fd
        PMI_t = np.nan_to_num(np.log((Pwc_t) / (Pw*Pc_t)))
        PMI_f = np.nan_to_num(np.log((Pwc_f) / (Pw*Pc_f)))
        self.PMI_values = {
            "True": {word:float(PMI_t[idx]) for word,idx in self.vocab.items()},
            "False": {word:float(PMI_f[idx]) for word,idx in self.vocab.items()}
        }
        self.PMI_values["True"] = {k: v for k, v in sorted(self.PMI_values["True"].items(), reverse = True, key=lambda item: item[1]) }
        self.PMI_values["False"] = {k: v for k, v in sorted(self.PMI_values["False"].items(), reverse = True, key=lambda item: item[1]) }
        with open(pmi_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.PMI_values, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)
            