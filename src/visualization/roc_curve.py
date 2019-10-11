import json
import matplotlib.pyplot as plt
import os

class RocCurve:
    def __init__(self,data_location):
        with open(data_location, "r") as read_file:
            data = json.load(read_file)
        self.roc_curve = data['roc_curve']
        self.auc = data['roc_auc_score']
    def plot(self,figure_location):
        plt.plot(self.roc_curve[0],self.roc_curve[1])
        plt.fill_between(self.roc_curve[0],self.roc_curve[1],color='aliceblue',label='Area under the curve = {:.3f}'.format(self.auc))
        plt.title('Receiver Operating Characteristics Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_location)