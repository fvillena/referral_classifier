import json
import matplotlib.pyplot as plt
import os

class RocCurve:
    def __init__(self,data_location, discrete_measures_location = False):
        with open(data_location, "r") as read_file:
            data = json.load(read_file)
        self.roc_curve = data['roc_curve']
        self.auc = data['roc_auc_score']
        if discrete_measures_location:
            with open(discrete_measures_location, "r") as read_file:
                self.discrete_measures = json.load(read_file)
    def plot(self,figure_location, discrete_measures = False):
        plt.plot(self.roc_curve[0],self.roc_curve[1])
        plt.fill_between(self.roc_curve[0],self.roc_curve[1],color='aliceblue',label='Area under the curve = {:.3f}'.format(self.auc))
        if discrete_measures:
            for _,metrics in self.discrete_measures.items():
                plt.plot(metrics["FPR"],metrics["TPR"], color="darkorange", marker="o", label="Human", ls="")
        plt.title('Receiver Operating Characteristics Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.legend()
        plt.tight_layout()
        plt.savefig(figure_location)
        plt.close()