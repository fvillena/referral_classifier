import json
import matplotlib.pyplot as plt
import os

class GridSearchVisualizer:
    def __init__(self,results_location):
        self.scores_total = {}
        for filename in os.listdir(results_location):
            estimator_name = filename.split('.')[0]
            self.scores_total[estimator_name] = {}
            with open(results_location + filename, "r") as read_file:
                data = json.load(read_file)
                self.scores_total[estimator_name]['params'] = [str(param) for param in data[0]['params']]
                splits = int(sum(('split' in s) for s in data[0])/2)
                self.scores_total[estimator_name]['scores'] = [[data[0]['split{}_test_score'.format(i)][j]  for i in range(splits)] for j in range(len(self.scores_total[estimator_name]['params']))]
    def plot(self,figure_location):
        fig, axs = plt.subplots(
            len( self.scores_total.keys() ),
            1,
            sharex=True,
            sharey=False
            )

        fig.set_size_inches(20,len( self.scores_total.keys() ) * 10)
        for i,model in enumerate(self.scores_total.keys()):
            axs[i].boxplot(
                self.scores_total[model]['scores'],
                labels=self.scores_total[model]['params'],
                vert=False
                )
            axs[i].set_title(model)
        plt.tight_layout()
        plt.savefig(figure_location)

            