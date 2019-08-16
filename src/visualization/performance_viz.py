import matplotlib.pyplot as plt
import json
import os
performance_files_location = r'../../models/model_performances/'
performances = {}
for performance_file in os.listdir(performance_files_location):
    if performance_file.startswith('lr'):
        with open(performance_files_location + performance_file) as file:
            performance_file = performance_file.split('__')
            model = performance_file[0]
            sentence_embedding_method = performance_file[1]
            embedding = performance_file[2].replace('.json','')
            performance_dict = json.load(file)
            for key,val in performance_dict.items():
                if key in performances:
                    performances[key]['performances'].append(val)
                    performances[key]['labels'].append(model+sentence_embedding_method+embedding)
                else:
                    performances[key] = {}
                    performances[key]['performances'] = [val]
                    performances[key]['labels'] = [model+sentence_embedding_method+embedding]
print(performances.keys())

fig, axs = plt.subplots(
    len( performances.keys() ) - 2,
    1,
    sharex=True,
    sharey=False
    )

fig.set_size_inches(10,30)

for i,metric in enumerate(performances.keys()):
    if i > 1:
        axs[i-2].boxplot(performances[metric]['performances'], labels=performances['test_precision_micro']['labels'], vert=False)
        axs[i-2].set_title(list(performances.keys())[i])
plt.tight_layout()
plt.savefig('performance.pdf')

