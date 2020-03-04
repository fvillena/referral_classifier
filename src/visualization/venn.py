import matplotlib.pyplot as plt
import matplotlib_venn
import json
class Venn:
    def __init__(self,data_location):
        with open(data_location, "r") as read_file:
            self.data = json.load(read_file)["venn_data"]
    def plot(self,figure_location):
        matplotlib_venn.venn3_circles(
            [set(self.data["ignacio"]), set(self.data["maricella"]), set(self.data["nury"])],
            linewidth=1,
            alpha=0.2)
        matplotlib_venn.venn3(
            [set(self.data["ignacio"]), set(self.data["maricella"]), set(self.data["nury"])],
            set_labels = ('Human 1', 'Human 2', 'Human 3'),
            alpha=0.4)
        plt.title('Venn Diagram for Human Agreement')
        plt.savefig(figure_location)
        plt.close()