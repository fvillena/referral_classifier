import pandas as pd
import matplotlib.pyplot as plt

class EmbeddingCloud:
    def __init__(self, dataset_2d_location):
        self.data = pd.read_csv(dataset_2d_location)
    def plot(self, figure_location, word="diente", e=0.3):
        center = (float(self.data[self.data.word == word].x), float(self.data[self.data.word == word].y))
        x_limits = (center[0]-e,center[0]+e)
        y_limits = (center[1]-e,center[1]+e)
        zoomed_region = self.data[
            (self.data.x > x_limits[0])
            & (self.data.x < x_limits[1])
            & (self.data.y > y_limits[0])
            & (self.data.y < y_limits[1])
        ]

        fig,ax = plt.subplots(2,figsize=(5, 7))
        fig.suptitle('t-SNE Projection of Word Embeddings')
        ax[0].scatter(self.data.x,self.data.y,s=0.5, alpha=0.1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].text(-0.05, 0.02, 'a', transform=ax[0].transAxes)
        ax[1].scatter(self.data.x,self.data.y)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].text(-0.05, 0.02, 'b', transform=ax[1].transAxes)
        ax[1].set_ylim(y_limits)
        ax[1].set_xlim(x_limits)
        for _, row in zoomed_region.iterrows():
            ax[1].annotate(row['word'], (row['x']+0.01, row['y']))
        ax[0].indicate_inset_zoom(ax[1], linewidth=2, alpha=0.5,edgecolor='black')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(figure_location)