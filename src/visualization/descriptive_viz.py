import matplotlib.pyplot as plt
import pandas as pd


class DescriptiveVisualizer:
    def __init__(self, dataset_location):
        self.data = pd.read_csv(dataset_location, parse_dates=True)

    def plot(
        self,
        figure_location,
        column_name,
        title,
        xlabel,
        ylabel,
        min_quantile=0.025,
        max_quantile=0.975,
    ):
        try:
            min_data = self.data[column_name].quantile(min_quantile)
            max_data = self.data[column_name].quantile(max_quantile)
            data_clipped = self.data[column_name].clip(min_data,max_data)
        except TypeError:
            self.data[column_name] = pd.to_datetime(self.data[column_name])
            min_data = self.data[column_name].quantile(min_quantile)
            max_data = self.data[column_name].quantile(max_quantile)
            data_clipped = self.data[column_name].clip(min_data,max_data)
        plt.hist(data_clipped, edgecolor="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(figure_location)
