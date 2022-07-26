"""
Graph of different Experiments (Average Returns or QF loss)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def create_graph(env_name, csv_file, num_epochs, metrics):
    data = pd.read_csv(csv_file, index_col=0)
    # Index 8 for average returns
    # Index 10 for std returns
    # Index 50 for QF Loss

    avg_returns = data.iloc[:, metrics].values
    epochs = np.arange(num_epochs)

    plt.plot(epochs, avg_returns, linestyle='--', color='m')
    plt.xlabel('Epochs')

    if metrics == 8:
        metrics_name = 'Average Returns'
    elif metrics == 50:
        metrics_name = 'QF Loss'

    plt.ylabel(metrics_name)
    plt.title(f'{metrics_name} for {env_name} with DQN ({num_epochs} epochs)')
    plt.show()

if __name__ == "__main__":
    env_name, csv_file, num_epochs, metrics = sys.argv[1:]
    create_graph(env_name, csv_file, int(num_epochs), int(metrics))