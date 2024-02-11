import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pprint import pprint
from matplotlib import pylab
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

params = {
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 18,
    "legend.title_fontsize": 22,
    "figure.titlesize": 24,
}
pylab.rcParams.update(params)

ASSAF_ID = 207376807
DANIEL_ID = 209088723

# Q1
data_frame = pd.read_csv("virus_data.csv")
# print(data_frame)

# Q2
# print(data_frame["conversations_per_day"].value_counts().sort_index())

# Q3
# pprint(list(data_frame.keys()))
# pprint(data_frame["happiness_score"].value_counts())
# pprint(data_frame["household_income"].value_counts())
# pprint(data_frame["sport_activity"].value_counts())

# Q4
randomness = ASSAF_ID % 100 + DANIEL_ID % 100
X = data_frame.drop("spread", axis=1)
Y = data_frame["spread"]
train_set, test_set, train_labels, test_labels = train_test_split(
    X, Y, random_state=randomness, test_size=0.2
)

# ===================== part 2 ================================
# Q5
# plot = sns.pairplot(
#     train_set, vars=["PCR_01", "PCR_03"], hue="spread", plot_kws={"s": 16}, palette=["#90EE90", "#8B0000"]
# )
# for ax in np.ravel(plot.axes):
#     ax.grid(alpha=0.5)
# plot.fig.set_size_inches(12, 8)
# # plt.tight_layout()
# plot.fig.suptitle("Correlation between PCR_01 and PCR_03", y=1.01)
# plt.savefig("img.png", bbox_inches="tight")


# Q6
# corr_01_spr = data_frame[["PCR_01", "PCR_03", "spread"]].corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_01_spr, annot=True, cmap="coolwarm", annot_kws={"size": 36})
# plt.title("Correlation matrix of features")
# plt.savefig("img.png")
# plt.show()
# print(corr_01_spr)


# Q7
from sklearn.base import BaseEstimator, ClassifierMixin


# Tip: Read about scipy...cdist, np.copy, and np.argsort (or better: np.argpartition)
# Avoid using for loops, list, map, lambda, etc


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y.to_numpy()
        return self

    def predict(self, X):
        distances = cdist(X, self.X)
        distance_idx = np.argpartition(distances, self.n_neighbors, -1)[:, :self.n_neighbors]
        return np.sign(np.sum(self.y[distance_idx], axis=-1))



# Q8

train_set, test_set, train_labels, test_labels = train_test_split(
    data_frame[["PCR_01", "PCR_03"]],
    data_frame["spread"],
    random_state=randomness,
    test_size=0.2,
)
# print(len(train_set), len(test_set))
knn = kNN(n_neighbors=1)
# knn.fit(train_set, train_labels)
# predictions = knn.predict(test_set)


import visualize_clf
# visualize_clf.visualize_clf(knn, test_set, test_labels, "spread", "PCR_01", "PCR_03")
# train_score = knn.score(train_set, train_labels)
# test_score = knn.score(test_set, test_labels)
# print(train_score, test_score)

# Q9

test_set = MinMaxScaler((-1, 1)).fit_transform(test_set)
train_set = MinMaxScaler((-1, 1)).fit_transform(train_set)
knn = kNN(n_neighbors=5)
knn.fit(train_set, train_labels)
visualize_clf.visualize_clf(knn, test_set, test_labels, "normalized spread", "PCR_01", "PCR_03")
train_score = knn.score(train_set, train_labels)
test_score = knn.score(test_set, test_labels)
print(train_score, test_score)


