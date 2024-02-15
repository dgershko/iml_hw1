from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
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
# Q5 / Task A
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


# Q7 / Task B
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



# Q8 / Task C

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

# test_set = MinMaxScaler((-1, 1)).fit_transform(test_set)
# train_set = MinMaxScaler((-1, 1)).fit_transform(train_set)
# knn = kNN(n_neighbors=5)
# knn.fit(train_set, train_labels)
# visualize_clf.visualize_clf(knn, test_set, test_labels, "normalized spread", "PCR_01", "PCR_03")
# train_score = knn.score(train_set, train_labels)
# test_score = knn.score(test_set, test_labels)
# print(train_score, test_score)

# Q11

# samples_x = np.random.chisquare(df=2, size=100)
# samples_y = np.random.uniform(2, 5, size=100)
# samples = np.column_stack((samples_x, samples_y))
# scaler = MinMaxScaler((-1,1)).fit(samples)
# print(scaler.transform(((1, 3), (5, 3))))
# normalized_samples = MinMaxScaler((-1,1)).fit_transform(samples)
# df = pd.DataFrame(samples, columns=['x','y'])
# df_1 = pd.DataFrame(normalized_samples, columns=['x', 'y'])
# sns.scatterplot(data=df, x='x', y='y', color='red', label='regular')
# sns.scatterplot(data=df_1, x='x', y='y', color='blue', label='normalized')
# plt.show()

# Task D
data_frame_1 = data_frame.copy()
data_frame_1["SpecialProperty"] = data_frame_1["blood_type"].isin(("O+", "B+"))  # Pascal case replaces snake case :(
data_frame_1 = data_frame_1.drop("blood_type", axis=1)

# Q13

# Extracting possible symptoms and adding them as Boolean values
possible_symptoms = data_frame_1["symptoms"].unique()
possible_symptoms = set(symptom for combination in possible_symptoms if isinstance(combination, str) for symptom in combination.split(';'))
for symptom in possible_symptoms:
    data_frame_1[symptom] = data_frame_1["symptoms"].apply(lambda symptoms_str: int(symptom in symptoms_str) if isinstance(symptoms_str, str) else 0)
data_frame_1 = data_frame_1.drop("symptoms", axis=1)

# Task E
import ast


def date_to_int(date: str):
    """
    Converts a date in the format of `dd-MM-yy` to an integer.
    Accuracy is compromised for easier code (Not accounting for different number of days in month)
    """
    splitted_date = [int(n) for n in date.split("-")]
    return 375 * splitted_date[2] + 31 * splitted_date[1] + splitted_date[0]

data_frame_1["current_location_x"] = data_frame_1["current_location"].apply(lambda location: ast.literal_eval(location)[0])
data_frame_1["current_location_y"] = data_frame_1["current_location"].apply(lambda location: ast.literal_eval(location)[1])
data_frame_1["is_special_blood"] = data_frame_1["SpecialProperty"].apply(lambda is_special: int(is_special))
data_frame_1["is_male"] = data_frame_1["sex"].apply(lambda sex: int(sex == "M"))
data_frame_1["pcr_date"] = data_frame_1["pcr_date"].apply(date_to_int)

data_frame_1 = data_frame_1.drop(["patient_id", "current_location", "sex", "SpecialProperty"], axis=1)

# Q14, Q15
def generate_graphs(df, col_name):
    # COL_NAME = ['PCR_01', 'num_of_siblings']
    COLS, ROWS = (2, len(col_name))
    plt.figure(figsize=(5 * COLS, 4 * ROWS))
    for row in range(ROWS):
        column = col_name[row]
        for j, cls in enumerate(["risk", "spread"]):
            plt.subplot(ROWS,COLS, row * COLS + 1 + j)
            isContinuous = "float" in df[column].dtype.name
            sns.histplot(data=df, x=column, hue=cls, palette=["#90EE90", "#8B0000"], line_kws={"linewidth": 3},
            kde=isContinuous, multiple="layer" if isContinuous else "dodge")
            # sums = df[cls].groupby(column).sum().reset_index()
            plt.grid(alpha=0.5)
    plt.tight_layout()


# for i in data_frame_1.columns:
#     generate_graphs(data_frame_1, [i])
#     plt.savefig(f"{i}.png")
#     plt.show()

# Q16
def pairplot(df: pd.DataFrame, prediction="risk"):
    sns.pairplot(df[df.filter(like='PCR').columns.tolist() + [prediction]], plot_kws={"s": 5}, hue=prediction, palette=["#90EE90", "#8B0000"])
    plt.show()

pairplot(data_frame_1[data_frame_1['is_special_blood'] == True][["PCR_02", "PCR_06", "risk"]])
pairplot(data_frame_1[data_frame_1['is_special_blood'] == False][["PCR_02", "PCR_06", "risk"]])
    

# sns.jointplot(data=data_frame_1, x=)