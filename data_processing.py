import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-01-12_12-20.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-01-12_11-36.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-01-29_18-59.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-07_11-46.csv", encoding="utf-16")
df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-13_11-54.csv", encoding="utf-16")

df.rename(columns={
    "A104": "consent",
    "A202": "metric_understanding",
    "A204": "lang_understanding",
    "A601": "otherFactors",
    "A602": "additionals",

    "A301_01": "ExpDem_01", "A301_02": "ExpDem_02", "A301_03": "ExpDem_03", "A301_04": "ExpDem_04",
    "A301_05": "ExpDem_05", "A301_06": "ExpDem_06", "A301_07": "ExpDem_07", "A301_08": "ExpDem_08",
    "A301_09": "ExpDem_09", "A301_10": "ExpDem_10", "A301_11": "ExpDem_11", "A301_12": "ExpDem_12",
    "A301_13": "ExpDem_13", "A401_01": "Rel_01",
    "A401_02": "Rel_02", "A401_03": "Rel_03", "A401_04": "Rel_04", "A401_05": "Rel_05", "A401_06": "Rel_06",
    "A401_07": "Rel_07", "A401_08": "Rel_08", "A401_09": "Rel_09", "A401_10": "Rel_10", "A401_11": "Rel_11",
    "A401_12": "Rel_12", "A401_13": "Rel_13", "A402": "Rel_likert", "A507_01": "PlofA_01",
    "A507_02": "PlofA_02", "A507_03": "PlofA_03", "A507_04": "PlofA_04", "A507_05": "PlofA_05", "A507_06": "PlofA_06",
    "A507_07": "PlofA_07", "A507_08": "PlofA_08", "A507_09": "PlofA_09", "A507_10": "PlofA_10", "A507_11": "PlofA_11",
    "A507_12": "PlofA_12", "A507_13": "PlofA_13", "A502": "PlofA_likert"
}, inplace=True)

df = df[(df.FINISHED != 0) & (df.MISSING <= 10)]

def get_answerspan_text(input_string):
    answerspans = ["Earthworms", "the ability to regenerate lost segments", "species", "the extent of the damage",
                   "Stephenson", "a chapter of his monograph", "C.E. Gates", "20 years",
                   "regeneration in a variety of species", "Gates", "two whole worms", "a bisected specimen",
                   "certain species"]

    match = re.match(r"(ExpDem|Rel|PlofA)_(\d{2})", input_string)
    if not match:
        return -1
    else:
        return answerspans[int(match.group(2))]


Rel = ['Rel_01', 'Rel_02', 'Rel_03', 'Rel_04', 'Rel_05', 'Rel_06', 'Rel_07', 'Rel_08', 'Rel_09', 'Rel_10', 'Rel_11',
       'Rel_12', 'Rel_13']
PlofA = ['PlofA_01', 'PlofA_02', 'PlofA_03', 'PlofA_04', 'PlofA_05', 'PlofA_06', 'PlofA_07', 'PlofA_08', 'PlofA_09',
         'PlofA_10', 'PlofA_11', 'PlofA_12', 'PlofA_13']
ExpDem = ["ExpDem_01", "ExpDem_02", "ExpDem_03", "ExpDem_04", "ExpDem_05", "ExpDem_06", "ExpDem_07", "ExpDem_08",
         "ExpDem_09", "ExpDem_10", "ExpDem_11", "ExpDem_12", "ExpDem_13"]

df_Rel = df[Rel].transpose()
df_Plofa = df[PlofA].transpose()
df_ExpDem = df[ExpDem].transpose()

def create_df_all(df_Rel, df_Plofa, df_ExpDem):
    #df_all = pd.DataFrame(df_Rel.iloc[0:1])
    #df_all = pd.concat([df_all, df_Plofa.iloc[0:1], df_ExpDem.iloc[0:1]])
    df_all = pd.DataFrame()
    for i in range(df_Rel.shape[0]):
         df_all = pd.concat([df_all, df_Rel.iloc[i], df_Plofa.iloc[i], df_ExpDem.iloc[i]], axis=1)
    return df_all

df_all = create_df_all(df_Rel, df_Plofa, df_ExpDem)

from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor()
#clf.fit_predict(x)
df_all["outlier_Rel"] = clf.fit_predict(df_all[Rel])
df_all["outlier_PlofA"] = clf.fit_predict(df_all[PlofA])
df_all["outlier_ExpDem"] = clf.fit_predict(df_all[ExpDem])
df_all["outlier_LOF"] = [-1 if -1 in outliers else 1 for outliers in
                     zip(df_all["outlier_Rel"],df_all["outlier_PlofA"],df_all["outlier_ExpDem"])]

df_all = df_all[df_all["outlier_LOF"] != -1]

#df_all["outlier_Rel"] = clf.fit_predict(df_all.transpose().iloc[::3])
#df_all["outlier_PlofA"] = clf.fit_predict(df_all.transpose().iloc[1::3])
#df_all["outlier_ExpDem"] = clf.fit_predict(df_all.transpose().iloc[2::3])

#df_a = df_all.transpose()

def create_final_df(df_all):
    long_df = df_all.transpose().stack().rename_axis(['letter', 'index']).reset_index()
    long_df['word'] = long_df['letter'].str[-2:]
    long_df['word'] = long_df['word'].astype(int)
    start = 0
    participants_count = long_df['letter'].value_counts()['Rel_01']
    offset = participants_count * 3
    final_df = pd.DataFrame()
    for i in range(long_df.shape[0] // offset):
        # for j in range(23):
        Rel = long_df[0][start:start + participants_count].values.tolist()
        PlofA = long_df[0][start + participants_count:start + participants_count * 2].values.tolist()
        ExpDem = long_df[0][start + participants_count * 2:start + offset].where(
            long_df["letter"].str.startswith("ExpDem")).values.tolist() #muss das sein??
        index = long_df[["index"]][start:start + participants_count].values.tolist()
        index = [i[0] for i in index]
        word = long_df[["word"]][start:start + participants_count].values.tolist()
        word = [i[0] for i in word]
        add_df = pd.DataFrame(
            {"word": word, "index": index,
             "Rel": list(Rel), "PlofA": list(PlofA), "ExpDem": list(ExpDem)})
        # add_df = pd.DataFrame({"index": long_df[["index"]][start:start+offset], "word": long_df[["word"]][start:start+offset],"Rel": Rel, "PlofA": PlofA, "ExpDem": ExpDem})
        # pd.DataFrame([long_df[["index","word"]][start:start+offset], pd.DataFrame([[Rel, PlofA, ExpDem]])])
        final_df = pd.concat([final_df, add_df])

        start += offset
    return final_df

RelPlofAExpDem = sorted(list(Rel+PlofA+ExpDem), key=lambda x: x[-2:])
#df_all ohne outlier-columns
final_df = create_final_df(df_all[RelPlofAExpDem])

import statsmodels.api as sm

#Kollinearität
from statsmodels.formula.api import ols
reg = ols("Rel ~ PlofA", final_df).fit()
print(reg.summary())

#Grouped_multiple regression
import statsmodels.formula.api as smf
multireg_grouped = smf.mixedlm("ExpDem ~ Rel + PlofA", final_df, groups=final_df["word"]).fit()
print(multireg_grouped.summary())

#multiple regression
from statsmodels.formula.api import ols
multireg = ols("ExpDem ~ Rel + PlofA", final_df).fit()
print(multireg.summary())



import krippendorff
krippendorff.alpha(value_counts=df_all[ExpDem])
#0.03605281993487608
krippendorff.alpha(value_counts=df_all[PlofA])
#0.020112634064339607
krippendorff.alpha(value_counts=df_all[Rel])
#0.025592442366210277


from statsmodels.stats.diagnostic import kstest_normal
def calcualte_KolomogorovSmirnov(df_values):
    start = 0
    norm_array = []
    for i in range(0,df_values.shape[0]):
        ks_stat, p_value = kstest_normal(df_values.iloc[i])
        #print("Kolmogorov-Smirnov statistic:", ks_stat)
        #print("p-value:", p_value)
        if p_value > 0.05:
            norm_array += [True]
        else:
            norm_array += [False]
    return norm_array

eval_df = pd.DataFrame()
#calcualte_KolomogorovSmirnov(df_a.iloc[::3])
eval_df["norm_Rel"] = calcualte_KolomogorovSmirnov(df_all[RelPlofAExpDem].transpose().iloc[::3])
eval_df["norm_PlofA"] = calcualte_KolomogorovSmirnov(df_all[RelPlofAExpDem].transpose().iloc[1::3])
eval_df["norm_ExpDem"] = calcualte_KolomogorovSmirnov(df_all[RelPlofAExpDem].transpose().iloc[2::3])


def plot_distribution(column_name):
    df[column_name].hist()
    plt.title(get_answerspan_text(column_name))
    plt.show()

def plot_distribution_sns(column_name):
    sns.histplot(df, x=column_name, binwidth=50, kde=True)
    plt.title(get_answerspan_text(column_name))
    plt.show()

for i in ["ExpDem_01", "ExpDem_02", "ExpDem_03", "ExpDem_04", "ExpDem_05", "ExpDem_06", "ExpDem_07", "ExpDem_08",
          "ExpDem_09", "ExpDem_10", "ExpDem_11", "ExpDem_12", "ExpDem_13"]:
    plot_distribution_sns(i)