import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-01-12_12-20.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-01-12_11-36.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-01-29_18-59.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-07_11-46.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-13_11-54.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-19_17-13.csv", encoding="utf-16")
#df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-21_10-35.csv", encoding="utf-16")
df = pd.read_csv("C:/Users/lenar/Downloads/data_metric-expertisedemand_2024-02-29_18-58.csv", encoding="utf-16")

#df = pd.read_csv("/Users/lenard/Downloads/data_metric-expertisedemand_2024-02-21_10-39.csv", encoding="utf-16")

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
Rel = ['Rel_01', 'Rel_02', 'Rel_03', 'Rel_04', 'Rel_05', 'Rel_06', 'Rel_07', 'Rel_08', 'Rel_09', 'Rel_10', 'Rel_11',
       'Rel_12', 'Rel_13']
PlofA = ['PlofA_01', 'PlofA_02', 'PlofA_03', 'PlofA_04', 'PlofA_05', 'PlofA_06', 'PlofA_07', 'PlofA_08', 'PlofA_09',
         'PlofA_10', 'PlofA_11', 'PlofA_12', 'PlofA_13']
ExpDem = ["ExpDem_01", "ExpDem_02", "ExpDem_03", "ExpDem_04", "ExpDem_05", "ExpDem_06", "ExpDem_07", "ExpDem_08",
         "ExpDem_09", "ExpDem_10", "ExpDem_11", "ExpDem_12", "ExpDem_13"]
#RelPlofAExpDem = Rel+PlofA+ExpDem
RelPlofAExpDem = sorted(list(Rel+PlofA+ExpDem), key=lambda x: x[-2:])
answerspans = ["Earthworms", "the ability to regenerate lost segments", "species", "the extent of the damage",
                   "Stephenson", "a chapter of his monograph", "C.E. Gates", "20 years",
                   "regeneration in a variety of species", "Gates", "two whole worms", "a bisected specimen",
                   "certain species"]

expert = df[df["CASE"]==399]
df = df[df["CASE"]!=399]
df_raw = df
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
        return answerspans[int(match.group(2))-1]

def create_df_all(df):
    #df_all = pd.DataFrame(df_Rel.iloc[0:1])
    #df_all = pd.concat([df_all, df_Plofa.iloc[0:1], df_ExpDem.iloc[0:1]])
    df_Rel = df[Rel].transpose()
    df_Plofa = df[PlofA].transpose()
    df_ExpDem = df[ExpDem].transpose()
    df_all = pd.DataFrame()
    for i in range(df_Rel.shape[0]):
         df_all = pd.concat([df_all, df_Rel.iloc[i], df_Plofa.iloc[i], df_ExpDem.iloc[i]], axis=1)
    return df_all

df_all = create_df_all(df)

#Outlier
def outlier_LOF(df_all):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor()
    outlier_Rel = clf.fit_predict(df_all[Rel])
    outlier_PlofA = clf.fit_predict(df_all[PlofA])
    outlier_ExpDem = clf.fit_predict(df_all[ExpDem])
    df_all["outlier_LOF"] = [-1 if -1 in outliers else 1 for outliers in
                         zip(outlier_Rel,outlier_PlofA,outlier_ExpDem)]
    return df_all #del df_all["outlier_LOF"]

def ABOD_outlier_processing():
    from pyod.models.abod import ABOD
    model = ABOD(contamination=0.14, method='default', n_neighbors=10)
    for factor in [Rel, PlofA, ExpDem]:
        model.fit(df_all[factor])
        outlier = model.predict(df_all[factor])
        df_all["outlier"+factor[0][:-3]] = outlier

    for con in [0.14,0.22]:
        model = ABOD(contamination=con, method='default', n_neighbors=10)
        model.fit(df_all[RelPlofAExpDem])
        labels = model.predict(df_all[RelPlofAExpDem])
        df_all[str(con)+"outlier_ABOD"] = labels
        print(sum(labels), labels)


from pyod.models.abod import ABOD
model = ABOD(contamination=0.14, method='default', n_neighbors=10)
model.fit(df_all[RelPlofAExpDem])
labels = model.predict(df_all[RelPlofAExpDem])
df_all["outlier_ABOD"] = labels


df_outlier = df_all
df_outliersonly = df_all[df_all["outlier_ABOD"] == 1]
df_all = df_all[df_all["outlier_ABOD"] != 1][Rel+PlofA+ExpDem]


def create_final_df(df_all):
    long_df = df_all.transpose().stack().rename_axis(['letter', 'Participant']).reset_index()
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
        index = long_df[["Participant"]][start:start + participants_count].values.tolist()
        index = [i[0] for i in index]
        word = long_df[["word"]][start:start + participants_count].values.tolist()
        word = [i[0] for i in word]
        add_df = pd.DataFrame(
            {"word": word, "Participant": index,
             "Rel": list(Rel), "PlofA": list(PlofA), "ExpDem": list(ExpDem)})
        # add_df = pd.DataFrame({"index": long_df[["index"]][start:start+offset], "word": long_df[["word"]][start:start+offset],"Rel": Rel, "PlofA": PlofA, "ExpDem": ExpDem})
        # pd.DataFrame([long_df[["index","word"]][start:start+offset], pd.DataFrame([[Rel, PlofA, ExpDem]])])
        final_df = pd.concat([final_df, add_df])

        start += offset
    return final_df

final_df = create_final_df(df_all[RelPlofAExpDem])

def flatten_df(df_all, columns = [Rel,PlofA,ExpDem], flatflat = False): ##############HLPRPLRPR
    #unnötiger scheiß
    [Rel, PlofA, ExpDem] = columns
    flattened_df = pd.DataFrame()
    flattened_df["Rel"] = df_all[Rel].melt(value_name="value")["value"]
    flattened_df["PlofA"] = df_all[PlofA].melt(value_name="value")["value"]
    flattened_df["ExpDem"] = df_all[ExpDem].melt(value_name="value")["value"]
    flattened_df["Answerspan"] = df_all[Rel].melt(var_name="Answerspan")["Answerspan"]
    flattened_df["Answerspan"] = flattened_df["Answerspan"].str[-2:].astype(int)
    #flattened_df["Person"] = df_all.index * flattened_df.shape[0]/len(df_all.index)
    flattened_df["Person"] = [df_all.index[i % len(df_all.index)] for i in range(len(flattened_df))]
    if not flatflat: return flattened_df
    Rel = flattened_df[["Rel","Answerspan","Person"]].rename(columns={"Rel": "value"})
    Rel["factor"] = ["Rel"] * flattened_df.shape[0]
    PlofA = flattened_df[["PlofA","Answerspan","Person"]].rename(columns={"PlofA": "value"})
    PlofA["factor"] = ["PlofA"] * flattened_df.shape[0]
    ExpDem = flattened_df[["ExpDem", "Answerspan", "Person"]].rename(columns={"ExpDem": "value"})
    ExpDem["factor"] = ["ExpDem"] * flattened_df.shape[0]
    return pd.concat([Rel, PlofA, ExpDem], ignore_index=True)

flattened_df = flatten_df(df_all, columns = [Rel,PlofA,ExpDem], flatflat= True)

def likert_scores(df, df_all):
    #1 = stimme voll und ganz zu #5 = stimme überhaupt nicht zu
    likert_items = df.loc[df_all.index][["Rel_likert", "PlofA_likert"]]

    likert_items["Rel_likert"].mean() #2.142
    sns.countplot(x=likert_items["Rel_likert"], color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
    plt.xticks(ticks=range(5), labels=["Completely Agree", "Agree", "Neither agree,\n nor disagree","Disagree", "Completely Disagree"])

    likert_items["PlofA_likert"].mean() #2.232
    sns.countplot(x=likert_items["PlofA_likert"], color=(1.0, 0.4980392156862745, 0.054901960784313725))
    plt.xticks(ticks=range(5), labels=["Completely Agree", "Agree", "Neither agree,\n nor disagree","Disagree", "Completely Disagree"])

#simple linear multiple regression
def simple_linear_multiple_regression():
    from statsmodels.formula.api import ols
    multireg = ols("ExpDem ~ Rel + PlofA", final_df)
    results = multireg.fit()
    print(results.summary())


#Mixed effects LM
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2

null_model_formula = "ExpDem ~ 1"
null_model = MixedLM.from_formula(null_model_formula, groups="word", data=final_df)
null_result = null_model.fit(reml = False)
#print(null_result.summary())

#Likelihood comparison
print("null_result.llf:", null_result.llf)
for model_formula in ["ExpDem ~ 1", "ExpDem ~ Rel", "ExpDem ~ PlofA", "ExpDem ~ Rel + PlofA"]:
    #model_formula = "ExpDem ~ Rel + PlofA"
    mixedlm_model = MixedLM.from_formula(model_formula, groups="word", data=final_df)
    mixedlm_result = mixedlm_model.fit(reml = False)
    #print(mixedlm_result.summary())
    #print(model_formula)

    lr_statistic = -2 * (null_result.llf - mixedlm_result.llf)
    df_difference = mixedlm_result.df_modelwc - null_result.df_modelwc
    p_value = 1 - chi2.cdf(lr_statistic, df_difference)
    print("\n\n",model_formula, "\n","-"*20)
    print("Likelihood Ratio Test Statistic:", lr_statistic)
    print("P-value:", p_value)
    print("df_difference:", df_difference)
    #print("null_result.llf:", null_result.llf)
    print("mixedlm_result.llf:", mixedlm_result.llf)
    print("bic:", mixedlm_result.bic)

#Mixed effects assumptions
random_intercepts = mixedlm_result.random_effects
random_intercepts_list = [value.values[0] for value in mixedlm_result.random_effects.values()]
shapiro_stat, shapiro_p_val = shapiro(random_intercepts_list)
print("mean:", np.mean(random_intercepts_list))
print("Shapiro-Wilk Test Statistic:", shapiro_stat)
print("p-value:", shapiro_p_val)

from scipy.stats import pearsonr
fixed_effects = mixedlm_result.fe_params
correlations = {}
for random_effect, intercepts in enumerate(random_intercepts_list):
    random_effect += 1
    correlation, p_value = pearsonr(intercepts, final_df[random_effect])
    correlations[random_effect] = (correlation, p_value)

# Print correlations
for random_effect, (correlation, p_value) in correlations.items():
    print(f"Correlation between {random_effect} and fixed effects: {correlation}, p-value: {p_value}"

# Kollinearität
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = variance_inflation_factor(final_df[["Rel", "PlofA"]].values, 0)

# heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
lm, lm_pvalue, fval, breuschfpval = het_breuschpagan(mixedlm_result.resid, mixedlm_model.exog, robust=True)
print("Lagrange multiplier statistic:",lm, lm_pvalue)
print("Breusch F and p:",fval, breuschfpval)

# normality of residuals
from scipy.stats import shapiro
shapiro_stat, shapiro_p_val = shapiro(mixedlm_result.resid)
print("shapiro Wilk normal residuals:", shapiro_stat, shapiro_p_val)
#sns.histplot(mixedlm_result.resid, bins=20)

# independence of error
residuals_mean = np.mean(mixedlm_result.resid)

#Krippendorff Alpha
import krippendorff
alpha_Rel = krippendorff.alpha(reliability_data=df_all[Rel], level_of_measurement="ratio")
alpha_PlofA = krippendorff.alpha(reliability_data=df_all[PlofA], level_of_measurement="ratio")
alpha_ExpDem = krippendorff.alpha(reliability_data=df_all[ExpDem], level_of_measurement="ratio")

"""
#Plotten
def plot_list(values_list):
    df = pd.DataFrame({'value': values_list,
                       'index': [i for i in range(len(values_list))]
                       })
    df.plot(kind='bar', x='index', y='value')

def plot_distribution(column_name):
    df[column_name].hist()
    plt.title(get_answerspan_text(column_name))
    plt.show()

def plot_distribution_sns(column_name, i):
    plt.clf()
    sns.histplot(df, x=column_name, binwidth=50, kde=True)

    #title = str(get_answerspan_text(column_name) + "(normal distributed: " +  eval_df[column_name[:-3]][i] + ")")
    plt.title(get_answerspan_text(column_name))
    plt.savefig( str(r"C:\ Users\lenar\Documents\Masterarbeit\plots\Hist_" + column_name + '.png'))
    #plt.show()

for i, column_name in enumerate(RelPlofAExpDem):
    plot_distribution_sns(column_name, i)


#Violinplot-Overview
#sns.violinplot(final_df, x="PlofA", y="word", orient="y", fill=False) #verdeckt
#flattened_dff = flatten_df(df_all, columns = [Rel,PlofA,ExpDem], flatflat = True)
#sns.violinplot(flattened_dff, x="Answerspan", y="value", hue="factor") #nebeneinander

#Overview
#sns.boxplot(flattened_dff, x="Answerspan", y="value", hue="factor")
#sns.violinplot(flattened_df, x="Answerspan", y="value", hue="factor")
###sns.lineplot(flattened_df, x="Answerspan", y="value", hue="factor")

#Multiple Regression viz
r = flattened_df[flattened_df["factor"] == "Rel"]
e = flattened_df[flattened_df["factor"] == "ExpDem"]["value"]

##OUTLIER

#outlier-trash
#outlier = flatten_df(df_outlier[df_outlier["outlier_LOF"] == -1], columns=[Rel,PlofA,ExpDem], flatflat=True)
#sns.lineplot(outlier, x="Answerspan", y="value", hue="factor", style="Person")

#Outlier Rel
outlier_Rel = flatten_df( df_outlier[df_outlier["outlier_Rel"] == -1], columns=[Rel,PlofA,ExpDem], flatflat=True)
sns.boxplot(flattened_df[flattened_df["factor"] == "Rel"], x="Answerspan", y="value", color="lightgrey", width =0.4)
#plt.clf()
sns.lineplot(outlier_Rel[outlier_Rel["factor"] == "Rel"], x="Answerspan", y="value", hue="Person", palette="tab10", linewidth=2) #, style="Person"

#Outlier ExpDem
outlier_ExpDem = flatten_df( df_outlier[df_outlier["outlier_ExpDem"] == -1], columns=[Rel,PlofA,ExpDem], flatflat=True)
sns.boxplot(flattened_df[flattened_df["factor"] == "ExpDem"], x="Answerspan", y="value", color="lightgrey", width =0.4)
#plt.clf()
sns.lineplot(outlier_ExpDem[outlier_ExpDem["factor"] == "ExpDem"], x="Answerspan", y="value", hue="Person", palette="tab10", linewidth=4)


#Scatter Plot
regression_plot_df = flattened_df[flattened_df['factor'].isin(['Rel', 'PlofA'])][['Answerspan', 'Person', 'factor', 'value']]
exp_dem_values = flattened_df[flattened_df['factor'] == 'ExpDem']['value'].tolist() * 2  # Duplicate for redundancy
regression_plot_df['ExpDem'] = exp_dem_values
regression_plot_df = regression_plot_df.rename(columns={'value': 'Rel / PlofA'})
sns.scatterplot(regression_plot_df, x="ExpDem", y="Rel / PlofA", hue="factor")

#Expert Plot
sns.boxplot(flattened_df[flattened_df["factor"] == "Rel"], x="Answerspan", y="value", color="lightgrey", width =0.4)
sns.lineplot(expertflat[expertflat["factor"] == "Rel"], x="Answerspan", y="value", color="red")
sns.boxplot(flattened_df[flattened_df["factor"] == "PlofA"], x="Answerspan", y="value", color="lightgrey", width =0.4)
sns.lineplot(expertflat[expertflat["factor"] == "PlofA"], x="Answerspan", y="value", color="red")
sns.boxplot(flattened_df[flattened_df["factor"] == "ExpDem"], x="Answerspan", y="value", color="lightgrey", width =0.4)
sns.lineplot(expertflat[expertflat["factor"] == "ExpDem"], x="Answerspan", y="value", color="red")
"""

def ppplot(results_resid):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    # Generate a P-P plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    stats.probplot(results.resid, dist="norm", plot=ax)
    ax.set_title('P-P Plot of Residuals')
    ax.get_lines()[1].set_linestyle('--')  # Add a dashed line for reference
    plt.show()

def plot_outlier(df_all):
    flattened_df = flatten_df(df_all, flatflat=True)
    df_outliersonly = df_all[df_all["outlier_ABOD"] == 1]
    outflat = flatten_df(df_outliersonly, columns=[Rel, PlofA, ExpDem], flatflat=True)
    for factor in ["Rel", "PlofA", "ExpDem"]:
        sns.boxplot(flattened_df[flattened_df["factor"] == factor],
                    x="Answerspan", y="value", color="lightgrey", width=0.4)
        plt.savefig(r"C:\Users\lenar\Documents\Masterarbeit\plots\Boxplot_" + factor + ".png")
        plt.clf()
        sns.lineplot(
            outflat[(outflat["factor"] == factor) &
                    (outflat["Person"].isin(list(df_outliersonly[df_outliersonly["outlier" + factor] == 1].index)))],
            x="Answerspan", y="value", hue="Person", palette="tab10", linewidth=2)  # , style="Person"
        plt.savefig(r"C:\Users\lenar\Documents\Masterarbeit\plots\Outlier_" + factor + ".png", transparent=True)
        plt.clf()


#####################compare to data

from metric_creation import load_final_documents, apply_min_max_normalization
from main import final_documents_path
documents = load_final_documents(final_documents_path)
metric_results = pd.DataFrame({
        'CosineSim_CosineRel': documents[0]['CosineSim_CosineRel'],
        'WordnetSim_SinglerankRel': documents[0]['WordnetSim_SinglerankRel'],
        'WordnetSim_CosineRel': documents[0]['WordnetSim_CosineRel'],
        'CosineSim_SinglerankRel': documents[0]['CosineSim_SinglerankRel'],
        'CosineRel': documents[0]['relevance_cossim'],
        "SinglerankRel": documents[0]['relevance_singlerank'],
        'WordnetSim': documents[0]['similarity_wordnet'],
        'CosineSim': documents[0]['similarity_cossim']
    })

#Normalverteilung
from statsmodels.stats.diagnostic import kstest_normal
def calcualte_KolomogorovSmirnov(df_values):
    start = 0
    norm_array = []
    for i in range(0,df_values.shape[0]):
        ks_stat, p_value = kstest_normal(df_values.iloc[i])
        #print("Kolmogorov-Smirnov statistic:", ks_stat)
        #print("p-value:", p_value)
        if p_value <= 0.1:
            norm_array += [True] #normalverteilt
        else:
            norm_array += [(p_value, ks_stat)]
    return norm_array

eval_df = pd.DataFrame()
eval_df["Rel"] = calcualte_KolomogorovSmirnov(df_all[Rel].transpose())
eval_df["PlofA"] = calcualte_KolomogorovSmirnov(df_all[PlofA].transpose())
eval_df["ExpDem"] = calcualte_KolomogorovSmirnov(df_all[ExpDem].transpose())

from scipy import stats

def calculate_t_test(annotation_series, metric_prediction_value):
    tstat, pval = stats.ttest_1samp(annotation_series, metric_prediction_value, alternative='two-sided')
    return pval

import numpy as np
from scipy import stats

def calculate_belonging_via_confidence_interval(annotation_series, metric_prediction_value, confidence_level=0.95):
  mean = annotation_series.mean()
  stddev = annotation_series.std()

  # Calculate the critical z-score for the desired confidence level.
  critical_z_score = stats.norm.ppf(confidence_level)

  lower_bound = mean - critical_z_score * stddev
  upper_bound = mean + critical_z_score * stddev
  return lower_bound < metric_prediction_value < upper_bound


def create_z_test_df(metric_version='CosineSim_SinglerankRel'):
    df_t_test_results = pd.DataFrame()
    for i, column in enumerate([Rel, PlofA, ExpDem]):
        print(column)
        t_test_results = []
        for j, column in enumerate(column):
            t_test_results += [
                calculate_belonging_via_confidence_interval(df_all[column], metric_results[metric_version][j])]
        df_t_test_results[column[:-3]] = pd.Series(t_test_results)
    print(df_t_test_results)
    return df_t_test_results


###EXPERT
"""
df_compare = final_df[:]
df_compare["ExpDem"] = [x//800 for x in df_compare["ExpDem"]]
sns.lineplot(df_compare, x="word", y="ExpDem")
sns.lineplot(pd.DataFrame({
    'expert': list(map(lambda x: x / 1000, (expertflat[expertflat["factor"]=="ExpDem"]["value"]))),
    'CosineSim_CosineRel': documents[0]['CosineSim_CosineRel'],
    'CosineSim_SinglerankRel': documents[0]['CosineSim_SinglerankRel']}))

sns.lineplot(pd.DataFrame({
    'expert': list(map(lambda x: x / 1000, (expertflat[expertflat["factor"]=="Rel"]["value"]))),
    'CosineRel': documents[0]['relevance_cossim'],
    'SinglerankRel': documents[0]['relevance_singlerank']}))

sns.lineplot(pd.DataFrame({
    'expert': list(map(lambda x: x / 1000, (expertflat[expertflat["factor"]=="PlofA"]["value"]))),
    'CosineSim': documents[0]['similarity_cossim'],
    'WordnetSim': documents[0]['similarity_wordnet']}))
"""


#Später Feedback
"""
Outlier berichten
Multiple Regression berichten
Krippendorffs Alpha berichten
Boxplotes / Histogramme / Normalverteilung? (Anhang) (sehr einige/sehr uneinig)


calculate automatic values 
calculate correlation
"""