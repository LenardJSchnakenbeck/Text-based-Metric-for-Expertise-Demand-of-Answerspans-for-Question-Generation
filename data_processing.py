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

#min-max-normalization
norm_df_all = pd.DataFrame(columns=RelPlofAExpDem)
for var in [Rel, PlofA, ExpDem]:
    normalized_values = df_all[var].sub(df_all[var].min(axis=1), axis=0).div(
        df_all[var].max(axis=1) - df_all[var].min(axis=1), axis=0).mul(1000)
    norm_df_all[var] = normalized_values

norm_flattened_df = flatten_df(norm_df_all, flatflat=True)
norm_final_df = create_final_df(norm_df_all)

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


def loglikelihood_ratio_chi2():
    from scipy.stats import chi2
    lr_statistic = -2 * (null_result.llf - mixedlm_result.llf)
    df_difference = mixedlm_result.df_modelwc - null_result.df_modelwc
    p_value = 1 - chi2.cdf(lr_statistic, df_difference)
    print("Likelihood Ratio Test Statistic:", lr_statistic)
    print("P-value:", p_value)
    print("df_difference:", df_difference)

Rsq_McFadden = 1 - (np.log(abs(mixedlm_result.llf)) / np.log(abs(null_result.llf)))
Rsq_CoxSnell = 1 - (null_result.llf / mixedlm_result.llf) ** (2 / mixedlm_result.nobs)

#Mixed effects assumptions
#The random effects are normally distributed with mean zero
from scipy.stats import shapiro
random_intercepts = mixedlm_result.random_effects
random_intercepts_list = [value.values[0] for value in mixedlm_result.random_effects.values()]
shapiro_stat, shapiro_p_val = shapiro(random_intercepts_list)
print("mean:", np.mean(random_intercepts_list))
print("Shapiro-Wilk Test Statistic:", shapiro_stat)
print("p-value:", shapiro_p_val)

#Within-group errors are independent with mean zero and variance σ^2.
within_group_error = mixedlm_result.scale
np.mean(within_group_error)

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


for var in [Rel,PlofA,ExpDem]:
    ranges = []
    for index in df_all.index[0:-1]:
        values = df_all[var].loc[index].to_list()
        ranges += [[min(values), max(values)]]
    df = pd.DataFrame(ranges, columns=['Min', 'Max'])
    df["Max"] = 1000-df["Max"]
    df = pd.melt(df, value_vars=["Min", "Max"],
                        var_name="Deviation", value_name="Value")
    sns.histplot(df, x="Value", hue="Deviation", binwidth=16.6666666666666, edgecolor='none')
    plt.title("Deviations from Min/Max " + var[0][:-3])
    plt.savefig(str(r"C:\Users\lenar\Documents\Masterarbeit\plots\Deviations_from_MinMax_" + var[0][:-3] + '.png'))
    plt.clf()




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




sns.boxplot(flattened_df[flattened_df["factor"] == "Rel"], x="Answerspan", y="value", color="lightgrey", width =0.4)
sns.lineplot(expert_flat[expert_flat["factor"] == "Rel"], x="Answerspan", y="value", color="brown")
plt.savefig(r"C:\ Users\lenar\Documents\Masterarbeit\plots\expert_" + "Rel" + ".png", transparent=True)
plt.clf()
sns.boxplot(flattened_df[flattened_df["factor"] == "PlofA"], x="Answerspan", y="value", color="lightgrey", width =0.4)
sns.lineplot(expert_flat[expert_flat["factor"] == "PlofA"], x="Answerspan", y="value", color="brown")
plt.savefig(r"C:\ Users\lenar\Documents\Masterarbeit\plots\expert_" + "PlofA" + ".png", transparent=True)
plt.clf()
sns.boxplot(flattened_df[flattened_df["factor"] == "ExpDem"], x="Answerspan", y="value", color="lightgrey", width =0.4)
sns.lineplot(expert_flat[expert_flat["factor"] == "ExpDem"], x="Answerspan", y="value", color="brown")
plt.savefig(r"C:\ Users\lenar\Documents\Masterarbeit\plots\expert_" + "ExpDem" + ".png", transparent=True)
plt.clf()
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

from metric_creation import apply_custom_min_max_scaling
def scale_0_1000(data):
    return apply_custom_min_max_scaling(data, 0, 1000)

metric_results = pd.DataFrame({
        'CosineSim_CosineRel': documents[0]['CosineSim_CosineRel'],
        'WordnetSim_SinglerankRel': documents[0]['WordnetSim_SinglerankRel'],
        'WordnetSim_CosineRel': documents[0]['WordnetSim_CosineRel'],
        'CosineSim_SinglerankRel': documents[0]['CosineSim_SinglerankRel'],
        "WordnetSim_Singlerank_meanedRel": [0.16385775380576473, 0.1423786132718048, 0.0, 0.44110595737491887,
                            0.14618598645630637, 0.11131071070856899, 0.14792367556203895, 0.15109098497034362,
                            0.1759149912095902, 0.25, 0.15937771403008177, 0.10758972133190861, 0.15937771403008177],
        "CosineSim_Singlerank_meanedRel": [0.07157772718945661, 0.314196068401447, 0.0, 0.44110595737491887,
                                           0.42907374750144595, 0.32939431507101263, 0.36148564737241373,
                                           0.3489175174570406, 0.1898866604968936, 0.5581822096651609,
                                           0.39264079207261154, 0.26429759607817793, 0.24802020200416192],
        'CosineRel': documents[0]['relevance_cossim'],
        "SinglerankRel": documents[0]['relevance_singlerank'],
        "SinglerankRel_meaned": [0.029143582831594822, 0.03508173096191098, 0.07444388769533512, 0.04395672141515013,
                             0.03402914015078341, 0.04367079937584677, 0.03354873660307338, 0.03267309856665751,
                             0.05012705633762287, 0.005328597433516314, 0.030382139829518113, 0.04469950841916288,
                             0.052413013762426616],
    'WordnetSim': documents[0]['similarity_wordnet'],
        'CosineSim': documents[0]['similarity_cossim']
    })


def plot_results(metric_results):
    import matplotlib.pyplot as plt
    for method in metric_results.keys():#["CosineRel", "SinglerankRel", 'WordnetSim', 'CosineSim']:
        color = sns.color_palette()[0]
        if method[-3:] == "Sim": color = sns.color_palette()[1]
        elif "_" in method: color = sns.color_palette()[2]
        sns.lineplot(pd.DataFrame(scale_0_1000(metric_results[method])), linewidth=4, palette=[color])
        plt.savefig( str(r"C:\Users\lenar\Documents\Masterarbeit\plots\metric\predictions_" + method + '.png'), transparent=True)
        plt.clf()


means_Rel = [np.mean(df_all[answerspan]) for answerspan in Rel]
means_PlofA = [np.mean(df_all[answerspan]) for answerspan in PlofA]
means_ExpDem = [np.mean(df_all[answerspan]) for answerspan in ExpDem]

#Expert Plot
expert_df_all = create_df_all(expert)
expert_final = flatten_df(create_df_all(expert), flatflat=False)
expert_flat = flatten_df(create_df_all(expert), flatflat=True)

expert_PlofA = list(expert_df_all[PlofA].iloc[0])
expert_Rel = list(expert_df_all[Rel].iloc[0])
expert_ExpDem = list(expert_df_all[ExpDem].iloc[0])




#Compare to mean

def compare_to_ref(ref_Rel, ref_PlofA, ref_ExpDem):
    def calculate_distances_to_reference(means, predictions, squared=True):
        distance_to_mean = []
        for i in range(len(predictions)):
            if squared:
                distance_to_mean += [(predictions[i] - means[i]) ** 2]
            else:
                distance_to_mean += [abs(predictions[i] - means[i])]
        return distance_to_mean

    df_evaluation = pd.DataFrame({
        #Rel
        "sqr_deviation_CosineRel":
            calculate_distances_to_reference(ref_Rel,  scale_0_1000(metric_results['CosineRel']),    squared=True),
        "sqr_deviation_SingleRankRel":
            calculate_distances_to_reference(ref_Rel,  scale_0_1000(metric_results["SinglerankRel"]),squared=True),
        "sqr_deviation_SingleRankRel_meaned":
            calculate_distances_to_reference(ref_Rel, scale_0_1000(metric_results["SinglerankRel_meaned"]), squared=True),
        "sqr_deviation_MeanRel":
            calculate_distances_to_reference(ref_Rel, [np.mean(means_Rel) for i in range(13)], squared=True),
        "sqr_deviation_ExpertRel":
            calculate_distances_to_reference(ref_Rel, expert_Rel, squared=True),
        #PlofA
        "sqr_deviation_WordnetSim":
            calculate_distances_to_reference(ref_PlofA,scale_0_1000(metric_results['WordnetSim']),   squared=True),
        "sqr_deviation_CosineSim":
            calculate_distances_to_reference(ref_PlofA,scale_0_1000(metric_results['CosineSim']),    squared=True),
        "sqr_deviation_MeanSim":
            calculate_distances_to_reference(ref_PlofA,[np.mean(means_PlofA) for i in range(13)],    squared=True),
        "sqr_deviation_ExpertSim":
            calculate_distances_to_reference(ref_PlofA, expert_PlofA, squared=True),
        #ExpDem
        "sqr_deviation_CosineSim_CosineRel":
            calculate_distances_to_reference(ref_ExpDem, scale_0_1000(metric_results['CosineSim_CosineRel']), squared=True),
        "sqr_deviation_WordnetSim_SinglerankRel":
            calculate_distances_to_reference(ref_ExpDem, scale_0_1000(metric_results["WordnetSim_SinglerankRel"]),squared=True),
        "sqr_deviation_WordnetSim_CosineRel":
            calculate_distances_to_reference(ref_ExpDem, scale_0_1000(metric_results['WordnetSim_CosineRel']), squared=True),
        "sqr_deviation_CosineSim_SinglerankRel":
            calculate_distances_to_reference(ref_ExpDem, scale_0_1000(metric_results['CosineSim_SinglerankRel']),squared=True),
        "sqr_WordnetSim_Singlerank_meanedRel":
            calculate_distances_to_reference(ref_ExpDem, scale_0_1000(metric_results['WordnetSim_Singlerank_meanedRel']),
                                        squared=True),
        "sqr_CosineSim_Singlerank_meanedRel":
            calculate_distances_to_reference(ref_ExpDem, scale_0_1000(metric_results['CosineSim_Singlerank_meanedRel']),
                                        squared=True),
        "sqr_deviation_Expert":
            calculate_distances_to_reference(ref_ExpDem, expert_ExpDem, squared=True),
        "sqr_deviation_Mean":
            calculate_distances_to_reference(ref_ExpDem, [np.mean(means_ExpDem) for i in range(13)], squared=True)
        })

    mean_dev = {
    'CosineRel': np.mean(df_evaluation["sqr_deviation_CosineRel"]),
    "SinglerankRel": np.mean(df_evaluation["sqr_deviation_SingleRankRel"]),
    "SinglerankRel_meaned": np.mean(df_evaluation["sqr_deviation_SingleRankRel_meaned"]),
    'MeanRel': np.mean(df_evaluation["sqr_deviation_MeanRel"]),
    'ExpertRel': np.mean(df_evaluation["sqr_deviation_ExpertRel"]),

    'WordnetPlofA': np.mean(df_evaluation["sqr_deviation_WordnetSim"]),
    'CosinePlofA': np.mean(df_evaluation["sqr_deviation_CosineSim"]),
    'MeanPlofA': np.mean(df_evaluation["sqr_deviation_MeanSim"]),
    'ExpertPlofA': np.mean(df_evaluation["sqr_deviation_ExpertSim"]),

    'CosinePlofA_CosineRel': np.mean(df_evaluation["sqr_deviation_CosineSim_CosineRel"]),
    "WordnetPlofA_SinglerankRel": np.mean(df_evaluation["sqr_deviation_WordnetSim_SinglerankRel"]),
    'WordnetPlofA_CosineRel': np.mean(df_evaluation["sqr_deviation_WordnetSim_CosineRel"]),
    'CosinePlofA_SinglerankRel': np.mean(df_evaluation["sqr_deviation_CosineSim_SinglerankRel"]),
    "CosineSim_Singlerank_meanedRel": np.mean(df_evaluation["sqr_CosineSim_Singlerank_meanedRel"]),
    "WordnetSim_Singlerank_meanedRel": np.mean(df_evaluation["sqr_WordnetSim_Singlerank_meanedRel"]),

    'ExpertExpDem': np.mean(df_evaluation["sqr_deviation_Expert"]),
    'MeanExpDem': np.mean(df_evaluation["sqr_deviation_Mean"])
               }
    return mean_dev

mean_dev = compare_to_ref(means_Rel, means_PlofA, means_ExpDem)
expert_dev = compare_to_ref(expert_Rel, expert_PlofA, expert_ExpDem)

def plot_deviations_bar_mean(mean_dev):
    mean_dev = pd.DataFrame(mean_dev, index=[0])
    #mean_dev[['CosineRel', "SinglerankRel", "SinglerankRel_meaned", 'MeanRel', 'ExpertRel']].plot(kind='bar', colormap="Set2")
    #mean_dev[['WordnetPlofA', 'CosinePlofA', 'MeanPlofA', 'ExpertPlofA']].plot(kind='bar', colormap="Set2")
    #mean_dev[['CosinePlofA_CosineRel', "WordnetPlofA_SinglerankRel", 'WordnetPlofA_CosineRel', 'CosinePlofA_SinglerankRel', "CosineSim_Singlerank_meanedRel", "WordnetSim_Singlerank_meanedRel", 'ExpertExpDem', 'MeanExpDem']].plot(kind='bar', colormap="Set2")
    plt.xlabel('method')
    plt.ylabel('Mean Squared Deviation')
    plt.title('Mean Deviation Comparison')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_deviations_bar_expert(expert_dev):
    expert_dev = pd.DataFrame(expert_dev, index=[0])
    #expert_dev[['CosineRel', "SinglerankRel", "SinglerankRel_meaned", 'MeanRel']].plot(kind='bar', colormap="Set2")
    #expert_dev[['WordnetPlofA', 'CosinePlofA', 'MeanPlofA']].plot(kind='bar', colormap="Set2")
    expert_dev[['CosinePlofA_CosineRel', "WordnetPlofA_SinglerankRel", 'WordnetPlofA_CosineRel', 'CosinePlofA_SinglerankRel', "CosineSim_Singlerank_meanedRel", "WordnetSim_Singlerank_meanedRel", 'MeanExpDem']].plot(kind='bar', colormap="Set2")
    plt.xlabel('method')
    plt.ylabel('Expert Squared Deviation')
    plt.title('Expert Deviation Comparison')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_deviations_PER_ANSWERSPAN_bar_mean():
    df_evaluation[["sqr_deviation_CosineRel", "sqr_deviation_SingleRankRel", "sqr_deviation_MeanRel", "sqr_deviation_ExpertRel"]].plot(kind='bar', figsize=(10, 6))
    #df_evaluation[["sqr_deviation_WordnetSim", "sqr_deviation_CosineSim"]].plot(kind='bar', figsize=(10, 6))
    #df_evaluation[["sqr_deviation_CosineRel", "sqr_deviation_SingleRankRel", "sqr_deviation_CosineSim"]].plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Answerspans')
    plt.ylabel('Squared Deviation')
    plt.title('Squared Deviation Comparison')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

#compare to expert
df_evaluation_w_expert = pd.DataFrame({
    #Rel
    "sqr_deviation_CosineRel":
        calculate_distances_to_mean(expert_Rel,  scale_0_1000(metric_results['CosineRel']),    squared=True),
    "sqr_deviation_SingleRankRel":
        calculate_distances_to_mean(expert_Rel,  scale_0_1000(metric_results["SinglerankRel"]),squared=True),
    "sqr_deviation_MeanRel":
        calculate_distances_to_mean(expert_Rel, [np.mean(means_Rel) for i in range(13)], squared=True),
    #PlofA
    "sqr_deviation_WordnetSim":
        calculate_distances_to_mean(expert_PlofA,scale_0_1000(metric_results['WordnetSim']),   squared=True),
    "sqr_deviation_CosineSim":
        calculate_distances_to_mean(expert_PlofA,scale_0_1000(metric_results['CosineSim']),    squared=True),
    "sqr_deviation_MeanSim":
        calculate_distances_to_mean(expert_PlofA,[np.mean(means_PlofA) for i in range(13)],    squared=True),
    #ExpDem
    "sqr_deviation_CosineSim_CosineRel":
        calculate_distances_to_mean(expert_ExpDem, scale_0_1000(metric_results['CosineSim_CosineRel']), squared=True),
    "sqr_deviation_WordnetSim_SinglerankRel":
        calculate_distances_to_mean(expert_ExpDem, scale_0_1000(metric_results["WordnetSim_SinglerankRel"]), squared=True),
    "sqr_deviation_WordnetSim_CosineRel":
        calculate_distances_to_mean(expert_ExpDem, scale_0_1000(metric_results['WordnetSim_CosineRel']), squared=True),
    "sqr_deviation_CosineSim_SinglerankRel":
        calculate_distances_to_mean(expert_ExpDem, scale_0_1000(metric_results['CosineSim_SinglerankRel']), squared=True),
    "sqr_deviation_Mean":
        calculate_distances_to_mean(expert_ExpDem, [np.mean(means_ExpDem) for i in range(13)], squared=True)
    })

mean_dev_expert = {
'CosineRel': np.mean(df_evaluation_w_expert["sqr_deviation_CosineRel"]),
"SinglerankRel": np.mean(df_evaluation_w_expert["sqr_deviation_SingleRankRel"]),
'MeanRel': np.mean(df_evaluation_w_expert["sqr_deviation_MeanRel"]),
'WordnetPlofA': np.mean(df_evaluation_w_expert["sqr_deviation_WordnetSim"]),
'CosinePlofA': np.mean(df_evaluation_w_expert["sqr_deviation_CosineSim"]),
'MeanPlofA': np.mean(df_evaluation_w_expert["sqr_deviation_MeanSim"]),
'CosinePlofA_CosineRel': np.mean(df_evaluation_w_expert["sqr_deviation_CosineSim_CosineRel"]),
"WordnetPlofA_SinglerankRel": np.mean(df_evaluation_w_expert["sqr_deviation_WordnetSim_SinglerankRel"]),
'WordnetPlofA_CosineRel': np.mean(df_evaluation_w_expert["sqr_deviation_WordnetSim_CosineRel"]),
'CosinePlofA_SinglerankRel': np.mean(df_evaluation_w_expert["sqr_deviation_CosineSim_SinglerankRel"]),
'MeanExpDem': np.mean(df_evaluation_w_expert["sqr_deviation_Mean"])
           }

def plot_deviations_bar_expert(mean_dev_expert):
    mean_dev_expert = pd.DataFrame(mean_dev_expert, index=[0])
    #mean_dev_expert[['CosineRel', "SinglerankRel", 'MeanRel']].plot(kind='bar', colormap="Set2")
    mean_dev_expert[['WordnetPlofA', 'CosinePlofA', 'MeanPlofA']].plot(kind='bar', colormap="Set2")
    mean_dev_expert[['CosinePlofA_CosineRel', "WordnetPlofA_SinglerankRel", 'WordnetPlofA_CosineRel', 'CosinePlofA_SinglerankRel', 'MeanExpDem']].plot(kind='bar', colormap="Set2")
    plt.xlabel('method')
    plt.ylabel('Expert Squared Deviation')
    plt.title('Expert Deviation Comparison')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()



#Normalverteilung
#from statsmodels.stats.diagnostic import kstest_normal
from scipy.stats import shapiro
def calcualte_KolomogorovSmirnov(df_values):
    start = 0
    norm_array = []
    for i in range(0,df_values.shape[0]):
        shapiro_stat, p_value = shapiro(df_values.iloc[i])
        #ks_stat, p_value = kstest_normal(df_values.iloc[i])

        #if p_value <= 0.1:
        #    norm_array += [True] #normalverteilt
        #else:
        norm_array += [round(p_value, 3)] #, ks_stat)]
    return norm_array

eval_df = pd.DataFrame()
eval_df["Rel"] = calcualte_KolomogorovSmirnov(df_all[Rel].transpose())
eval_df["PlofA"] = calcualte_KolomogorovSmirnov(df_all[PlofA].transpose())
eval_df["ExpDem"] = calcualte_KolomogorovSmirnov(df_all[ExpDem].transpose())



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