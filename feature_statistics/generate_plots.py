import matplotlib.pyplot as plt
import seaborn as sns
from config import WORK_SPACE, SEG_DIR
import numpy as np
import pandas as pd
import os
import glob

"""
questions:
# distribution of sequence lengths & sequence time span
# number of sequences with diff diagnosis
# number of sequences with injections
# distribution of time until dry
# distribution of 3 and 6 month treatment effect

# possible questions for doctors:

1. average time for injection to reduce fluid?
2. when are injections administrated?
3. are there other treatments next to injections that we should monitor?
4. how much does fluid vary naturally and how much are due to injections?
5. conversion from normal to wet AMD from features?

todo:

have time until dry start after first injection
"""


def n_of_visits(table, variable):
    injections_bool = table.number_of_injections > 0
    sns.distplot(table[f"number_of_{variable}"], kde = False, label = f'number of {variable} - all')
    sns.distplot(table.loc[injections_bool][f"number_of_{variable}"],
                 kde = False, label = f'number of {variable} - w injections')

    # Plot formatting
    plt.legend(prop = {'size': 12})
    plt.title(f'number of {variable}')
    plt.xlabel(f'{variable}')
    plt.ylabel('number of sequences')
    plt.savefig(os.path.join(SAVE_DIR, f"number_of_{variable}.png"))
    plt.close()


def n_of_diseases(table):
    amd_bool = table.diagnosis == "AMD"
    dr_bool = table.diagnosis == "DR"
    naive_bool = table.naive == True
    injections_bool = table.number_of_injections > 0

    y_values = [table.loc[amd_bool].shape[0],
                table.loc[dr_bool].shape[0],
                table.loc[naive_bool].shape[0],
                table.loc[amd_bool & injections_bool].shape[0],
                table.loc[dr_bool & injections_bool].shape[0],
                table.loc[naive_bool & injections_bool].shape[0]]

    x_labels = ["amd", "dr", "naive", "amd w inj", "dr with inj", "naive with inj"]
    sns.barplot(x = x_labels, y = y_values)

    # Plot formatting
    plt.legend(prop = {'size': 12})
    plt.title(f'number of cases')
    plt.xlabel(f'group')
    plt.ylabel('cases')
    plt.savefig(os.path.join(SAVE_DIR, f"number_of_diseases.png"))
    plt.close()


def time_until_dry(table):
    injections_bool = table.number_of_injections > 0
    fluid_bool = table.time_until_dry != "no fluid"
    sns.distplot(table.loc[fluid_bool].time_until_dry, kde = False, label = f'time until dry of all - all')
    sns.distplot(table.loc[injections_bool & fluid_bool].time_until_dry, kde = False,
                 label = f'time until dry of treated - all')
    sns.distplot(table.loc[~injections_bool & fluid_bool].time_until_dry, kde = False,
                 label = f'time until dry non treated - all')

    # Plot formatting
    plt.legend(prop = {'size': 12})
    plt.title('time until dry')
    plt.xlabel('time in months')
    plt.ylabel('number of sequences')
    plt.savefig(os.path.join(SAVE_DIR, f"time_until_dry.png"))
    plt.close()


def treatment_effect(table):
    table = table.replace([np.inf, -np.inf], np.nan)
    sns.distplot(table.dropna().three_month_effect, kde = False, label = f'three month')
    sns.distplot(table.dropna().six_month_effect, kde = False, label = f'six month')

    # Plot formatting
    plt.legend(prop = {'size': 12})
    plt.title('treatment effect on fluid')
    plt.xlabel('progression')
    plt.ylabel('number of sequences')
    plt.savefig(os.path.join(SAVE_DIR, f"treatment_effect.png"))
    plt.close()


def treatment_coefficient(table, naive):
    if naive:
        table = table.iloc[table.naive.values]
        name_ = "naive"
    else:
        name_ = "all"

    injection = table[table.fluid_change_injections != "no injection"]
    no_injection = table[table.fluid_change_no_injections != "no fluid"]

    sns.distplot(injection.fluid_change_injections, label = "injection")
    sns.distplot(no_injection.fluid_change_no_injections, label = "no injection")
    plt.xlim([-10000, 10000])
    plt.legend()
    plt.title("treatment coefficient all patients")
    plt.savefig(os.path.join(SAVE_DIR, f"treatment_coeff_{name_}_patients.png"))
    plt.close()


def fluid_distribution(table):
    sns.distplot(table["total_fluid"].values)
    plt.xlim(0, 3000000)
    plt.title("Number of pixels with fluid accross volumes")
    plt.savefig(os.path.join(SAVE_DIR, f"fluid_distribution.png"))
    plt.close()

def atrophy_delta_distribution(table):
    sns.distplot(table.atrophy_delta, kde=False, label="atrophy delta all")
    sns.distplot(table.atrophy_delta.loc[table.naive], kde=False, label="atrophy delta naive")

    # Plot formatting
    plt.legend(prop = {'size': 12})
    plt.title(f'atrophy delta distribution')
    plt.xlabel(f'atrophy delta')
    plt.ylabel('number of sequences')
    plt.savefig(os.path.join(SAVE_DIR, f"number_of_atrophy_delta.png"))
    plt.close()

def atrophy_wrt_injections_table(table):
    log = {}
    injections_levels = [0, 1, 2, 3, 4]
    for inj_lev in injections_levels:
        inj_lev_bool = table.number_of_injections == inj_lev
        inj_level_stat = table.atrophy_delta.loc[inj_lev_bool]
        log[inj_lev] = [np.mean(inj_level_stat), np.std(inj_level_stat)]

    inj_lev_bool = table.number_of_injections >= 5
    inj_level_stat = table.atrophy_delta.loc[inj_lev_bool]
    log[5] = [np.mean(inj_level_stat), np.std(inj_level_stat)]


def fluid_wrt_injections_table(table):
    log = {}
    injections_levels = [0, 1, 2, 3, 4]
    for inj_lev in injections_levels:
        inj_lev_bool = table.number_of_injections == inj_lev
        inj_level_stat = table.total_fluid.loc[inj_lev_bool]
        log[inj_lev] = [np.mean(inj_level_stat), np.std(inj_level_stat)]

    inj_lev_bool = table.number_of_injections >= 5
    inj_level_stat = table.total_fluid.loc[inj_lev_bool]
    log[5] = [np.mean(inj_level_stat), np.std(inj_level_stat)]


'''
# sequence data
'''

SAVE_DIR = os.path.join(WORK_SPACE, "plots")

statistics_pd = pd.read_csv(os.path.join(WORK_SPACE, "sequence_data/statistics.csv"), index_col = 0)
#time_until_dry_pd = pd.read_csv(os.path.join(WORK_SPACE, "sequence_data/time_until_dry.csv"), index_col = 0)

#fluid_wrt_injections_table(time_until_dry_pd)
'''
atrophy_pd = pd.read_csv(os.path.join(WORK_SPACE, "sequence_data/atrophy.csv"), index_col = 0)
segmentation_pd = pd.read_csv(os.path.join(WORK_SPACE, "segmentation_statistics.csv"), index_col = 0)

# add total fluid to data frame
fluid_columns = list(filter(lambda x: ("3" in x) or ("4" in x), segmentation_pd.columns.tolist()))
segmentation_pd["total_fluid"] = segmentation_pd[fluid_columns].sum(1)

n_of_visits(time_until_dry_pd, variable = "visits")
n_of_visits(time_until_dry_pd, variable = "months")
n_of_diseases(table = time_until_dry_pd)
time_until_dry(table = time_until_dry_pd)
fluid_distribution(segmentation_pd)
treatment_coefficient(table = time_until_dry_pd, naive = True)
treatment_coefficient(table = time_until_dry_pd, naive = False)
atrophy_delta_distribution(atrophy_pd)
'''
print("stop")

atrophy_deltas = []
for l_ in statistics_pd.atrophy_deltas_treated:
    for elem in l_:
        atrophy_deltas.append(elem)
