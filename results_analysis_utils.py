import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import torch
import csv
import collections
import pandas as pd
import scikit_posthocs as sp
from statsmodels.sandbox.stats.runs import cochrans_q

from itertools import chain, combinations

def read_surveys_file(file_path):
    csv_rows, fields = [], []
    results = {}
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                fields = row
            else:
                csv_rows.append(row)
                results['user' + str(line)] = {}
                for i in range(len(row)):
                    results['user' + str(line)][fields[i]] = row[i]
            line += 1
        return results, fields


def read_csv_file(file_path):
    csv_rows = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_rows.append(row)
        return csv_rows


def prepare_histogram_bars(survey_votes, field_to_filter="", filter_criteria=""):
    idx = np.array(survey_votes[field_to_filter]) == filter_criteria
    print(idx)
    tot_votes = sum([int(i) for i in idx])

    larger15 = sum(np.array(survey_votes['larger15'])[idx]) / tot_votes * 100
    smaller15 = sum(np.array(survey_votes['smaller15'])[idx]) / tot_votes * 100
    larger20 = sum(np.array(survey_votes['larger20'])[idx]) / tot_votes * 100
    smaller20 = sum(np.array(survey_votes['smaller20'])[idx]) / tot_votes * 100
    baseline = sum(np.array(survey_votes['baseline'])[idx]) / tot_votes * 100

    larger = sum(np.maximum(np.array(survey_votes["larger15"])[idx],
                            np.array(survey_votes["larger20"])[idx])) / tot_votes * 100
    smaller = sum(np.maximum(np.array(survey_votes["smaller15"])[idx],
                             np.array(survey_votes["smaller20"])[idx])) / tot_votes * 100

    return larger15, smaller15, larger20, smaller20, baseline, smaller, larger


def statistical_tests(tuple_of_arrays, dataframe):
    matrix = np.concatenate((tuple_of_arrays), axis=0)
    st, pvalue = cochrans_q(matrix.T)
    # st, pvalue = stats.cochrans_q(matrix.T)

    print("Cochran’s test: Q", round(st, 3), "p-value", round(pvalue, 3))
    # "Since the p-value is smaller than α, we can reject the null hypothesis and
    # conclude that there is a difference between the classification accuracies""
    matrix2 = pd.melt(dataframe, var_name='options', value_name='chosen')
    print("Dunn’s test")
    dunns = sp.posthoc_dunn(matrix2, val_col='chosen', group_col='options',
                            p_adjust='bonferroni')
    print(round(dunns, 3))
    return st, pvalue, dunns


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def combine_responses(chosen_1over20, chosen_1over15, chosen_baseline, chosen_15, chosen_20):
    names = ("1/2.0", "1/1.5", "baseline", "1.5", "2.0")
    resp_per_scaling = [chosen_1over20, chosen_1over15, chosen_baseline, chosen_15, chosen_20]
    scaling_combinations = {}

    tot_votes = sum([int(i) for i in (chosen_20 | chosen_15 | chosen_baseline | chosen_1over20 | chosen_1over15)])

    for comb in list(powerset(range(5)))[1:]:
        comb_idx = list(comb)
        to_combine_names = [names[c] for c in comb_idx]
        to_combine = [resp_per_scaling[c] for c in range(5) if c in comb_idx] + [~resp_per_scaling[c] for c in range(5)
                                                                                 if c not in comb_idx]
        name = "_".join(to_combine_names)
        vector = to_combine[0]
        for v in to_combine[1:]:
            vector = vector * v
        scaling_combinations[name] = vector

    larger_preference = ['1.5_2.0', '1.5', '2.0']
    smaller_preference = ['1/2.0_1/1.5', '1/1.5', '1/2.0']
    baseline_preference = ['baseline']
    factor15_preference = ['1.5']
    factor20_preference = ['2.0']
    no_preference = [p for p in list(scaling_combinations.keys()) \
                     if p not in larger_preference and \
                     p not in smaller_preference and \
                     p not in baseline_preference]

    vector = scaling_combinations[larger_preference[0]]
    for i in larger_preference[1:]:
        vector += scaling_combinations[i]
    larger_preference_value = sum([int(i) for i in vector]) / tot_votes * 100
    larger_to_return = [int(i) for i in vector]

    vector = scaling_combinations[smaller_preference[0]]
    for i in smaller_preference[1:]:
        vector += scaling_combinations[i]
    smaller_preference_value = sum([int(i) for i in vector]) / tot_votes * 100

    vector = scaling_combinations[baseline_preference[0]]
    for i in baseline_preference[1:]:
        vector += scaling_combinations[i]
    baseline_preference_value = sum([int(i) for i in vector]) / tot_votes * 100

    vector = scaling_combinations[no_preference[0]]
    for i in no_preference[1:]:
        vector += scaling_combinations[i]
    no_preference_value = sum([int(i) for i in vector]) / tot_votes * 100

    # For the other plots
    vector = scaling_combinations[factor15_preference[0]]
    for i in factor15_preference[1:]:
        vector += scaling_combinations[i]
    factor15_preference_value = [int(i) for i in vector]

    vector = scaling_combinations[factor20_preference[0]]
    for i in factor20_preference[1:]:
        vector += scaling_combinations[i]
    factor20_preference_value = [int(i) for i in vector]

    tot_votes_larger = sum([sum(larger_to_return), sum(factor15_preference_value), sum(factor20_preference_value)])

    return larger_preference_value, smaller_preference_value, \
           baseline_preference_value, no_preference_value, \
           larger_to_return, factor15_preference_value, \
           factor20_preference_value, tot_votes_larger


def load_data(filename, with_torch=False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else:
        print("File", filename, "does not exists.")


def add_curve(ax, x, y, label, y_min, y_max, x_min, x_max,
              title, x_label, y_label, fmt='-o', color=''):
    ax.errorbar(x, y, fmt=fmt, label=label, color=color)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    # ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    return ax


