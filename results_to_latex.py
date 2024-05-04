import pickle

import numpy as np
import pandas as pd


def load_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def contract_dict(data, labels, stats):
    new_data = dict()

    for model_name, results in data.items():
        results_contracted = {
            label: {st: val for st, val in stat.items() if st in stats}
            for label, stat in results.items()
            if label in labels
        }
        new_data[model_name] = results_contracted

    return new_data

def format_value(value):
    if np.isnan(value):
        return "-"
    return f"{value:.3f}"

def to_latex_table(data, major_columns):
    col_names = "Model"

    for col in major_columns:
        col_names += f"& {col} & {col} pval "

    table = "\\begin{tabular}{lcccc}\n\\toprule \n"
    table += col_names
    table += "\\\\ \n\\midrule\n"


    for model_name, results in data.items():
        entry = model_name

        for col in major_columns:
            mean = results['mean'][col]
            std = results['std'][col]
            pval = results['pval two-sided'][col]
            entry += f" & {format_value(mean)} ({format_value(std)}) & {format_value(pval)}"

        table += entry + "\\\\ \n"

    table += "\\bottomrule\n\\end{tabular}"
    return table


if __name__=="__main__":
    # from results_digits import data
    # from results_yeast import data
    # from synth import data
    data = load_file("test_exp1_digits_stats.pickle")
    # data = contract_dict(data, ["mean", "std", "pval two-sided"], ["ground truth"])
    data = contract_dict(data, ["mean", "std", "pval two-sided"], ["map", "exp"])
    print(to_latex_table(data,  ["map", "exp"]))