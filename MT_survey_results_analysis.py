import os
from time import time
import csv
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from statsmodels.sandbox.stats.runs import cochrans_q
import simpledorff
import matplotlib.pyplot as plt


def read_surveys_file(file_path):
    csv_rows, fields = [], []
    res = {}
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                fields = row
            else:
                csv_rows.append(row)
                res['entry' + str(line)] = {}
                for i in range(len(row)):
                    res['entry' + str(line)][fields[i]] = row[i]
            line += 1
        return res, fields


def summarize_results(res, filter_by_category = ''):
    summary = {}
    users_to_remove = []
    attention_check = ["578922_toothbrush0_medium",
                       "62808_pizza0_large",
                       "365745_traffic_light0",
                       "62808_person0_large"]

    for entry in res:
        user_id = res[entry]["WorkerId"]
        task_id = res[entry]["AssignmentId"]
        link1 = res[entry]["Input.imageLink1"]
        link2 = res[entry]["Input.imageLink2"]
        link3 = res[entry]["Input.imageLink3"]
        link4 = res[entry]["Input.imageLink4"]
        image_id = link1.split("/")[-1].split("_IoU")[0]

        for i in range(4):
            if res[entry]['Answer.detection'+str(i+1)+'.'+str(i+1)] == 'true':
                choice = i+1
        detection1 = 1 if choice == 1 else 0
        detection2 = 1 if choice == 2 else 0
        detection3 = 1 if choice == 3 else 0
        detection4 = 1 if choice == 4 else 0

        if image_id not in attention_check:
            if filter_by_category in image_id:
                summary[entry] = {'user_id': user_id, 'task_id': task_id, 'choice': choice, 'image_id': image_id,
                              'link1': link1, 'link2': link2, 'link3': link3, 'link4': link4,
                              'detection1' : detection1, 'detection2' : detection2,
                              'detection3': detection3, 'detection4': detection4}
        else:
            for i in range(4):
                if 'correct' not in res[entry]['Input.imageLink' + str(i + 1)] and \
                    res[entry]['Answer.detection'+str(i+1)+'.'+str(i+1)] == 'true':
                    users_to_remove.append(user_id)

    # Discard users that failed an attention check
    print("users_to_remove", users_to_remove)
    for entry in summary:
        if summary[entry]['user_id'] in users_to_remove:
            summary.pop[entry]

    return summary


def create_dataframe(summary):
    keys = summary[list(summary.keys())[0]].keys()
    survey_votes = {k:[] for k in keys}
    for entry in summary:
        for k in keys:
            survey_votes[k] += [summary[entry][k]]
    return survey_votes


def statistical_tests(tuple_of_arrays, dataframe):
    # print("tuple_of_arrays", tuple_of_arrays)
    matrix = np.concatenate((tuple_of_arrays), axis=0)
    st, pvalue = cochrans_q(matrix.T)

    print("Cochran’s test: Q", round(st, 3), "p-value", round(pvalue, 3))
    # "Since the p-value is smaller than α, we can reject the null hypothesis and
    # conclude that there is a difference between the classification accuracies""
    # matrix2 = pd.melt(dataframe, var_name='options', value_name='chosen')
    print("Dunn’s test")
    dunns = sp.posthoc_dunn(matrix, val_col='chosen', group_col='options',
                            p_adjust='bonferroni')
    print(round(dunns, 3))

    krippendorff_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(dataframe,
                                                                          experiment_col='image_id',
                                                                          annotator_col='user_id',
                                                                          class_col='choice')
    print("krippendorff_alpha", round(krippendorff_alpha,3))
    return st, pvalue, dunns


def plot_histogram(labels, detection1, detection2, detection3, detection4, fig_path, width=1):

    fig, ax = plt.subplots()
    rects1 = ax.barh(labels, [detection1, detection2, detection3, detection4],
                     width, label=labels[0], color="#1D60E7")
    rects1[0].set_color("#1D60E7")
    rects1[1].set_color("#87AAF0")
    rects1[2].set_color("#FFCA6E")
    rects1[3].set_color("#EA8181")
    # rects1[4].set_color("#F21A1A")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Percentage Chosen Bounding Box', fontsize=14)
    ax.set_xlim([0, 50])
    # ax.set_yticklabels(x, fontsize=12)
    ax.figure.canvas.draw()
    ax.tick_params(axis='y', which='major')

    fig.tight_layout()

    plt.savefig(fig_path, bbox_inches='tight', format='png', dpi=1200)
    plt.show()
    return


def analize_results(results_summary):
    survey_votes = create_dataframe(results_summary)

    print("Statistical tests for all the responses")
    dataframe = pd.DataFrame(survey_votes)
    st, pvalue, dunns = statistical_tests((np.array([survey_votes["detection1"]]),
                                           np.array([survey_votes["detection2"]]),
                                           np.array([survey_votes["detection3"]]),
                                           np.array([survey_votes["detection4"]])),
                                           dataframe)

    tot_votes = len(list(survey_votes.values())[0])
    print("Number of judgements", tot_votes)
    grouped_by_image = dataframe.groupby(['image_id'])['image_id'].count()
    print("Number of judgements per image", round(grouped_by_image.mean(axis=0),2),
          "pm", round(grouped_by_image.std(axis=0),2))

    detection1 = sum(survey_votes["detection1"]) / tot_votes * 100
    detection2 = sum(survey_votes["detection2"]) / tot_votes * 100
    detection3 = sum(survey_votes["detection3"]) / tot_votes * 100
    detection4 = sum(survey_votes["detection4"]) / tot_votes * 100
    print(round(detection1,1), round(detection2,1), round(detection3,1), round(detection4,1))
    return detection1, detection2, detection3, detection4


def main():

    # csv_file = "asymmetric_loss_survey_images/survey_results/first_results.csv"
    csv_file = "asymmetric_loss_survey_images/survey_results/Batch_376145_batch_results.csv"
    figures_dir = "asymmetric_loss_survey_images/survey_results/plots/"
    if not os.path.exists(figures_dir): os.mkdir(figures_dir)

    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (5.5, 3.5)
    x = [r'$\alpha=1$ (smooth L1 loss)', r'$\alpha=10$', r'$\alpha=100$', 'Scaling factor 1.5']

    results, fields = read_surveys_file(csv_file)

    print("All images")
    summary = summarize_results(results)
    detection1, detection2, detection3, detection4 = analize_results(summary)
    plot_histogram(x, detection1, detection2, detection3, detection4, figures_dir + 'results.png')
    print("Small objects")
    summary = summarize_results(results, filter_by_category='small')
    detection1, detection2, detection3, detection4 = analize_results(summary)
    plot_histogram(x, detection1, detection2, detection3, detection4, figures_dir + 'small_results.png')
    print("Medium objects")
    summary = summarize_results(results, filter_by_category='medium')
    detection1, detection2, detection3, detection4 = analize_results(summary)
    plot_histogram(x, detection1, detection2, detection3, detection4, figures_dir + 'medium_results.png')
    print("Large objects")
    summary = summarize_results(results, filter_by_category='large')
    detection1, detection2, detection3, detection4 = analize_results(summary)
    plot_histogram(x, detection1, detection2, detection3, detection4, figures_dir + 'large_results.png')

    return results


if __name__ == "__main__":
    results = main()
