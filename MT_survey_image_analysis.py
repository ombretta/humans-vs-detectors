import os
import csv
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import rc


from MT_survey_results_analysis import read_surveys_file, summarize_results, create_dataframe


def analize_results(results_summary):
    survey_votes = create_dataframe(results_summary)

    print("Statistical tests for all the responses")
    dataframe = pd.DataFrame(survey_votes)

    tot_votes = len(list(survey_votes.values())[0])
    print("Number of judgements", tot_votes)
    grouped_by_image = dataframe.groupby(['image_id'])['image_id'].count()
    print("Number of judgements per image", round(grouped_by_image.mean(axis=0),2),
          "pm", round(grouped_by_image.std(axis=0),2))

    detection1 = sum(survey_votes["detection1"]) / tot_votes * 100
    detection2 = sum(survey_votes["detection2"]) / tot_votes * 100
    detection3 = sum(survey_votes["detection3"]) / tot_votes * 100
    detection4 = sum(survey_votes["detection4"]) / tot_votes * 100
    # print(round(detection1,1), round(detection2,1), round(detection3,1), round(detection4,1))
    return dataframe, detection1, detection2, detection3, detection4

def display_images(images, values, images_path, fig_path, var_names):
    rc('font', **{'family': 'serif', 'serif': ['Times']})

    _, axs = plt.subplots(len(values), 4, figsize=(28, 28), gridspec_kw = {'wspace': 0.0001, 'hspace': 0.2})
    # plt.subplots_adjust(left=0, bottom=0, right=1.5, top=1, wspace=0, hspace=0)

    for ix, image in enumerate(images):
        image_preference = values[values["image_id"].str.contains(image)]
        axx = axs[ix].flatten()
        alpha1_path = [i for i in os.listdir(images_path+"alpha1/") if image in i][0]
        alpha10_path = [i for i in os.listdir(images_path + "alpha10/") if image in i][0]
        alpha100_path = [i for i in os.listdir(images_path + "alpha100/") if image in i][0]
        factor15_path = [i for i in os.listdir(images_path + "factor1.5/") if image in i][0]

        alpha1 = cv2.imread(images_path+"alpha1/"+alpha1_path)[:, :, [2, 1, 0]]
        alpha10 = cv2.imread(images_path+"alpha10/"+alpha10_path)[:, :, [2, 1, 0]]
        alpha100 = cv2.imread(images_path+"alpha100/"+alpha100_path)[:, :, [2, 1, 0]]
        factor15 = cv2.imread(images_path+"factor1.5/"+factor15_path)[:, :, [2, 1, 0]]
        for det, (img, ax) in enumerate(zip([alpha1, alpha10, alpha100, factor15], axx)):
            ax.axis('off')
            ax.imshow(img)
            ax.text(0.5, -0.08, str(image_preference.iloc[0,det+1])+"%", transform=ax.transAxes, fontsize=30,
                    horizontalalignment='center', verticalalignment='center')
            if image_preference.iloc[0,det+1] == max(image_preference.iloc[0,1:]):
                rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], edgecolor='red', linewidth=5, facecolor='none')
                ax.add_patch(rect)
            if ix == 0:
                ax.text(0.5, 1.2, var_names[det], transform=ax.transAxes, fontsize=30,
                        horizontalalignment='center', verticalalignment='center')

    plt.savefig(fig_path, bbox_inches='tight', format='png')#, dpi=1200)
    plt.savefig(fig_path.replace("png", "pdf"), bbox_inches='tight', format='pdf')#, dpi=1200)

def check_votes(dataframe, images_path, figures_dir, var_names):

    # filter attention_check
    filtered = dataframe[~dataframe["link1"].str.contains("attention")]
    selected = filtered[["image_id", 'detection1', 'detection2', 'detection3', 'detection4']]
    grouped = selected.groupby("image_id").sum()
    tot_votes = grouped.sum(axis=1)
    divided = round(grouped.div(tot_votes, axis=0)*100, 2).reset_index()

    # Top alpha 1
    top = divided.sort_values(by=['detection1'], ascending=False)[:5]
    print("ALPHA=1 (ORIGINAL MODEL)\n", top)
    fig_path = figures_dir + "alpha1_preferred.png"
    display_images(top["image_id"], top, images_path, fig_path, var_names)

    # Top alpha 10
    top = divided.sort_values(by=['detection2'], ascending=False)[:5]
    print("ALPHA=10\n", top)
    fig_path = figures_dir + "alpha10_preferred.png"
    display_images(top["image_id"], top, images_path, fig_path, var_names)

    # Top alpha 100
    top = divided.sort_values(by=['detection3'], ascending=False)[:5]
    print("ALPHA=100\n", top)
    fig_path = figures_dir + "alpha100_preferred.png"
    display_images(top["image_id"], top, images_path, fig_path, var_names)

    # Top factor 1.5
    top = divided.sort_values(by=['detection4'], ascending=False)[:5]
    print("FACTOR=1.5\n", top)
    fig_path = figures_dir + "factor1.5_preferred.png"
    display_images(top["image_id"], top, images_path, fig_path, var_names)

    # No preference
    min = 37
    no_pref = divided[(divided['detection1']<min) & (divided['detection2']<min) &
            (divided['detection3']<min) & (divided['detection4']<min)]
    print("No preference\n", no_pref)
    fig_path = figures_dir + "no_pref.png"
    display_images(no_pref["image_id"], no_pref, images_path, fig_path, var_names)

    # Images for the qualitative analysis:
    picked = ["424545_cat0_large", "428280_chair0_medium",
              "231879_cake0_medium", "466156_cat0_medium",
               "142472_person1_small", "360951_mouse1_small",
              "885_person2_medium"]
    handpicked = divided[(divided["image_id"].isin(picked))]
    print("Qualitative analysis\n", handpicked)
    fig_path = figures_dir + "qualitative_analysis.png"
    display_images(picked, handpicked, images_path, fig_path, var_names)



def main():

    # csv_file = "asymmetric_loss_survey_images/survey_results/first_results.csv"
    csv_file = "asymmetric_loss_survey_images/survey_results/Batch_376145_batch_results.csv"
    figures_dir = "asymmetric_loss_survey_images/survey_results/plots/"
    images_path = "asymmetric_loss_survey_images/chosen_images/"
    if not os.path.exists(figures_dir): os.mkdir(figures_dir)

    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (5.5, 3.5)
    x = [r'$\alpha=1$ (smooth L1 loss)', r'$\alpha=10$', r'$\alpha=100$', 'Scaling factor 1.5']

    results, fields = read_surveys_file(csv_file)
    summary = summarize_results(results)
    dataframe, detection1, detection2, detection3, detection4 = analize_results(summary)

    check_votes(dataframe, images_path, figures_dir, x)


    return results, dataframe


if __name__ == "__main__":
    results, dataframe = main()
