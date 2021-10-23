import datetime as dt
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentence_transformers.util
import sklearn.preprocessing as pre
import statsmodels.api as sm
import tabulate
import umap.umap_ as umap
import sklearn.metrics as metrics

from contrastive_loss_model import train
from data import ManifestoDataset

sns.set_theme("paper")


def visualize_before_tranform(manifesto_data, ):
    # TODO: push this into the training loop as per the example

    #manifesto_data = ManifestoDataset.load()
    take_indices = np.random.choice(np.arange(manifesto_data.texts_encoded.shape[0]), size=4000, replace=False)

    stacked = np.vstack(
        [
            manifesto_data.labels_encoded.numpy(),
            manifesto_data.texts_encoded[take_indices].numpy()
        ]
    )

    decomp = umap.UMAP()
    two_dim = decomp.fit_transform(stacked)

    colors = np.hstack([np.arange(manifesto_data.labels_encoded.shape[0]), np.array(manifesto_data.labels)[take_indices]])

    fig,ax = plt.subplots(1)
    ax.scatter(*two_dim.T, s=.5, c=colors)

    plt.show()


def time_series_by_year(manifesto_data):
    #manifesto_data.keys
    strptime = dt.datetime.strptime
    dates = [strptime(key.split("_")[1], "%Y%m") for key in manifesto_data.keys]
    df_annotated = pd.DataFrame({"date": dates, "label": manifesto_data.labels})
    df_aggregated = df_annotated.groupby(["date", "label"]).size().to_frame("n").reset_index().set_index("date")

    #labels_array = np.array(manifesto_data.labels)
    #distinct_labels = set(manifesto_data.labels)
    #n_random_labels = 3
    #random_labels = random.sample(list(distinct_labels), k=n_random_labels)

    holdout_labels = manifesto_data.holdout_labels
    selected_label_texts = [manifesto_data.labels_texts[i] for i in holdout_labels]

    all_sentences_by_period = df_aggregated.n.resample("Y").sum()

    _, trunk_model_linear, embed_model_linear = train("linear", epochs=1)
    _, trunk_model_nn, embed_model_nn = train("nn", epochs=2)

    scores = {}
    scores["raw"] = sentence_transformers.util.dot_score(
        pre.normalize(manifesto_data.texts_encoded),
        pre.normalize(manifesto_data.labels_encoded)
    )
    scores["linear"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_linear(trunk_model_linear(manifesto_data.texts_encoded)).detach().numpy()),
        pre.normalize(embed_model_linear(trunk_model_linear(manifesto_data.labels_encoded)).detach().numpy())
    )
    scores["nn"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_nn(trunk_model_nn(manifesto_data.texts_encoded)).detach().numpy()),
        pre.normalize(embed_model_nn(trunk_model_nn(manifesto_data.labels_encoded)).detach().numpy())
    )

    df_scores = {}
    for k in ["raw", "linear", "nn"]:
        df_scores[k] = pd.DataFrame(
            scores[k][:, holdout_labels].detach().numpy(),
            columns=[f"label_{label}" for label in holdout_labels]
        )
        df_scores[k]["date"] = dates
        df_scores[k].set_index("date", inplace=True)

    start_year = "2000"
    threshold = .3

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(12, 8))

    random_holdout_labels = random.sample(holdout_labels, 3)
    holdout_labels_no_plot = [x for x in holdout_labels if x not in random_holdout_labels]
    selected_label_texts = [manifesto_data.labels_texts[i] for i in random_holdout_labels]

    model_nice_names = ["reference", "cosine", "linear-weighted", "neural"]

    # store the trend lines
    label_regr_slope = []

    for label_idx, label in enumerate(random_holdout_labels + holdout_labels_no_plot):
        label_by_period = df_aggregated[df_aggregated.label == label].n.resample("Y").sum()
        pct_label_by_period = label_by_period/(all_sentences_by_period+1)
        label_col = f"label_{label}"
        for model_idx, model in enumerate(["reference", "raw", "linear", "nn"]):

            #pct_label_by_period[start_year:].plot(ax=axs[idx, 0])
            #(sum_scores_by_period/all_sentences_by_period)[start_year:].plot(ax=axs[idx, 1])

            if model == "reference":
                to_plot = pct_label_by_period[start_year:].fillna(0)
            else:
                df_scores_this_model = df_scores[model]
                sum_scores_by_period = df_scores_this_model[df_scores_this_model[label_col]>threshold].resample("Y")[label_col].sum()
                to_plot = (sum_scores_by_period/all_sentences_by_period)[start_year:].dropna()

            # plot a regression

            if label in random_holdout_labels:
                sns.regplot(y=to_plot, x=to_plot.index.year, ax=axs[label_idx, model_idx])
            # do a regression
            ols = sm.OLS(exog=sm.add_constant( to_plot.index.year), endog=to_plot)
            fit = ols.fit()
            slope = fit.params.x1
            label_regr_slope.append([label, model, slope])


            #sns.regplot(y=to_plot_soft, x=to_plot_soft.index.year, ax=axs[1, idx])
            #axs[0, idx].set_title(f"\\textit{{{manifesto_data.labels_texts[label]}}}")

        #d_pos = scores[:, label][labels_array == label]
        #d_neg = scores[:, label][labels_array != label]
        #axs[idx].hist(d_neg.numpy(), bins=100, alpha=.5)
        #axs[idx].hist(d_pos.numpy(), bins=100, alpha=.5)
        #axs[idx].semilogy()
        #axs[idx].set_title(f"\\textit{{{selected_label_texts[idx]}}}")

    for ax, col in zip(axs[0], model_nice_names):
        ax.set_title(col)

    for ax, row in zip(axs[:,0], selected_label_texts):
        ax.set_ylabel(row, rotation=90,)

    fig.tight_layout()
    fig.savefig("trend_analysis.png")
    plt.show()

    # analyse difference in trends
    df_slopes = pd.DataFrame(label_regr_slope, columns=["label", "model", "slope"])
    df_slopes = df_slopes.pivot_table(index="label", columns="model").slope
    df_slopes.index.name = None
    df_slope_abs_diff = df_slopes[["raw", "linear", "nn"]].subtract(df_slopes.reference, axis=0).abs().reset_index()
    df_slope_abs_diff.insert(0, 'label', df_slope_abs_diff["index"].map(manifesto_data.labels_texts.__getitem__))
    del df_slope_abs_diff["index"]
    print(tabulate.tabulate(df_slope_abs_diff.round(4), tablefmt="latex", headers=df_slope_abs_diff.columns, showindex=False))

    # are the signs of the slopes (directions in trends) the same?
    df_sign_equal = np.sign(df_slopes[["raw", "linear", "nn"]]).apply(lambda ser: ser == np.sign(df_slopes.reference)).reset_index()
    df_sign_equal.insert(0, 'label', df_sign_equal["index"].map(manifesto_data.labels_texts.__getitem__))
    del df_sign_equal["index"]
    df_sign_equal.mean()
    print(tabulate.tabulate(df_sign_equal.set_index("label").apply(lambda ser: ser.map(lambda el: "✓" if el else "")).reset_index(), tablefmt="latex", headers=df_sign_equal.columns, showindex=False))


def plot_distributions_test_queries(manifesto_data):
    scores = sentence_transformers.util.dot_score(pre.normalize(manifesto_data.texts_encoded), pre.normalize(manifesto_data.test_query_encodings))
    fig,axs = plt.subplots(3,2, sharex=True, sharey=True)
    for idx, query in enumerate(manifesto_data.test_queries):
        axs[idx%3, idx//3].hist(scores[:,idx].numpy(), bins=100)
        axs[idx%3, idx//3].set_title(f"\\textit{{{query}}}")
        #plt.setp(axs[idx].get_xticklabels(), visible=False)
    fig.savefig("corpus_distributions_unseen.png")


def show_best_queries_unseen_labels(manifesto_data):
    all_scores = {}
    all_scores["raw"] = sentence_transformers.util.dot_score(
        pre.normalize(manifesto_data.texts_encoded),
        pre.normalize(manifesto_data.test_query_encodings)
    )
    all_scores["linear"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_linear((manifesto_data.texts_encoded)).detach().numpy()),
        pre.normalize(embed_model_linear((manifesto_data.test_query_encodings)).detach().numpy())
    )
    all_scores["nn"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_nn((manifesto_data.texts_encoded)).detach().numpy()),
        pre.normalize(embed_model_nn((manifesto_data.test_query_encodings)).detach().numpy())
    )
    for k, scores in all_scores.items():
        arg_sorted = scores.argsort(dim=0, descending=True)[:5]
        for col, query_label in enumerate(manifesto_data.test_queries):
            print(k, query_label)
            for idx in arg_sorted[:, col]:
                print(manifesto_data.texts[idx])
            print()


def plot_label_space(manifesto_data: ManifestoDataset):
    umap = umap.UMAP()
    transformed = umap.fit_transform(manifesto_data.labels_encoded)

    fig, ax = plt.subplots(1)
    ax.scatter(transformed[:,0], transformed[:,1], s=2)
    for i, label in enumerate(manifesto_data.labels_texts):
        ax.annotate(label, (transformed[i,0], transformed[i,1]), fontproperties=matplotlib.text.FontProperties(size="xx-small"))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.gca().set_frame_on(False)
    fig.show()
    fig.savefig("labels_embedded.png")


def show_distances_labelled_data(manifesto_data: ManifestoDataset):
    scores = sentence_transformers.util.dot_score(pre.normalize(manifesto_data.texts_encoded), pre.normalize(manifesto_data.labels_encoded))
    labels_array = np.array(manifesto_data.labels)
    distinct_labels = set(manifesto_data.labels)
    n_random_labels = 3
    random_labels = random.choices(list(distinct_labels), k=n_random_labels)
    selected_label_texts = [manifesto_data.labels_texts[i] for i in random_labels]

    fig, axs = plt.subplots(n_random_labels,sharex=True, sharey=True)

    for idx, label in enumerate(random_labels):
        d_pos = scores[:, label][labels_array == label]
        d_neg = scores[:, label][labels_array != label]
        axs[idx].hist(d_neg.numpy(), bins=100, alpha=.5)
        axs[idx].hist(d_pos.numpy(), bins=100, alpha=.5)
        axs[idx].semilogy()
        axs[idx].set_title(f"\\textit{{{selected_label_texts[idx]}}}")
    fig.savefig("distances_labelled_data.png")


def cosine_model_eval_roc_auc(manifesto_data: ManifestoDataset):

    _, trunk_model_linear, embed_model_linear = train("linear", epochs=1)
    _, trunk_model_nn, embed_model_nn = train("nn", epochs=2)

    manifesto_train, manifesto_test = manifesto_data.split_by_holdout_labels()

    scores = {}
    scores["cosine"] = sentence_transformers.util.dot_score(
        pre.normalize(manifesto_data.texts_encoded[manifesto_test.indices]),
        pre.normalize(manifesto_data.labels_encoded[manifesto_data.holdout_labels])
    )
    scores["linear"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_linear(trunk_model_linear(manifesto_data.texts_encoded[manifesto_test.indices])).detach().numpy()),
        pre.normalize(embed_model_linear(trunk_model_linear(manifesto_data.labels_encoded[manifesto_data.holdout_labels])).detach().numpy())
    )
    scores["nn"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_nn(trunk_model_nn(manifesto_data.texts_encoded[manifesto_test.indices])).detach().numpy()),
        pre.normalize(embed_model_nn(trunk_model_nn(manifesto_data.labels_encoded[manifesto_data.holdout_labels])).detach().numpy())
    )

    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)

    for idx, label in enumerate(manifesto_data.holdout_labels[:4]):
        y_true = [x == label for x in np.array(manifesto_data.labels)[manifesto_test.indices]]
        for model, model_scores in scores.items():
            y_score = model_scores[:, idx].detach().numpy()
            fpr, tpr, thrsh = metrics.roc_curve(y_true, y_score)
            axs[idx//2, idx%2].plot(fpr, tpr, label=model)
        axs[idx//2, idx%2].set_title(f"\\textit{{{manifesto_data.labels_texts[label]}}}")
        if idx//2 == 1:
            axs[idx//2, idx%2].set_xlabel("False positive rate")
        if idx%2 == 0:
            axs[idx//2, idx%2].set_ylabel("True positive rate")

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
    axs.flatten()[-1].legend(loc='lower center', bbox_to_anchor=(0.7, 0), ncol=1)
    fig.show()
    fig.savefig("roc_curve_multi.png")

    #fig.legend( lines, labels, loc = (0.5, 0), ncol=5 )

    manifesto_train, manifesto_test = manifesto_data.split_by_holdout_labels()


    auc_scores = []
    for idx, label in enumerate(manifesto_data.holdout_labels):
        y_true = [x == label for x in np.array(manifesto_data.labels)[manifesto_test.indices]]
        for model, model_scores in scores.items():
            y_score = model_scores[:, idx].detach().numpy()
            auc = metrics.roc_auc_score(y_true, y_score)
            auc_scores.append((model, manifesto_data.labels_texts[label], auc))

    df = pd.DataFrame(auc_scores, columns=["model", "label", "auc_score"]).pivot_table(columns="model", index="label").auc_score
    print(tabulate.tabulate(df, tablefmt="latex",headers=df.columns, floatfmt=".2f"))
    print(tabulate.tabulate(df.mean(), tablefmt="latex",headers=df.columns, floatfmt=".2f"))



def results_table():
    cosine= {'val': {'epoch': 1,
              'AMI_level0': 0.24873027842900694,
              'NMI_level0': 0.24948181982787582,
              'mean_average_precision_level0': 0.30689698276907584,
              'mean_average_precision_at_r_level0': 0.26387371807075183,
              'precision_at_1_level0': 0.21179001721170396,
              'r_precision_level0': 0.295923781713968}}
    nn = {'val': {'epoch': 2, 'AMI_level0': 0.0923391042352795, 'NMI_level0': 0.0929133084283648, 'mean_average_precision_level0': 0.3813426309955886, 'mean_average_precision_at_r_level0': 0.31152548248617495, 'precision_at_1_level0': 0.21730165424809036, 'r_precision_level0': 0.39693066580539244}}
    linear = {'val': {'epoch': 1, 'AMI_level0': 0.13866463869321424, 'NMI_level0': 0.13971167740765608, 'mean_average_precision_level0': 0.41829824386550946, 'mean_average_precision_at_r_level0': 0.3354986656817992, 'precision_at_1_level0': 0.23833412619469793, 'r_precision_level0': 0.42161760661906034}}

    results = []
    for x in [cosine, linear, nn]:
        d = x["val"]
        del d["epoch"]
        results.append(d)
    df = pd.DataFrame(results).T
    df.columns = ["cosine", "linear", "nn"]
    print(tabulate.tabulate(df, floatfmt=".2f", tablefmt="latex", headers="keys"))


def trend_analysis_new():
    scores = {}
    scores["raw"] = sentence_transformers.util.dot_score(
        pre.normalize(manifesto_data.texts_encoded),
        pre.normalize(manifesto_data.test_query_encodings)
    )
    scores["linear"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_linear(trunk_model_linear(manifesto_data.texts_encoded)).detach().numpy()),
        pre.normalize(embed_model_linear(trunk_model_linear(manifesto_data.test_query_encodings)).detach().numpy())
    )
    scores["nn"] = sentence_transformers.util.dot_score(
        pre.normalize(embed_model_nn(trunk_model_nn(manifesto_data.texts_encoded)).detach().numpy()),
        pre.normalize(embed_model_nn(trunk_model_nn(manifesto_data.test_query_encodings)).detach().numpy())
    )
    strptime = dt.datetime.strptime

    dates = [strptime(key.split("_")[1], "%Y%m") for key in manifesto_data.keys]
    df_annotated = pd.DataFrame({"date": dates, "label": manifesto_data.labels})
    df_aggregated = df_annotated.groupby(["date", "label"]).size().to_frame("n").reset_index().set_index("date")
    all_sentences_by_period = df_aggregated.n.resample("Y").sum()

    df_scores = {}
    for k in ["raw", "linear", "nn"]:
        df_scores[k] = pd.DataFrame(
            scores[k].detach().numpy(),
            columns=[label for label in manifesto_data.test_queries]
        )
        df_scores[k]["date"] = dates
        df_scores[k].set_index("date", inplace=True)

    start_year = "2000"
    threshold = .3

    fig, axs = plt.subplots(len(manifesto_data.test_queries), 3, sharex=True, sharey=False, figsize=(12, 8))

    model_nice_names = ["cosine", "linear-weighted", "neural"]

    # store the trend lines
    label_regr_slope = []

    for label_idx, label in enumerate(manifesto_data.test_queries):
        label_col = label
        for model_idx, model in enumerate(["raw", "linear", "nn"]):

            #pct_label_by_period[start_year:].plot(ax=axs[idx, 0])
            #(sum_scores_by_period/all_sentences_by_period)[start_year:].plot(ax=axs[idx, 1])

            df_scores_this_model = df_scores[model]
            sum_scores_by_period = df_scores_this_model[df_scores_this_model[label_col]>threshold].resample("Y")[label_col].sum()
            to_plot = (sum_scores_by_period/all_sentences_by_period)[start_year:].dropna()

            # plot a regression

            sns.regplot(y=to_plot, x=to_plot.index.year, ax=axs[label_idx, model_idx])
            # do a regression
            ols = sm.OLS(exog=sm.add_constant( to_plot.index.year), endog=to_plot)
            fit = ols.fit()
            slope = fit.params.x1
            label_regr_slope.append([label, model, slope])


            #sns.regplot(y=to_plot_soft, x=to_plot_soft.index.year, ax=axs[1, idx])
            #axs[0, idx].set_title(f"\\textit{{{manifesto_data.labels_texts[label]}}}")

        #d_pos = scores[:, label][labels_array == label]
        #d_neg = scores[:, label][labels_array != label]
        #axs[idx].hist(d_neg.numpy(), bins=100, alpha=.5)
        #axs[idx].hist(d_pos.numpy(), bins=100, alpha=.5)
        #axs[idx].semilogy()
        #axs[idx].set_title(f"\\textit{{{selected_label_texts[idx]}}}")

    for ax, col in zip(axs[0], model_nice_names):
        ax.set_title(col)

    for ax, row in zip(axs[:,0], manifesto_data.test_queries):
        ax.set_ylabel(row, rotation=90,)

    fig.tight_layout()
    fig.savefig("trend_analysis_new.png")
    plt.show()


    examples = []
    for k, label_scores in scores.items():
        arg_sorted = label_scores.argsort(dim=0, descending=True)[:3]
        for col, query_label in enumerate(manifesto_data.test_queries):
            for idx in arg_sorted[:, col]:
                examples.append((query_label, k, manifesto_data.texts[idx]))


