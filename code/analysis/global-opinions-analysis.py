from matplotlib import font_manager as fm, pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("./results/6_globalopinionsqa/globalopinionsqa.csv")

df = df[(df["China"] != -1) & (df["United States"] != -1) & (df["Jordan"] != -1) & (df["Brazil"] != -1) & (df["Nigeria"] != -1)  & (df["Germany"] != -1)  & (df["Australia"] != -1) ]

# df = df.drop(df[df['model_id'].isin(['Llama-2-7b-chat-hf', 'zephyr-7b-beta'])].index)
df = df[['model_id', 'China', 'United States', 'Jordan', 'Brazil', 'Nigeria', 'Germany', 'Australia']]
df_melted = df.melt(id_vars=['model_id'], var_name='Country', value_name='Similarity')

from scipy import stats
import seaborn as sns

sns.set(style="darkgrid")

# Function to compute bootstrap confidence intervals for each model and language
def compute_bootstrap_ci(data, model_column, answer_column, language_column, alpha=0.05):
    model_ids = data[model_column].unique()
    languages = data[language_column].unique()
    bootstrap_cis = {}

    for model_id in model_ids:
        for language in languages:
            model_data = data[(data[model_column] == model_id) & (data[language_column] == language)][answer_column].to_numpy()
            model_data = (model_data,)
            res = stats.bootstrap(model_data, np.mean, confidence_level=1-alpha)
            bootstrap_cis[(model_id, language)] = res.confidence_interval

    return bootstrap_cis

def compute_ci_bernoulli(data, model_column, answer_column, language_column, alpha=0.05):
    """
    Compute confidence intervals for the rate of correct answers for each model
    assuming a Bernoulli random variable and using the normal approximation.

    Parameters:
    data (pd.DataFrame): The dataset containing the model performance data.
    model_column (str): The column name for the model IDs.
    answer_column (str): The column name for the correct answer rates.
    alpha (float): The significance level for the confidence intervals.

    Returns:
    dict: A dictionary containing the model IDs as keys and the confidence intervals as values.
    """
    model_ids = data[model_column].unique()
    languages = data[language_column].unique()
    ci_bernoulli = {}
    
    for model_id in model_ids:
        for language in languages:
            model_data = data[(data[model_column] == model_id) & (data[language_column] == language)][answer_column]
            n = len(model_data)
            p_hat = model_data.mean()
            z = stats.norm.ppf(1 - alpha/2)
            half_width = z * (np.sqrt(p_hat * (1 - p_hat) / n))
            ci_bernoulli[(model_id, language)] = (p_hat - half_width, p_hat + half_width)
    
    return ci_bernoulli

def compute_gaussian_ci(data, model_column, answer_column, language_column, alpha=0.05):
    """
    Compute confidence intervals for the rate of correct answers for each model
    assuming a Gaussian distribution, using the sample mean and standard deviation.

    Parameters:
    data (pd.DataFrame): The dataset containing the model performance data.
    model_column (str): The column name for the model IDs.
    answer_column (str): The column name for the correct answer rates.
    language_column (str): The column name for the languages.
    alpha (float): The significance level for the confidence intervals.

    Returns:
    dict: A dictionary containing the model IDs as keys and the confidence intervals as values.
    """
    model_ids = data[model_column].unique()
    languages = data[language_column].unique()
    ci_gaussian = {}

    for model_id in model_ids:
        for language in languages:
            model_data = data[(data[model_column] == model_id) & (data[language_column] == language)][answer_column]
            n = len(model_data)
            if n <= 1:
                # If there's only one sample, we cannot compute the confidence interval
                ci_gaussian[(model_id, language)] = (float('nan'), float('nan'))
            else:
                sample_mean = model_data.mean()
                sample_std = model_data.std(ddof=1)
                z = stats.norm.ppf(1 - alpha/2)
                half_width = z * (sample_std / np.sqrt(n))
                ci_gaussian[(model_id, language)] = (sample_mean - half_width, sample_mean + half_width)

    return ci_gaussian


# Function to generate comparison plots
def generate_comparison_plots(dataframe, model_pairs, plot_titles, xlabel, ylabel, fontsize, grid_size, alpha, output_file, metric_column='Similarity', group_column='Country', model_column='model_id', ylim=(0,1)):

    bootstrap_cis = compute_gaussian_ci(dataframe, model_column, metric_column, group_column, alpha=alpha)

    fig_width = 11 * grid_size[1]
    fig_height = 8 * grid_size[0]

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(fig_width, fig_height))
    axes = axes.ravel()

    base_model_color = "#B0C4DE"
    performance_diff_color = "#CD5C5C"
    performance_imp_color = "#3CB371"

    for i, (model_pair, title) in enumerate(zip(model_pairs, plot_titles)):
        base_model_df = dataframe[dataframe[model_column] == model_pair[0]]
        tuned_model_df = dataframe[dataframe[model_column] == model_pair[1]]

        average_accuracy_base = base_model_df.groupby(group_column)[metric_column].mean().reset_index()
        average_accuracy_tuned = tuned_model_df.groupby(group_column)[metric_column].mean().reset_index()

        merged_accuracy = average_accuracy_base.merge(average_accuracy_tuned, on=group_column, suffixes=('_base', '_tuned'))

        # Sort the languages by the base model accuracy
        merged_accuracy = merged_accuracy.sort_values(by=metric_column+'_base', ascending=True)

        merged_accuracy['performance_diff'] = merged_accuracy[metric_column+'_tuned'] - merged_accuracy[metric_column+'_base']
        better_performance = merged_accuracy['performance_diff'] > 0
        worse_performance = merged_accuracy['performance_diff'] < 0

        base_ci = [bootstrap_cis[(model_pair[0], language)] for language in merged_accuracy[group_column]]
        tuned_ci = [bootstrap_cis[(model_pair[1], language)] for language in merged_accuracy[group_column]]

        # Adjusting confidence intervals for plotting
        base_error = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(merged_accuracy[metric_column+'_base'], base_ci)]

        # Filter the confidence intervals for the tuned model
        tuned_ci_better = [ci for ci, is_better in zip(tuned_ci, better_performance) if is_better]
        tuned_ci_worse = [ci for ci, is_worse in zip(tuned_ci, worse_performance) if is_worse]

        # Tuned model error bars
        tuned_error_better = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(merged_accuracy[better_performance][metric_column+'_tuned'], tuned_ci_better)]
        tuned_error_worse = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(merged_accuracy[worse_performance][metric_column+'_tuned'], tuned_ci_worse)]

        sns.barplot(ax=axes[i], x=group_column, y=metric_column+'_base', data=merged_accuracy, color=base_model_color, label='Base Model Rate')

        axes[i].bar(merged_accuracy[worse_performance][group_column], abs(merged_accuracy[worse_performance]['performance_diff']),
                    bottom=merged_accuracy[worse_performance][metric_column+'_base'] - abs(merged_accuracy[worse_performance]['performance_diff']),
                    color=base_model_color, hatch='///', edgecolor=performance_diff_color, label='Performance Drop')

        axes[i].bar(merged_accuracy[better_performance][group_column], abs(merged_accuracy[better_performance]['performance_diff']),
                    bottom=merged_accuracy[better_performance][metric_column+'_base'],
                    color=performance_imp_color, label='Performance Gain')
        
        axes[i].errorbar(merged_accuracy[group_column], merged_accuracy[metric_column+'_base'], yerr=np.array(base_error).T, fmt='none', color='black', capsize=5)

        # Error bars for tuned model
        if len(tuned_error_worse) > 0:
            axes[i].errorbar(merged_accuracy[worse_performance][group_column], merged_accuracy[worse_performance][metric_column+'_base']-abs(merged_accuracy[worse_performance]['performance_diff']), yerr=np.array(tuned_error_worse).T, fmt='none', color='red', capsize=5)
        if len(tuned_error_better) > 0:
            axes[i].errorbar(merged_accuracy[better_performance][group_column], merged_accuracy[better_performance][metric_column+'_tuned'], yerr=np.array(tuned_error_better).T, fmt='none', color='green', capsize=5)

        axes[i].set_title(title, fontsize=fontsize)
        axes[i].set_xlabel(xlabel, fontsize=fontsize * 0.83)
        if i == 0:
            axes[i].set_ylabel(ylabel, fontsize=fontsize * 0.83)
        else:
            axes[i].set_ylabel(None)
        
        axes[i].set_ylim(ylim[0], ylim[1])
        axes[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
        axes[i].tick_params(axis='y', labelsize=fontsize * 0.75)

        if i != 0:
            axes[i].get_legend().remove()

    a0_handles, a0_labels = axes[0].get_legend_handles_labels()
    a1_handles, a1_labels = axes[1].get_legend_handles_labels()
    true_handles = [a0_handles[0], a1_handles[1], a0_handles[2]]
    true_labels = [a0_labels[0], a1_labels[1], a0_labels[2]]

    axes[0].legend(true_handles, true_labels, fontsize=fontsize * 0.4)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

sdf = df
sdf["China"] = sdf["China"] - sdf["United States"]
sdf["Jordan"] = sdf["Jordan"] - sdf["United States"]
sdf["Brazil"] = sdf["Brazil"] - sdf["United States"]
sdf["Nigeria"] = sdf["Nigeria"] - sdf["United States"]
sdf["Germany"] = sdf["Germany"] - sdf["United States"]
sdf["Australia"] = sdf["Australia"] - sdf["United States"]
sdf["United States"] = sdf["United States"] - sdf["United States"]

# Drop the United States
sdf = sdf.drop(columns=["United States"])

sdf_melted = sdf.melt(id_vars=['model_id'], var_name='Country', value_name='Similarity')

import matplotlib.patches as mpatches

def generate_comparison_plots(dataframe, model_groups, plot_titles, xlabel, ylabel, fontsize=24, grid_size=(2, 2), alpha=0.05, output_file="./visualization/md3_comparison2.pdf", base_palette=None, category_col='dialect', value_col='generated_correct_answer_num', model_col='model', ylim=(0,1)):
    """
    Generate a grid of bar plots with a consistent legend showing all model types,
    and ensure bar colors match the corresponding model type, regardless of presence.
    """

    if base_palette is None:
        base_palette = sns.color_palette("mako_r", n_colors=3)

    fig_width = 11 * grid_size[1]
    fig_height = 8 * grid_size[0]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(fig_width, fig_height))
    axes = axes.ravel()

    legend_labels = ['Base', 'SFT', 'PT']  # Labels for all possible models
    legend_handles = [mpatches.Patch(color=base_palette[i], label=label) for i, label in enumerate(legend_labels)]

    for i, (model_group, title) in enumerate(zip(model_groups, plot_titles)):
        # order dialects by the order of the first model in the group
        order = ['Jordan', 'China', 'Nigeria', 'Brazil', 'Germany', 'Australia']

        df = dataframe[dataframe[model_col].isin(model_groups[model_group])]
        
        # Create a color map for the models to ensure consistency
        color_map = dict(zip(model_groups[model_group], base_palette))
        
        # If a model is missing, create a new palette excluding the color of the missing model
        current_palette = [color_map[model] for model in model_groups[model_group] if model is not None]

        df['model_order'] = df[model_col].apply(lambda x: model_groups[model_group].index(x) if x in model_groups[model_group] else -1)
        df = df.sort_values('model_order')

        sns.barplot(data=df, x=category_col, y=value_col, hue=model_col, ax=axes[i], palette=current_palette, errorbar=('ci', 95), order=order)
        axes[i].set_title(title, fontsize=fontsize)
        axes[i].set_xlabel(xlabel, fontsize=fontsize)
        if i % grid_size[1] == 0:
            axes[i].set_ylabel(ylabel, fontsize=fontsize)
        else:
            axes[i].set_ylabel('')

        axes[i].tick_params(labelsize=fontsize * 0.8)
        axes[i].set_ylim(ylim[0], ylim[1])
        axes[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
        axes[i].tick_params(axis='y', labelsize=fontsize * 0.75)

        # Set the legend on the first subplot
        if i == 0:
            axes[i].legend(handles=legend_handles, fontsize=fontsize * 0.5, loc='lower right')
        else:
            axes[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()

model_groups = {
    'Llama Group': ['Llama-2-7b-hf', None, 'Llama-2-7b-chat-hf'],
    'Tulu Group': ['Llama-2-7b-hf', 'tulu-2-7b', 'tulu-2-dpo-7b'],
    'Mistral Group': ['Mistral-7B-v0.1', 'mistral-7b-sft-beta', 'zephyr-7b-beta'],
    'OpenChat Group': ['Mistral-7B-v0.1', 'openchat_3.5', 'Starling-LM-7B-alpha']
}

# Prepare plot_titles based on the model_groups keys
plot_titles = ['Llama to Llama Chat', 'Llama to Tulu', 'Mistral to Zephyr', 'Mistral to Starling']

# Calling the function to plot the model comparisons
generate_comparison_plots(sdf_melted, model_groups, plot_titles, '', 'USA Difference', fontsize=64, grid_size=(2, 2), alpha=0.05, base_palette=None, output_file="./visualization/globalopinionsqa.svg", category_col='Country', value_col='Similarity', model_col='model_id', ylim=(-0.1, 0.1))
generate_comparison_plots(sdf_melted, model_groups, plot_titles, '', 'USA Difference', fontsize=64, grid_size=(2, 2), alpha=0.05, base_palette=None, output_file="./visualization/globalopinionsqa.pdf", category_col='Country', value_col='Similarity', model_col='model_id', ylim=(-0.1, 0.1))

model_groups = {
    'Qwen Group': ['Qwen1.5-7B', None, 'Qwen1.5-7B-Chat'],
    'Yi Group': ['Yi-6B', None, 'Yi-6B-Chat'],
}

# Prepare plot_titles based on the model_groups keys
plot_titles = ['Qwen to Qwen Chat', 'Yi to Yi Chat']

# Calling the function to plot the model comparisons
generate_comparison_plots(sdf_melted, model_groups, plot_titles, '', 'USA Difference', fontsize=64, grid_size=(1, 2), alpha=0.05, base_palette=None, output_file="./visualization/globalopinionsqa_qwen_yi.svg", category_col='Country', value_col='Similarity', model_col='model_id', ylim=(-0.1, 0.1))
generate_comparison_plots(sdf_melted, model_groups, plot_titles, '', 'USA Difference', fontsize=64, grid_size=(1, 2), alpha=0.05, base_palette=None, output_file="./visualization/globalopinionsqa_qwen_yi.pdf", category_col='Country', value_col='Similarity', model_col='model_id', ylim=(-0.1, 0.1))