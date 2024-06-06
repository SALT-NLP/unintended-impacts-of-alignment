BELEBELE_PATH = "./results/3_belebele/belebele.csv"

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
df = pd.read_csv(BELEBELE_PATH)

# Pivot table of language_name and average is_correct for each model
pivot_table_avg = df.pivot_table(values='is_correct', index='language_name', columns='model', aggfunc=np.mean)

# Calculating 95% confidence intervals for the average of is_correct for each language and model combination
# Defining a function to calculate the 95% CI
def ci95(x):
    return stats.sem(x) * stats.t.ppf((1 + 0.95) / 2., len(x)-1)

# Pivot table for 95% CI of is_correct
pivot_table_ci95 = df.pivot_table(values='is_correct', index='language_name', columns='model', aggfunc=ci95)

# Combine the average and 95% CI into a single DataFrame for easier interpretation
combined_df = pivot_table_avg.copy()

for model in pivot_table_avg.columns:
    combined_df[model] = combined_df[model].astype(str)

# Filter the combined DataFrame for the specified languages only
specified_languages = ["Telugu", "Bengali", "Mesopotamian Arabic", "Swahili", "Russian", "Korean", "Finnish", "English", "Indonesian"]
filtered_combined_df = combined_df.loc[specified_languages]

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
            if n == 0:
                continue
            p_hat = model_data.mean()
            z = stats.norm.ppf(1 - alpha/2)
            half_width = z * (np.sqrt(p_hat * (1 - p_hat) / n))
            ci_bernoulli[(model_id, language)] = (p_hat - half_width, p_hat + half_width)
    
    return ci_bernoulli

# Function to generate comparison plots
def generate_comparison_plots(dataframe, model_pairs, plot_titles, xlabel, ylabel, fontsize, grid_size, alpha, output_file):
    dataframe = dataframe[dataframe['language_name'].isin(specified_languages)]

    # Rename Mesopotamian Arabic to Arabic
    dataframe['language_name'] = dataframe['language_name'].replace('Mesopotamian Arabic', 'Arabic')

    bootstrap_cis = compute_ci_bernoulli(dataframe, 'model', 'is_correct', 'language_name', alpha=alpha)

    fig_width = 11 * grid_size[1]
    fig_height = 8 * grid_size[0]

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(fig_width, fig_height))
    axes = axes.ravel()

    base_model_color = "#B0C4DE"
    performance_diff_color = "#CD5C5C"
    performance_imp_color = "#3CB371"

    for i, (model_pair, title) in enumerate(zip(model_pairs, plot_titles)):
        base_model_df = dataframe[dataframe['model'] == model_pair[0]]
        tuned_model_df = dataframe[dataframe['model'] == model_pair[1]]

        average_accuracy_base = base_model_df.groupby('language_name')['is_correct'].mean().reset_index()
        average_accuracy_tuned = tuned_model_df.groupby('language_name')['is_correct'].mean().reset_index()

        merged_accuracy = average_accuracy_base.merge(average_accuracy_tuned, on='language_name', suffixes=('_base', '_tuned'))

        # Sort the languages by the base model accuracy
        merged_accuracy = merged_accuracy.sort_values(by='is_correct_base', ascending=True)

        merged_accuracy['performance_diff'] = merged_accuracy['is_correct_tuned'] - merged_accuracy['is_correct_base']
        better_performance = merged_accuracy['performance_diff'] > 0
        worse_performance = merged_accuracy['performance_diff'] < 0

        base_ci = [bootstrap_cis[(model_pair[0], language)] for language in merged_accuracy['language_name']]
        tuned_ci = [bootstrap_cis[(model_pair[1], language)] for language in merged_accuracy['language_name']]

        # Adjusting confidence intervals for plotting
        base_error = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(merged_accuracy['is_correct_base'], base_ci)]

        # Filter the confidence intervals for the tuned model
        tuned_ci_better = [ci for ci, is_better in zip(tuned_ci, better_performance) if is_better]
        tuned_ci_worse = [ci for ci, is_worse in zip(tuned_ci, worse_performance) if is_worse]

        # Tuned model error bars
        tuned_error_better = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(merged_accuracy[better_performance]['is_correct_tuned'], tuned_ci_better)]
        tuned_error_worse = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(merged_accuracy[worse_performance]['is_correct_tuned'], tuned_ci_worse)]

        sns.barplot(ax=axes[i], x='language_name', y='is_correct_base', data=merged_accuracy, color=base_model_color, label='Base Model Rate')

        axes[i].bar(merged_accuracy[worse_performance]['language_name'], abs(merged_accuracy[worse_performance]['performance_diff']),
                    bottom=merged_accuracy[worse_performance]['is_correct_base'] - abs(merged_accuracy[worse_performance]['performance_diff']),
                    color=base_model_color, hatch='///', edgecolor=performance_diff_color, label='Performance Drop')

        axes[i].bar(merged_accuracy[better_performance]['language_name'], abs(merged_accuracy[better_performance]['performance_diff']),
                    bottom=merged_accuracy[better_performance]['is_correct_base'],
                    color=performance_imp_color, label='Performance Gain')
        
        axes[i].errorbar(merged_accuracy['language_name'], merged_accuracy['is_correct_base'], yerr=np.array(base_error).T, fmt='none', color='black', capsize=5)

        # Error bars for tuned model
        if len(tuned_error_worse) > 0:
            axes[i].errorbar(merged_accuracy[worse_performance]['language_name'], merged_accuracy[worse_performance]['is_correct_base']-abs(merged_accuracy[worse_performance]['performance_diff']), yerr=np.array(tuned_error_worse).T, fmt='none', color='red', capsize=5)
        if len(tuned_error_better) > 0:
            axes[i].errorbar(merged_accuracy[better_performance]['language_name'], merged_accuracy[better_performance]['is_correct_tuned'], yerr=np.array(tuned_error_better).T, fmt='none', color='green', capsize=5)

        axes[i].set_title(title, fontsize=fontsize)
        axes[i].set_xlabel(xlabel, fontsize=fontsize * 0.83)
        if i == 0:
            axes[i].set_ylabel(ylabel, fontsize=fontsize * 0.83)
        else:
            axes[i].set_ylabel(None)
        
        axes[i].set_ylim(0, 1.0)
        axes[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
        axes[i].tick_params(axis='y', labelsize=fontsize * 0.75)

        if i != 0:
            axes[i].get_legend().remove()

        true_handles, true_labels = axes[1].get_legend_handles_labels()

        axes[0].legend(true_handles, true_labels, fontsize=fontsize * 0.4)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

df.head()

# add column with count of this model and language pair seen so far to each row in the dataframe
df['seen'] = df.groupby(['model', 'language_name']).cumcount() + 1

df_pivoted = df.pivot_table(index=['language_name', 'seen'], columns='model', values='is_correct', aggfunc='first').reset_index()

# Defining model groups with the specified structure for comparison
model_groups = {
    'Llama Group': ['Llama-2-7b-hf_8bit', None, 'Llama-2-7b-chat-hf_8bit'],
    'Tulu Group': ['Llama-2-7b-hf_8bit', 'Tulu-2-SFT_8bit', 'Tulu-2-DPO_8bit'],
    'Mistral Group': ['Mistral-7B-v0.1_8bit', 'Mistral-SFT-7b_8bit', 'zephyr-7b-beta_8bit'],
    'OpenChat Group': ['Mistral-7B-v0.1_8bit', 'OpenChat3.5-7B_8bit', 'starling-alpha-7b_8bit']
}

# For each group, calculate the difference in cfm_score from the base model to the tuned models
for group, models in model_groups.items():
    base_model = models[0]
    for i, model in enumerate(models[1:], start=1):  # Skip the base model
        if model:  # Check if model is not None
            diff_column_name = f'{group} Diff {i}'
            df_pivoted[diff_column_name] = df_pivoted[model] - df_pivoted[base_model]

# Rename Mesopotamian Arabic to Arabic
df_pivoted['language_name'] = df_pivoted['language_name'].replace('Mesopotamian Arabic', 'Arabic')

# lower case the language names
df_pivoted['language_name'] = df_pivoted['language_name'].str.lower()

# rename the language name column to "language"
df_pivoted = df_pivoted.rename(columns={'language_name': 'language'})

specified_languages.append('arabic')

# Filter the pivoted dataframe for the specified languages only (from TyDi QA)
df_pivoted_filtered = df_pivoted[df_pivoted['language'].isin([lang.lower() for lang in specified_languages])]

# Assuming 'df_pivoted' is already created from the CSV and processed as described previously
diff_columns = [col for col in df_pivoted_filtered.columns if 'Diff' in col]
df_melted = pd.melt(df_pivoted_filtered, id_vars=['language', 'seen'], value_vars=diff_columns, var_name='Group Difference', value_name='Difference')

# Specifying the figure dimensions and base font size
fig_width = 44
fig_height = 8
fontsize = 64

# Setting the color palette
base_palette = sns.color_palette("mako_r", n_colors=3)
llama_color = base_palette[-1]
tulu_mistral_openchat_colors = base_palette[1:3]

# Specifying the order of groups and languages for plotting
groups_ordered = ["Llama Group", "Tulu Group", "Mistral Group", "OpenChat Group"]
language_order = ["telugu", "bengali", "arabic", "swahili", "russian", "korean", "finnish", "english", "indonesian"]
new_titles = ["Llama to Llama Chat", "Llama to Tulu", "Mistral to Zephyr", "Mistral to Starling"]

# Creating the 1x4 horizontal figure layout
fig, axs = plt.subplots(1, 4, figsize=(fig_width, fig_height))

for i, group in enumerate(groups_ordered):
    group_data = df_melted[df_melted['Group Difference'].str.contains(group.split(' ')[0])]
    group_data['language'] = pd.Categorical(group_data['language'], categories=language_order, ordered=True)
    group_data_sorted = group_data.sort_values('language')
    
    if group == "Llama Group":
        sns.barplot(data=group_data_sorted, x='language', y='Difference', errorbar=('ci', 95), capsize=0, ax=axs[i], color=llama_color)
    else:
        sns.barplot(data=group_data_sorted, x='language', y='Difference', hue='Group Difference', errorbar=('ci', 95), capsize=0, ax=axs[i], palette=tulu_mistral_openchat_colors)
        if i != 3:  # Remove legends from the middle two figures
            axs[i].legend([],[], frameon=False)

    axs[i].set_title(new_titles[i], fontsize=fontsize)
    axs[i].set_ylim(-0.05, 0.35)
    axs[i].tick_params(labelsize=fontsize * 0.8)
    axs[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
    axs[i].tick_params(axis='y', labelsize=fontsize * 0.75)
    # add y ticks at 0, 0.15, and 0.3
    axs[i].set_yticks([0, 0.15, 0.3])

    axs[i].set_ylabel('Difference' if i == 0 else '', fontsize=fontsize * 0.75)
    axs[i].set_xlabel('')

# Adjusting the legend for the last plot
custom_legend = [plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[0], lw=4, label='SFT'), plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[1], lw=4, label='PT')]
axs[-1].legend(handles=custom_legend, title='Model Type', loc='upper left', fontsize=fontsize * 0.5, title_fontsize=fontsize * 0.5)

plt.tight_layout()
plt.savefig("./visualization/belebele.svg")
plt.savefig("./visualization/belebele.pdf")

# QWEN AND YI
df.head()

# add column with count of this model and language pair seen so far to each row in the dataframe
df['seen'] = df.groupby(['model', 'language_name']).cumcount() + 1

df_pivoted = df.pivot_table(index=['language_name', 'seen'], columns='model', values='is_correct', aggfunc='first').reset_index()

# Defining model groups with the specified structure for comparison
model_groups = {
    'Qwen Group': ['qwen1.5_8bit', None, 'qwen1.5_chat_8bit'],
    'Yi Group': ['yi-8b_8bit', None, 'yi-8b-chat_8bit'],
}

# For each group, calculate the difference in cfm_score from the base model to the tuned models
for group, models in model_groups.items():
    base_model = models[0]
    for i, model in enumerate(models[1:], start=1):  # Skip the base model
        if model:  # Check if model is not None
            diff_column_name = f'{group} Diff {i}'
            df_pivoted[diff_column_name] = df_pivoted[model] - df_pivoted[base_model]

# Rename Mesopotamian Arabic to Arabic
df_pivoted['language_name'] = df_pivoted['language_name'].replace('Mesopotamian Arabic', 'Arabic')

# lower case the language names
df_pivoted['language_name'] = df_pivoted['language_name'].str.lower()

# rename the language name column to "language"
df_pivoted = df_pivoted.rename(columns={'language_name': 'language'})

specified_languages.append('arabic')

# Filter the pivoted dataframe for the specified languages only (from TyDi QA)
df_pivoted_filtered = df_pivoted[df_pivoted['language'].isin([lang.lower() for lang in specified_languages])]

# Assuming 'df_pivoted' is already created from the CSV and processed as described previously
diff_columns = [col for col in df_pivoted_filtered.columns if 'Diff' in col]
df_melted = pd.melt(df_pivoted_filtered, id_vars=['language', 'seen'], value_vars=diff_columns, var_name='Group Difference', value_name='Difference')

# Specifying the figure dimensions and base font size
fig_width = 22
fig_height = 8
fontsize = 64

# Setting the color palette
base_palette = sns.color_palette("mako_r", n_colors=3)
llama_color = base_palette[-1]
tulu_mistral_openchat_colors = base_palette[1:3]

# Specifying the order of groups and languages for plotting
groups_ordered = ["Qwen Group", "Yi Group"]
language_order = ["telugu", "bengali", "arabic", "swahili", "russian", "korean", "finnish", "english", "indonesian"]
new_titles = ["Qwen to Qwen Chat", "Yi to Yi Chat"]

# Creating the 1x4 horizontal figure layout
fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_height))

for i, group in enumerate(groups_ordered):
    group_data = df_melted[df_melted['Group Difference'].str.contains(group.split(' ')[0])]
    group_data['language'] = pd.Categorical(group_data['language'], categories=language_order, ordered=True)
    group_data_sorted = group_data.sort_values('language')
    
    if group == "Llama Group" or group == "Qwen Group" or group == "Yi Group":
        sns.barplot(data=group_data_sorted, x='language', y='Difference', errorbar=('ci', 95), capsize=0, ax=axs[i], color=llama_color)
    else:
        sns.barplot(data=group_data_sorted, x='language', y='Difference', hue='Group Difference', errorbar=('ci', 95), capsize=0, ax=axs[i], palette=tulu_mistral_openchat_colors)
        if i != 3:  # Remove legends from the middle two figures
            axs[i].legend([],[], frameon=False)

    axs[i].set_title(new_titles[i], fontsize=fontsize)
    axs[i].set_ylim(-0.05, 0.35)
    axs[i].tick_params(labelsize=fontsize * 0.8)
    axs[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
    axs[i].tick_params(axis='y', labelsize=fontsize * 0.75)
    # add y ticks at 0, 0.15, and 0.3
    axs[i].set_yticks([0, 0.15, 0.3])

    axs[i].set_ylabel('Difference' if i == 0 else '', fontsize=fontsize * 0.75)
    axs[i].set_xlabel('')

# Adjusting the legend for the last plot
custom_legend = [plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[0], lw=4, label='SFT'), plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[1], lw=4, label='PT')]
axs[-1].legend(handles=custom_legend, title='Model Type', loc='upper left', fontsize=fontsize * 0.5, title_fontsize=fontsize * 0.5)

plt.tight_layout()
plt.savefig("./visualization/belebele_qwen_yi.svg")
plt.savefig("./visualization/belebele_qwen_yi.pdf")