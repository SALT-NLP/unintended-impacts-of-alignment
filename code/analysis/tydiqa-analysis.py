import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

TYDIQA_CSV = "./results/4_tydiqa/goldp.csv"

df = pd.read_csv(TYDIQA_CSV)

df = df[df["decoding_strat"] == "greedy"]
df = df[df["num_examples"] == "1-shot"]

sns.set_style('darkgrid')

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

# add column with count of this model and language pair seen so far to each row in the dataframe
df['seen'] = df.groupby(['model', 'language']).cumcount() + 1

df_pivoted = df.pivot_table(index=['language', 'seen'], columns='model', values='cfm_score', aggfunc='first').reset_index()

# Defining model groups with the specified structure for comparison
model_groups = {
    'Llama Group': ['Llama-2-7b-hf_8bit', None, 'Llama-2-7b-chat-hf_8bit'],
    'Tulu Group': ['Llama-2-7b-hf_8bit', 'Tulu-2-SFT_8bit', 'Tulu-2-DPO_8bit'],
    'Mistral Group': ['Mistral-7B-v0.1_8bit', 'Mistral-SFT-7B_8bit', 'zephyr-7b-beta_8bit'],
    'OpenChat Group': ['Mistral-7B-v0.1_8bit', 'OpenChat3.5-7B_8bit', 'Starling-7B-alpha_8bit']
}

# For each group, calculate the difference in cfm_score from the base model to the tuned models
for group, models in model_groups.items():
    base_model = models[0]
    for i, model in enumerate(models[1:], start=1):  # Skip the base model
        if model:  # Check if model is not None
            diff_column_name = f'{group} Diff {i}'
            df_pivoted[diff_column_name] = df_pivoted[model] - df_pivoted[base_model]

# Assuming 'df_pivoted' is already created from the CSV and processed as described previously
diff_columns = [col for col in df_pivoted.columns if 'Diff' in col]
df_melted = pd.melt(df_pivoted, id_vars=['language', 'seen'], value_vars=diff_columns, var_name='Group Difference', value_name='Difference')

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
    axs[i].set_ylim(-15, 10)
    axs[i].tick_params(labelsize=fontsize * 0.8)
    axs[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
    axs[i].tick_params(axis='y', labelsize=fontsize * 0.75)
    axs[i].set_ylabel('Difference' if i == 0 else '', fontsize=fontsize * 0.75)
    axs[i].set_xlabel('')

# Adjusting the legend for the last plot
custom_legend = [plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[0], lw=4, label='SFT'), plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[1], lw=4, label='PT')]
axs[-1].legend(handles=custom_legend, title='Model Type', loc='lower right', fontsize=fontsize * 0.5, title_fontsize=fontsize * 0.5)

plt.tight_layout()
plt.savefig("./visualization/tydiqa.svg")
plt.savefig("./visualization/tydiqa.pdf")

# QWEN AND YI GROUPS
# add column with count of this model and language pair seen so far to each row in the dataframe
df['seen'] = df.groupby(['model', 'language']).cumcount() + 1

df_pivoted = df.pivot_table(index=['language', 'seen'], columns='model', values='cfm_score', aggfunc='first').reset_index()

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

df_pivoted.head()

# Assuming 'df_pivoted' is already created from the CSV and processed as described previously
diff_columns = [col for col in df_pivoted.columns if 'Diff' in col]
df_melted = pd.melt(df_pivoted, id_vars=['language', 'seen'], value_vars=diff_columns, var_name='Group Difference', value_name='Difference')

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
    axs[i].set_ylim(-15, 10)
    axs[i].tick_params(labelsize=fontsize * 0.8)
    axs[i].tick_params(axis='x', labelsize=fontsize * 0.6, rotation=45)
    axs[i].tick_params(axis='y', labelsize=fontsize * 0.75)
    axs[i].set_ylabel('Difference' if i == 0 else '', fontsize=fontsize * 0.75)
    axs[i].set_xlabel('')

# Adjusting the legend for the last plot
custom_legend = [plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[0], lw=4, label='SFT'), plt.Line2D([0], [0], color=tulu_mistral_openchat_colors[1], lw=4, label='PT')]
axs[-1].legend(handles=custom_legend, title='Model Type', loc='lower right', fontsize=fontsize * 0.5, title_fontsize=fontsize * 0.5)

plt.tight_layout()
plt.savefig("./visualization/tydiqa_qwen_yi.svg")
plt.savefig("./visualization/tydiqa_qwen_yi.pdf")