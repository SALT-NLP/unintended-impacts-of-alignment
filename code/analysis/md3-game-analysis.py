import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import matplotlib.patches as mpatches

# Load the dataset
file_path = './results/2_md3game/md3game.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# For Qwen-Chat we must use the chat template, all other models do not use it
# Filter the data
good_qwen_chat = data[(data['model'] == 'Qwen/Qwen1.5-7B-Chat') & (data['using_chat_template'] == True)]
data = data[data['model'] != 'Qwen/Qwen1.5-7B-Chat']

data = data[data['using_chat_template'] == False]
data = pd.concat([data, good_qwen_chat])

# Grouping the data by 'model', 'prompt_mode', and 'dialect'
grouped_data = data.groupby(['model', 'prompt_mode', 'dialect'])

# Calculating the average values for 'chose_correct_answer_num' and 'generated_correct_answer_num'
average_values = grouped_data[['chose_correct_answer_num', 'generated_correct_answer_num']].mean()

average_values.reset_index()

# Re-grouping the data for 'prompt_mode' = 'no_answers' by 'model' and 'dialect'
filtered_data_no_answers = data[data['prompt_mode'] == 'no_answers']
grouped_data_no_answers_by_dialect = filtered_data_no_answers.groupby(['model', 'dialect'])

# Calculating the average value for 'chose_correct_answer_num' for each dialect
average_chose_correct_answer_no_answers_by_dialect = grouped_data_no_answers_by_dialect['chose_correct_answer_num'].mean()

# Pivoting the table to have a column for each dialect
pivot_table_dialects = average_chose_correct_answer_no_answers_by_dialect.unstack()

pivot_table_dialects.reset_index()

sns.set(style="darkgrid")

def generate_no_answer_plot(data, model_pairs, model_names, fontsize=24):
    filtered_data_no_answers = data[data['prompt_mode'] == 'no_answers']

    # Replacing the dialect codes with their full names
    filtered_data_no_answers.loc[filtered_data_no_answers['dialect'] == 'en_us', 'dialect'] = 'USA'
    filtered_data_no_answers.loc[filtered_data_no_answers['dialect'] == 'en_in', 'dialect'] = 'India'
    filtered_data_no_answers.loc[filtered_data_no_answers['dialect'] == 'en_ng', 'dialect'] = 'Nigeria'

    # Re-grouping the data for 'prompt_mode' = 'no_answers' by 'model' and 'dialect' for 'generated_correct_answer_num'
    grouped_data_no_answers_by_dialect_generated = filtered_data_no_answers.groupby(['model', 'dialect'])

    # Calculating the average value for 'generated_correct_answer_num' for each dialect
    average_generated_correct_answer_no_answers_by_dialect = grouped_data_no_answers_by_dialect_generated['generated_correct_answer_num'].mean()

    # Pivoting the table to have a column for each dialect for 'generated_correct_answer_num'
    pivot_table_dialects_generated = average_generated_correct_answer_no_answers_by_dialect.unstack()

    # Creating a plot for each pair for 'generated_correct_answer_num'
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()

    for i, ((model1, model2), (model_name1, model_name2)) in enumerate(zip(model_pairs, model_names)):
        ax = axes[i]
        pivot_table_dialects_generated.loc[[model1, model2]].T.plot(kind='bar', ax=ax)
        ax.set_title(f'{model_name1} vs {model_name2}', fontsize=fontsize)
        # ax.set_xlabel('Dialect', fontsize=fontsize* 0.83)
        ax.set_xlabel(None)
        ax.set_ylabel('Avg. Correct Answers', fontsize=fontsize* 0.83)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Rotate x-labels by 0 degrees
        ax.set_ylim(0, 0.5)
        
        ax.legend([model_name1, model_name2])

    plt.tight_layout()
    # plt.show()
    plt.savefig("./visualization/md3_comparison.pdf", bbox_inches='tight', pad_inches=0)

sns.set(style="darkgrid")

def compute_bootstrap_ci(data, model_column, answer_column, alpha=0.05):
    """
    Compute bootstrap confidence intervals for the rate of correct answers for each model.

    Parameters:
    data (pd.DataFrame): The dataset containing the model performance data.
    model_column (str): The column name for the model IDs.
    answer_column (str): The column name for the correct answer rates.
    alpha (float): The significance level for the confidence intervals.

    Returns:
    dict: A dictionary containing the model IDs as keys and the confidence intervals as values.
    """

    model_ids = data[model_column].unique()
    bootstrap_cis = {}

    for model_id in model_ids:
        for dialect, name in zip(['en_us', 'en_in', 'en_ng'], ['USA', 'India', 'Nigeria']):
            model_data = data[(data[model_column] == model_id) & (data["dialect"] == dialect) & (data["prompt_mode"] == 'no_answers')][answer_column].to_numpy()
            model_data = (model_data,)
            res = stats.bootstrap(model_data, np.mean, confidence_level=1-alpha)
            # ci = stats.t.interval(1 - alpha, len(bootstrap_samples[0]) - 1, loc=np.mean(bootstrap_samples[0]), scale=stats.sem(bootstrap_samples[0]))
            bootstrap_cis[(model_id, name)] = res.confidence_interval

    return bootstrap_cis

def generate_comparison_plots(dataframe, model_groups, plot_titles, xlabel, ylabel, fontsize=24, grid_size=(2, 2), alpha=0.05, output_file="./visualization/md3_comparison2.pdf", base_palette=None, category_col='dialect', value_col='generated_correct_answer_num', model_col='model'):
    """
    Generate a grid of bar plots with a consistent legend showing all model types,
    and ensure bar colors match the corresponding model type, regardless of presence.
    """

    dialect_order = ['India', 'Nigeria', 'USA']

    if base_palette is None:
        base_palette = sns.color_palette("mako_r", n_colors=3)

    filtered_data_no_answers = dataframe[dataframe['prompt_mode'] == 'no_answers'].copy()
    mapping = {'en_us': 'USA', 'en_in': 'India', 'en_ng': 'Nigeria'}
    filtered_data_no_answers[category_col] = filtered_data_no_answers[category_col].map(mapping)

    fig_width = 11 * grid_size[1]
    fig_height = 8 * grid_size[0]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(fig_width, fig_height))
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]

    legend_labels = ['Base', 'SFT', 'PT']  # Labels for all possible models
    legend_handles = [mpatches.Patch(color=base_palette[i], label=label) for i, label in enumerate(legend_labels)]

    for i, (model_group, title) in enumerate(zip(model_groups, plot_titles)):
        df = filtered_data_no_answers[filtered_data_no_answers[model_col].isin(model_groups[model_group])]
        
        # Create a color map for the models to ensure consistency
        color_map = dict(zip(model_groups[model_group], base_palette))
        
        # If a model is missing, create a new palette excluding the color of the missing model
        current_palette = [color_map[model] for model in model_groups[model_group] if model is not None]

        df['model_order'] = df[model_col].apply(lambda x: model_groups[model_group].index(x) if x in model_groups[model_group] else -1)
        df.sort_values('model_order', inplace=True)

        sns.barplot(data=df, x=category_col, y=value_col, hue=model_col, ax=axes[i], palette=current_palette, errorbar=('ci', 95), order=dialect_order)
        axes[i].set_title(title, fontsize=fontsize)
        axes[i].set_xlabel(xlabel, fontsize=fontsize)
        if i % grid_size[1] == 0:
            axes[i].set_ylabel(ylabel, fontsize=fontsize)
        else:
            axes[i].set_ylabel('')

        axes[i].tick_params(labelsize=fontsize * 0.8)
        axes[i].set_ylim(0, 0.5)

        # Set the legend on the first subplot
        if i == 0:
            axes[i].legend(handles=legend_handles, fontsize=fontsize * 0.65)
        else:
            if axes[i].get_legend():
                axes[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()

# Adjust dialect names in the dataset to match those expected by compute_bootstrap_ci
# Define the groups with corresponding model names from the dataset
model_groups = {
    'Llama Group': ['meta-llama/Llama-2-7b-hf', None, 'meta-llama/Llama-2-7b-chat-hf'],
    'Tulu Group': ['meta-llama/Llama-2-7b-hf', 'allenai/tulu-2-7b', 'allenai/tulu-2-dpo-7b'],
    'Mistral Group': ['mistralai/Mistral-7B-v0.1', 'HuggingFaceH4/mistral-7b-sft-beta', 'HuggingFaceH4/zephyr-7b-beta'],
    'OpenChat Group': ['mistralai/Mistral-7B-v0.1', 'openchat/openchat_3.5', 'berkeley-nest/Starling-LM-7B-alpha']
}

# Prepare plot_titles based on the model_groups keys
plot_titles = ['Llama to Llama Chat', 'Llama to Tulu', 'Mistral to Zephyr', 'Mistral to Starling']

# Calling the function to plot the model comparisons
generate_comparison_plots(data, model_groups, plot_titles, '', 'Correct Rate', fontsize=64, grid_size=(2, 2), alpha=0.05, base_palette=None, output_file="./visualization/md3_comparison.svg")
generate_comparison_plots(data, model_groups, plot_titles, '', 'Correct Rate', fontsize=64, grid_size=(2, 2), alpha=0.05, base_palette=None, output_file="./visualization/md3_comparison.pdf")

# Adjust dialect names in the dataset to match those expected by compute_bootstrap_ci
# Define the groups with corresponding model names from the dataset
model_groups = {
    'Qwen Group': ['Qwen/Qwen1.5-7B', None, 'Qwen/Qwen1.5-7B-Chat'],
    'Yi Group': ['01-ai/Yi-6B', None, '01-ai/Yi-6B-Chat'],
}

# Prepare plot_titles based on the model_groups keys
plot_titles = ['Qwen to Qwen Chat', 'Yi to Yi Chat']

# Calling the function to plot the model comparisons
generate_comparison_plots(data, model_groups, plot_titles, '', 'Correct Rate', fontsize=64, grid_size=(1,2), alpha=0.05, base_palette=None, output_file="./visualization/md3_comparison_qwen_yi.svg")
generate_comparison_plots(data, model_groups, plot_titles, '', 'Correct Rate', fontsize=64, grid_size=(1,2), alpha=0.05, base_palette=None, output_file="./visualization/md3_comparison_qwen_yi.pdf")