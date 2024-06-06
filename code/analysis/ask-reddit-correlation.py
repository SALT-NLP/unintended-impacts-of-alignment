from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
import numpy as np

countries_filter = ['Afghanistan',
 'Angola',
 'Albania',
 'United Arab Emirates',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Burundi',
 'Belgium',
 'Benin',
 'Burkina Faso',
 'Bangladesh',
 'Bulgaria',
 'Bahrain',
 'The Bahamas',
 'Bosnia and Herzegovina',
 'Belarus',
 'Belize',
 'Bolivia',
 'Brazil',
 'Barbados',
 'Brunei',
 'Bhutan',
 'Botswana',
 'Central African Republic',
 'Canada',
 'Switzerland',
 'Chile',
 'China',
 'Ivory Coast',
 'Cameroon',
 'Democratic Republic of the Congo',
 'Republic of Congo',
 'Colombia',
 'Comoros',
 'Cape Verde',
 'Costa Rica',
 'Cuba',
 'Cyprus',
 'Czech Republic',
 'Germany',
 'Djibouti',
 'Denmark',
 'Dominican Republic',
 'Algeria',
 'Ecuador',
 'Egypt',
 'Eritrea',
 'Spain',
 'Estonia',
 'Ethiopia',
 'Finland',
 'Fiji',
 'France',
 'Gabon',
 'United Kingdom',
 'Georgia',
 'Ghana',
 'Guinea',
 'Gambia',
 'Guinea Bissau',
 'Equatorial Guinea',
 'Greece',
 'Guatemala',
 'Guyana',
 'Hong Kong S.A.R.',
 'Honduras',
 'Croatia',
 'Haiti',
 'Hungary',
 'Indonesia',
 'India',
 'Ireland',
 'Iran',
 'Iraq',
 'Iceland',
 'Israel',
 'Italy',
 'Jamaica',
 'Jordan',
 'Japan',
 'Kazakhstan',
 'Kenya',
 'Kyrgyzstan',
 'Cambodia',
 'South Korea',
 'Kuwait',
 'Laos',
 'Lebanon',
 'Liberia',
 'Libya',
 'Sri Lanka',
 'Lesotho',
 'Lithuania',
 'Luxembourg',
 'Latvia',
 'Macao S.A.R',
 'Morocco',
 'Moldova',
 'Madagascar',
 'Maldives',
 'Mexico',
 'Macedonia',
 'Mali',
 'Malta',
 'Myanmar',
 'Montenegro',
 'Mongolia',
 'Mozambique',
 'Mauritania',
 'Mauritius',
 'Malawi',
 'Malaysia',
 'Namibia',
 'New Caledonia',
 'Niger',
 'Nigeria',
 'Nicaragua',
 'Netherlands',
 'Norway',
 'Nepal',
 'New Zealand',
 'Oman',
 'Pakistan',
 'Panama',
 'Peru',
 'Philippines',
 'Papua New Guinea',
 'Poland',
 'Puerto Rico',
 'North Korea',
 'Portugal',
 'Paraguay',
 'Palestine',
 'French Polynesia',
 'Qatar',
 'Romania',
 'Russia',
 'Rwanda',
 'Western Sahara',
 'Saudi Arabia',
 'Sudan',
 'South Sudan',
 'Senegal',
 'Singapore',
 'Solomon Islands',
 'Sierra Leone',
 'El Salvador',
 'Somalia',
 'Republic of Serbia',
 'Suriname',
 'Slovakia',
 'Slovenia',
 'Sweden',
 'Swaziland',
 'Syria',
 'Chad',
 'Togo',
 'Thailand',
 'Tajikistan',
 'Turkmenistan',
 'East Timor',
 'Trinidad and Tobago',
 'Tunisia',
 'Turkey',
 'Taiwan',
 'United Republic of Tanzania',
 'Uganda',
 'Ukraine',
 'Uruguay',
 'United States of America',
 'Uzbekistan',
 'Venezuela',
 'Vietnam',
 'Vanuatu',
 'Yemen',
 'South Africa',
 'Zambia',
 'Zimbabwe']

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
POPULATION_DATA = './data/countries/WPP2022_TotalPopulationBySex.csv'
POPULATION_CUTOFF = 250000

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
POPULATION_DATA = './data/countries/WPP2022_TotalPopulationBySex.csv'
POPULATION_CUTOFF = 250000

# Load Data
LLAMA_CSV = './outputs/ask-reddit-perplexity-testing/Llama-2-7b-hf-reformatted.csv'
LLAMA_CHAT_CSV = './outputs/ask-reddit-perplexity-testing/Llama-2-7b-chat-hf-reformatted.csv'
TULU_SFT_CSV = './outputs/ask-reddit-perplexity-testing/Tulu-2-SFT_8bit-reformatted.csv'
TULU_DPO_CSV = './outputs/ask-reddit-perplexity-testing/Tulu-2-DPO_8bit-reformatted.csv'
MISTRAL_CSV = './outputs/ask-reddit-perplexity-testing/Mistral-7B-v0.1-reformatted.csv'
MISTRAL_SFT_CSV = './outputs/ask-reddit-perplexity-testing/Mistral-SFT-7B_8bit-reformatted.csv'
ZEPHYR_CSV = './outputs/ask-reddit-perplexity-testing/zephyr-7b-beta_8bit-reformatted.csv'
OPEN_CHAT_CSV = './outputs/ask-reddit-perplexity-testing/OpenChat3.5_8bit-reformatted.csv'
STARLING_CSV = './outputs/ask-reddit-perplexity-testing/starling-alpha-7b-reformatted.csv'
STARLING_RM_CSV = './outputs/ask-reddit-reward-testing/starling-rm-7b-alpha-reformatted.csv'

llama_df = pd.read_csv(LLAMA_CSV)
llama_chat_df = pd.read_csv(LLAMA_CHAT_CSV)
tulu_sft_df = pd.read_csv(TULU_SFT_CSV)
tulu_dpo_df = pd.read_csv(TULU_DPO_CSV)
mistral_df = pd.read_csv(MISTRAL_CSV)
mistral_sft_df = pd.read_csv(MISTRAL_SFT_CSV)
zephyr_df = pd.read_csv(ZEPHYR_CSV)
open_chat_df = pd.read_csv(OPEN_CHAT_CSV)
starling_df = pd.read_csv(STARLING_CSV)
starling_rm_df = pd.read_csv(STARLING_RM_CSV)
starling_rm_df = pd.read_csv(STARLING_RM_CSV)

llama_df_filtered = llama_df.drop(columns=[col for col in llama_df.columns[10:] if col not in countries_filter])
llama_chat_df_filtered = llama_chat_df.drop(columns=[col for col in llama_chat_df.columns[10:] if col not in countries_filter])
tulu_sft_df_filtered = tulu_sft_df.drop(columns=[col for col in tulu_sft_df.columns[10:] if col not in countries_filter])
tulu_dpo_df_filtered = tulu_dpo_df.drop(columns=[col for col in tulu_dpo_df.columns[10:] if col not in countries_filter])
mistral_df_filtered = mistral_df.drop(columns=[col for col in mistral_df.columns[10:] if col not in countries_filter])
mistral_sft_df_filtered = mistral_sft_df.drop(columns=[col for col in mistral_sft_df.columns[10:] if col not in countries_filter])
zephyr_df_filtered = zephyr_df.drop(columns=[col for col in zephyr_df.columns[10:] if col not in countries_filter])
open_chat_df_filtered = open_chat_df.drop(columns=[col for col in open_chat_df.columns[10:] if col not in countries_filter])
starling_df_filtered = starling_df.drop(columns=[col for col in starling_df.columns[10:] if col not in countries_filter])
starling_rm_df_filtered = starling_rm_df.drop(columns=[col for col in starling_rm_df.columns[9:] if col not in countries_filter])

us_column_name = "United States of America"

def compute_stats(data, start_column=10, score_type='perplexity'):
    # Redefining the country columns
    country_columns = data.columns[start_column:]  # Country columns start from the 11th column 
    
    reverse = False
    if score_type == 'perplexity':
        reverse = True

    # Reinitializing the win count dictionaries  
    positive_wins = {country: 0 for country in country_columns} 
    negative_wins = {country: 0 for country in country_columns} 
    total_rank = {country: 0 for country in country_columns} 
    
    # Recalculating the wins for each country 
    for index, row in data.iterrows(): 
        us_score = row[us_column_name] 
        sentiment = row["Sentiment"] 
        negative = (sentiment == "negative") 
        if reverse: 
            negative = not negative
    
        sorted_row = row[country_columns].sort_values(ascending=negative) 

        for country in country_columns:
            country_score = row[country]

            # Winning criteria based on sentiment
            if sentiment == "positive":
                if not reverse and country_score > us_score:
                    positive_wins[country] += 1
                elif reverse and country_score < us_score:
                    positive_wins[country] += 1
            elif sentiment == "negative":
                if not reverse and country_score < us_score:
                    negative_wins[country] += 1
                elif reverse and country_score > us_score:
                    negative_wins[country] += 1
                
        for i in range(len(sorted_row)):
            country = sorted_row.index[i]
            total_rank[country] += i + 1

    # Creating a DataFrame to display the results
    win_counts_df = pd.DataFrame({
        "Country": country_columns,
        "Positive Wins": [positive_wins[country] for country in country_columns],
        "Negative Wins": [negative_wins[country] for country in country_columns],
        "Total Rank": [total_rank[country] for country in country_columns]
    })

    # Removing the 'add_article' row from the DataFrame
    win_counts_df = win_counts_df[win_counts_df['Country'] != 'add_article']

    # Counting the number of positive and negative prompts
    total_positive_prompts = data[data['Sentiment'] == 'positive'].shape[0]
    total_negative_prompts = data[data['Sentiment'] == 'negative'].shape[0]
    total_prompts = len(data)

    # Calculating win rates
    win_counts_df['Positive Win Rate'] = win_counts_df['Positive Wins'] / total_positive_prompts
    win_counts_df['Negative Win Rate'] = win_counts_df['Negative Wins'] / total_negative_prompts
    win_counts_df['Mean Rank'] = win_counts_df['Total Rank'] / (total_prompts)

    return win_counts_df

llama_stats = compute_stats(llama_df_filtered)
llama_chat_stats = compute_stats(llama_chat_df_filtered)
tulu_sft_stats = compute_stats(tulu_sft_df_filtered)
tulu_dpo_stats = compute_stats(tulu_dpo_df_filtered)
mistral_stats = compute_stats(mistral_df_filtered)
mistral_sft_stats = compute_stats(mistral_sft_df_filtered)
zephyr_stats = compute_stats(zephyr_df_filtered)
open_chat_stats = compute_stats(open_chat_df_filtered)
starling_stats = compute_stats(starling_df_filtered)
starling_rm_stats = compute_stats(starling_rm_df_filtered, start_column=9, score_type='reward')

country_file = "./data/countries/countries.geojson"

import json
with open(country_file) as f:
    counties = json.load(f)

def get_country_code_map():
    country_code_map = {}
    for country in counties['features']:
        country_code_map[country['properties']['ADMIN']] = country['properties']['ISO_A3']
    return country_code_map

country_code_map = get_country_code_map()
pop_data = pd.read_csv(POPULATION_DATA)
data_2021 = pop_data[(pop_data['Time'] == 2021) & (pop_data['PopTotal'] > (POPULATION_CUTOFF / 1000.0))]
iso3_codes_filter = data_2021['ISO3_code'].dropna().unique().tolist()

def filter_by_country_population(data):
    data["code"] = data["Country"].map(country_code_map)

    win_counts_df = data[data["code"].isin(iso3_codes_filter)]

    return win_counts_df

llama_stats_filtered = filter_by_country_population(llama_stats)
llama_chat_stats_filtered = filter_by_country_population(llama_chat_stats)
tulu_sft_stats_filtered = filter_by_country_population(tulu_sft_stats)
tulu_dpo_stats_filtered = filter_by_country_population(tulu_dpo_stats)
mistral_stats_filtered = filter_by_country_population(mistral_stats)
mistral_sft_stats_filtered = filter_by_country_population(mistral_sft_stats)
zephyr_stats_filtered = filter_by_country_population(zephyr_stats)
open_chat_stats_filtered = filter_by_country_population(open_chat_stats)
starling_stats_filtered = filter_by_country_population(starling_stats)
starling_rm_stats_filtered = filter_by_country_population(starling_rm_stats)
country_order = starling_rm_stats_filtered['Country'].to_list()

# Sort the dataframes by the country order
llama_stats_filtered = llama_stats_filtered.set_index('Country').loc[country_order].reset_index()
llama_chat_stats_filtered = llama_chat_stats_filtered.set_index('Country').loc[country_order].reset_index()
tulu_sft_stats_filtered = tulu_sft_stats.set_index('Country').loc[country_order].reset_index()
tulu_dpo_stats_filtered = tulu_dpo_stats.set_index('Country').loc[country_order].reset_index()
mistral_stats_filtered = mistral_stats_filtered.set_index('Country').loc[country_order].reset_index()
mistral_sft_stats_filtered = mistral_sft_stats.set_index('Country').loc[country_order].reset_index()
zephyr_stats_filtered = zephyr_stats.set_index('Country').loc[country_order].reset_index()
open_chat_stats_filtered = open_chat_stats_filtered.set_index('Country').loc[country_order].reset_index()
starling_stats_filtered = starling_stats_filtered.set_index('Country').loc[country_order].reset_index()
starling_rm_stats_filtered = starling_rm_stats_filtered.set_index('Country').loc[country_order].reset_index()

llama_ranks = llama_stats_filtered['Mean Rank'].rank()
llama_chat_ranks = llama_chat_stats_filtered['Mean Rank'].rank()
tulu_sft_ranks = tulu_sft_stats_filtered['Mean Rank'].rank()
tulu_dpo_ranks = tulu_dpo_stats_filtered['Mean Rank'].rank()
mistral_ranks = mistral_stats_filtered['Mean Rank'].rank()
mistral_sft_ranks = mistral_sft_stats_filtered['Mean Rank'].rank()
zephyr_ranks = zephyr_stats_filtered['Mean Rank'].rank()
open_chat_ranks = open_chat_stats_filtered['Mean Rank'].rank()
starling_ranks = starling_stats_filtered['Mean Rank'].rank()
starling_rm_ranks = starling_rm_stats_filtered['Mean Rank'].rank()

df = pd.DataFrame({
    'Llama': llama_ranks,
    'Llama Chat': llama_chat_ranks,
    'Tulu SFT': tulu_sft_ranks,
    'Tulu DPO': tulu_dpo_ranks,
    'Mistral': mistral_ranks,
    'Mistral SFT': mistral_sft_ranks,
    'Zephyr': zephyr_ranks,
    'OpenChat': open_chat_ranks,
    'Starling LM': starling_ranks,
    'Starling RM': starling_rm_ranks
})

corr = df.corr(method = 'spearman')

# Create a discrete colormap
# First, generate the color list from the 'coolwarm' colormap
n_colors = 10
palette = sns.color_palette("coolwarm", n_colors=n_colors)

# Use LinearSegmentedColormap to create a new colormap from the existing one
color_list = palette.as_hex()  # Convert palette to hex format
cmap = LinearSegmentedColormap.from_list("custom_coolwarm", color_list, N=n_colors)

# sns heatmap with correlation only across the lower triangle
mask = np.tril(np.ones_like(corr, dtype=bool))  # Mask for lower triangle
sns.heatmap(corr, annot=True, cmap=cmap, mask=~mask, cbar_kws={'label': 'Spearman Rank Correlation Coefficient'}, square=True)

# Tight layout
plt.tight_layout()

# Save the figure
plt.savefig('./visualization/askreddit-correlation-heatmap.pdf')

# Based on Gallup Poll https://news.gallup.com/poll/472421/canada-britain-favored-russia-korea-least.aspx#:~:text=WASHINGTON%2C%20D.C.%20%2D%2D%20Americans%20continue,the%20least%20favorably%20reviewed%20country.
us_preferred_countries = ["Canada", "United Kingdom", "France", "Japan", "Germany", "Taiwan", "India", "Israel", "Ukraine", "Egypt", "Brazil", "Mexico", "Cuba", "Saudi Arabia", "Palestine", "Iraq", "Afghanistan", "China", "Iran", "Russia", "North Korea"]
us_preferred_countries_2017 = ["Canada", "United Kingdom", "Japan", "France", "Germany", "India", "Taiwan", "Israel", "Philippines", "Mexico", "Egypt", "Cuba", "China", "Saudi Arabia", "Russia", "Palestine", "Iraq", "Afghanistan", "Syria", "Iran", "North Korea"]

us_preferred_countries_2023_limited = ["Canada", "United Kingdom", "France", "Japan", "Germany", "Taiwan", "India", "Israel", "Egypt", "Mexico", "Cuba", "Saudi Arabia", "Palestine", "Iraq", "Afghanistan", "China", "Iran", "Russia", "North Korea"]
us_preferred_countries_2017_limited = ["Canada", "United Kingdom", "Japan", "France", "Germany", "India", "Taiwan", "Israel", "Mexico", "Egypt", "Cuba", "China", "Saudi Arabia", "Russia", "Palestine", "Iraq", "Afghanistan", "Iran", "North Korea"]

# Filter the dataframe to only include the USA preferred countries
starling_rm_df_filtered_USA = starling_rm_stats_filtered[starling_rm_stats_filtered['Country'].isin(us_preferred_countries)]
starling_rm_df_filtered_USA_2017 = starling_rm_stats_filtered[starling_rm_stats_filtered['Country'].isin(us_preferred_countries_2017)]

# Sort the dataframes by the country order
starling_rm_df_filtered_USA = starling_rm_df_filtered_USA.set_index('Country').loc[us_preferred_countries].reset_index()
starling_rm_df_filtered_USA_2017 = starling_rm_df_filtered_USA_2017.set_index('Country').loc[us_preferred_countries_2017].reset_index()

# Rank the countries by mean rank
starling_rm_df_filtered_USA_rank = starling_rm_df_filtered_USA['Mean Rank'].rank()
starling_rm_df_filtered_USA_2017_rank = starling_rm_df_filtered_USA_2017['Mean Rank'].rank()

# Compute the correlation between the starling rm ranks and the USA preferred countries ranks
corr_starling_rm_usa = starling_rm_df_filtered_USA_rank.corr(pd.Series(range(1, len(us_preferred_countries) + 1)), method='spearman')
corr_starling_rm_usa_2017 = starling_rm_df_filtered_USA_2017_rank.corr(pd.Series(range(1, len(us_preferred_countries_2017) + 1)), method='spearman')

# Get p-values for the correlation
from scipy.stats import spearmanr

print("2017 Spearman Correlation")
corr_2017 = spearmanr(starling_rm_df_filtered_USA_2017_rank, pd.Series(range(1, len(us_preferred_countries_2017) + 1)))
print(corr_2017)

print("2023 Spearman Correlation")
corr_2023 = spearmanr(starling_rm_df_filtered_USA_rank, pd.Series(range(1, len(us_preferred_countries) + 1)))
print(corr_2023)

with open('./results/7_askreddit/correlations.txt', 'w') as f:
    f.write(f"2017 Spearman Correlation\n")
    f.write(str(corr_2017))
    f.write("\n\n")
    f.write(f"2023 Spearman Correlation\n")
    f.write(str(corr_2023))

print(starling_rm_df_filtered_USA.sort_values(by='Mean Rank', ascending=True).head(21))
starling_rm_df_filtered_USA.to_csv('./results/7_askreddit/starling_rm_df_filtered_USA.csv')