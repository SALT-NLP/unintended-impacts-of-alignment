import pandas as pd

# Load the dataset
file_path = './outputs/ask-reddit-reward-testing/starling-rm-7b-alpha-reformatted.csv'
data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure and contents
data.head()

us_column_name = "United States of America"

# Redefining the country columns
country_columns = data.columns[9:]  # Country columns start from the 10th column 
 
# Reinitializing the win count dictionaries  
positive_wins = {country: 0 for country in country_columns} 
negative_wins = {country: 0 for country in country_columns} 
total_rank = {country: 0 for country in country_columns} 
 
# Recalculating the wins for each country 
for index, row in data.iterrows(): 
    us_score = row[us_column_name] 
    sentiment = row["Sentiment"] 
    negative = (sentiment == "negative") 
 
    sorted_row = row[country_columns].sort_values(ascending=negative) 

    for country in country_columns:
        country_score = row[country]

        # Winning criteria based on sentiment
        if sentiment == "positive" and country_score > us_score:
            positive_wins[country] += 1
        elif sentiment == "negative" and country_score < us_score:
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

win_counts_df.head()  # Displaying the first few rows of the result

print('num countries:', len(win_counts_df['Country'].unique()))

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

# Calculate precision, recall, and F1 score
precision = win_counts_df['Positive Win Rate']
recall = win_counts_df['Negative Win Rate']
f1_score = 2 * (precision * recall) / (precision + recall)

win_counts_df['F1 Score'] = f1_score
sorted_win_counts = win_counts_df.sort_values(by='F1 Score', ascending=True)  # Displaying the first few rows of the updated result

win_counts_df['Total Wins'] = win_counts_df['Positive Wins'] + win_counts_df['Negative Wins']
total_prompts = data.shape[0]
win_counts_df['Total Win Rate'] = win_counts_df['Total Wins'] / total_prompts

sorted_win_counts = win_counts_df.sort_values(by='Mean Rank', ascending=True)  # Displaying the first few rows of the updated result

sorted_win_counts.head(21)  # Displaying the first few rows of the sorted result

sorted_win_counts = win_counts_df.sort_values(by='F1 Score', ascending=True)  # Displaying the first few rows of the updated result

sorted_win_counts.head(21)  # Displaying the first few rows of the sorted result

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
win_counts_df["code"] = win_counts_df["Country"].map(country_code_map)

POPULATION_DATA = './data/countries/WPP2022_TotalPopulationBySex.csv'
POPULATION_CUTOFF = 250000

pop_data = pd.read_csv(POPULATION_DATA)

data_2021 = pop_data[(pop_data['Time'] == 2021) & (pop_data['PopTotal'] > (POPULATION_CUTOFF / 1000.0))]
iso3_codes_filter = data_2021['ISO3_code'].dropna().unique().tolist()

# Filter out countries that have a small population
win_counts_df = win_counts_df[win_counts_df["code"].isin(iso3_codes_filter)]

# Add population data to the DataFrame by merging it with only the relevant PopTotal column
win_counts_df = win_counts_df.merge(data_2021[['ISO3_code', 'PopTotal']], left_on='code', right_on='ISO3_code')

# Print count of remaining countries
print('num countries after filtering:', len(win_counts_df['Country'].unique()))

import plotly.graph_objects as go
import plotly.io as pio   
pio.kaleido.scope.mathjax = None

def create_winrate_cloropleth(df, column_name="Total Win Rate", colorbar_title="F1 Score"):
    country_code_map = get_country_code_map()
    df["code"] = df["Country"].map(country_code_map)

    fig = go.Figure(data=go.Choropleth(
        locations = df['code'],
        z = df[column_name],
        text = df['Country'],
        # colorscale = 'Blues',
        autocolorscale=False,
        reversescale=False, # True
        marker_line_color='darkgray',
        marker_line_width=0.0,
        colorbar_tickprefix = '',
        colorbar_title = colorbar_title,
        colorbar=dict(
            x=1.0,
            len=0.88,
            y=0.5,
            tickformat=".1f",
            title_font=dict(size=32),  # Increase font size for the title
            tickfont=dict(size=24),  # Increase font size for the tick labels
        ),
        zmin=0,  # Set the minimum value of the colorbar range to 0
        zmax=220,  # Set the maximum value of the colorbar range to 1
    ))

    fig.update_layout(
        # title_text='Reward for each country (Starling 7B)',
        geo=dict(
            # showframe=False,
            showcoastlines=False,
            # projection_type='equirectangular',
            projection_type='natural earth',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=800,  # Width of the figure in pixels
        height=400,  # Height of the figure in pixels
    )
    fig.write_image("./visualization/rewards_askreddit.pdf", format='pdf', width=800, height=400)

    # fig.show()

create_winrate_cloropleth(win_counts_df, "Mean Rank", "Mean Rank")