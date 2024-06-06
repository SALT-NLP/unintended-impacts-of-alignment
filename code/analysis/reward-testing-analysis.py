country_file = "./data/countries/countries.geojson"
results = "./outputs/reward-testing/where_from_starling-rm-7b-alpha.csv"

import json
import time
with open(country_file) as f:
    counties = json.load(f)

import pandas as pd
df = pd.read_csv(results)

def get_country_code_map():
    country_code_map = {}
    for country in counties['features']:
        country_code_map[country['properties']['ADMIN']] = country['properties']['ISO_A3']
    return country_code_map

import plotly.graph_objects as go
import plotly.io as pio   
pio.kaleido.scope.mathjax = None

def create_reward_cloropleth(df):
    country_code_map = get_country_code_map()
    df["code"] = df["country"].map(country_code_map)
    max_reward = df["reward"].max()
    min_reward = df["reward"].min()

    fig = go.Figure(data=go.Choropleth(
        locations = df['code'],
        z = df['reward'],
        text = df['country'],
        # colorscale = 'Blues',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.0,
        colorbar_tickprefix = '',
        colorbar_title = 'Reward',
        colorbar=dict(
            x=1.0,
            len=0.95,
            y=0.5,
            tickformat=".1f",
            title_font=dict(size=32),  # Increase font size for the title
            tickfont=dict(size=24),  # Increase font size for the tick labels
            tickvals=[max_reward-0.025, min_reward+0.025],  # No tick labels
            ticktext=['Higher', 'Lower'],  # No tick labels
            title_side='right',
        ),
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
    time.sleep(1.1)
    fig.write_image("./visualization/where_from_rewards.pdf", format='pdf', width=800, height=400)
    # fig.write_image("../visualization/rewards.png", format='png', width=800, height=400)

    # fig.show()

create_reward_cloropleth(df)