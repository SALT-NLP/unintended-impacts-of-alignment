import pandas as pd
import numpy as np

LANGID = "./outputs/tulu-sft-langid/langid.csv"

df = pd.read_csv(LANGID)

df = df[df['agreed_lang_text'] != 'unknown']
df_user = df[df['role'] == 'user']
df_assistant = df[df['role'] == 'assistant']

import matplotlib.pyplot as plt

# Count the occurrences of each value in df_prompt
value_counts = df_user['agreed_lang_text'].value_counts()

# Calculate the proportion of each value
proportions = value_counts / len(df_user)

# Pivot table of language counts and proportion
pivot = df['agreed_lang_text'].value_counts(normalize=True)

language_mapping = {
    'es': 'Spanish',
    'de': 'German',
    'fa': 'Farsi',
    'fr': 'French',
    'hi': 'Hindi',
    'ja': 'Japanese',
    'te': 'Telugu',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ar': 'Arabic',
    'ta': 'Tamil',
    'pl': 'Polish',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'ml': 'Malayalam',
    'he': 'Hebrew',
    'mr': 'Marathi',
    'ca': 'Catalan',
    'bn': 'Bengali',
    'ko': 'Korean',
    'id': 'Indonesian',
    'nl': 'Dutch',
    'vi': 'Vietnamese',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'ru': 'Russian',
    'tl': 'Tagalog',
    'th': 'Thai',
    'fi': 'Finnish',
    'bg': 'Bulgarian',
    'ro': 'Romanian',
    'hr': 'Croatian',
    'cs': 'Czech',
    'el': 'Greek',
    'kn': 'Kannada',
    'da': 'Danish',
    'ne': 'Nepali',
    'lt': 'Lithuanian',
    'no': 'Norwegian',
    'sk': 'Slovak',
    'sw': 'Swahili',
    'so': 'Somali',
    'et': 'Estonian',
    'uk': 'Ukrainian',
    'lv': 'Latvian',
    'af': 'Afrikaans',
    'mk': 'Macedonian',
    'sl': 'Slovenian',
    'hu': 'Hungarian',
    'sq': 'Albanian',
    'cy': 'Welsh',
}

# Map language codes to language name in the pivot table
pivot = pivot.rename(index=language_mapping)

pivot.sort_values(ascending=False, axis=0)

print()
print("Tulu SFT Language Proportions")
print(pivot)

pivot.to_csv("./results/5_langid/tulu-sft-languages.csv")

# Pivot table of language counts 
pivot = df.pivot_table(index='agreed_lang_text', aggfunc='size', fill_value=0)

# Map language codes to language name in the pivot table
pivot = pivot.rename(index=language_mapping)

pivot.sort_values(ascending=False, axis=0)
pivot.to_csv("./results/5_langid/tulu-sft-languages-count.csv")