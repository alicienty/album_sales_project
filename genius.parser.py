import pandas as pd
import lyricsgenius
import time
import os

os.makedirs("saves", exist_ok=True)

# загрузка датасета

df = pd.read_csv("df_to_parse_3.csv")

df['track_name'] = df['track_name'].astype(str).str.strip()
df['artist_name'] = df['artist_name'].astype(str).str.strip()

# API

genius = lyricsgenius.Genius("-")

genius.remove_section_headers = True
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]

# поиск

def get_lyrics(row):
    try:
        song = genius.search_song(row['track_name'], row['artist_name'])
        if song:
            return song.lyrics
    except Exception as e:
        print(f"Error: {row['track_name']} - {e}")
    return None

lyrics_list = []

for i, row in df.iterrows():

    lyrics = get_lyrics(row)
    lyrics_list.append(lyrics)
    if lyrics:
        print(f"{i + 1}/{len(df)} - {row['track_name']}")

    else:
        print(f"{i + 1}/{len(df)} - {row['track_name']} НЕ НАЙДЕН")

    time.sleep(0.1)

    # сохраняем каждые 100 строк

    if (i + 1) % 100 == 0:
        temp_df = df.iloc[:i+1].copy()
        temp_df['lyrics'] = lyrics_list

        save_path = f"saves/checkpoint_{i+1}.csv"
        temp_df.to_csv(save_path, index=False)

        print(f"Сохранено: {save_path}")

# финальный файл
df['lyrics'] = lyrics_list

df = df.dropna(subset=['lyrics'])
df = df[df['lyrics'].str.strip() != '']

df.to_csv("saves/dataset_with_lyrics.csv", index=False)