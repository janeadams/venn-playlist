"""
    Venn Playlist: A visualization tool for finding hypergraphs in your Spotify playlists.
    Copyright (C) 2023 Jane L. Adams

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import time
import csv
import json
import requests
from tqdm import tqdm
from functools import wraps
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException  # Import the SpotifyException class
from dotenv import load_dotenv
import pandas as pd
import scipy.cluster.hierarchy as sch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
pio.templates.default = "plotly_dark"

import logging
#logging.basicConfig(level=logging.DEBUG)

order = ["red","orange","yellow","green","blue","purple", "pink", "brown", "black", "white"]
try:
    with open('color_mapping.json', 'r') as f:
        color_mapping = json.load(f)
except Exception as e:
    tqdm.write(f"Error loading color mapping: {e}")
try:
    with open('folder_mapping.json', 'r') as f:
        folder_mapping = json.load(f)
except Exception as e:
    tqdm.write(f"Error loading folder mapping: {e}")


def auth_call():
    # Load environment variables from .env file
    load_dotenv()

    # Get credentials from environment variables
    SPOTIPY_CLIENT_ID = os.getenv('CLIENT_ID')
    SPOTIPY_CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    SPOTIPY_REDIRECT_URI = os.getenv('REDIRECT_URI')
    tqdm.write(SPOTIPY_REDIRECT_URI)

    # Get the authorization URL
    auth_manager = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                client_secret=SPOTIPY_CLIENT_SECRET,
                                redirect_uri=SPOTIPY_REDIRECT_URI,
                                scope='playlist-read-private',
                                open_browser=False)
    url = auth_manager.get_authorize_url()
    tqdm.write("Open the following URL in your browser. It should redirect to a localhost address which will appear broken. Copy that link and enter it:")
    tqdm.write(url)
    return auth_manager

def auth_response(auth_manager):
    # Manually set the URL
    url = input("Paste the URL you were redirected to: ")
    sp = spotipy.Spotify(auth_manager=auth_manager)
    sp.auth_manager.get_access_token(code=sp.auth_manager.parse_response_code(url), check_cache=False)
    return sp

def rate_limit_wrapper(func, *args, **kwargs):
    """Wrap a function to handle rate limiting."""
    retries = 3  # Number of retries
    wait_time = 30  # Initial wait time, can be increased after each retry if needed

    for _ in range(retries):
        try:
            response = func(*args, **kwargs)
            if isinstance(response, dict) and 'error' in response:
                error_status = response['error'].get('status')
                if error_status > 400:
                    retry_after = response.headers.get('Retry-After')
                    sleep_time = int(retry_after) if retry_after is not None else 30
                    tqdm.write(f"Rate limit reached. Waiting for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                else:
                    tqdm.write(f"Error: {response['error'].get('message', 'Unknown error')}")
                    return response
            else:
                return response
        except Exception as e:
            if "Too Many Requests" in str(e):
                tqdm.write("Rate limit reached. Waiting...")
                time.sleep(wait_time)
                wait_time *= 2  # Double the wait time for each retry
            else:
                tqdm.write(f"Error: {e}")
                return None
    print("Max retries reached. Exiting.")
    return None

def get_folder_by_playlist(playlist_name, df=None):
    """
    Function to get the associated folder name based on a playlist's name.
    Performs a fuzzy matching based on the substrings in the provided folder_mapping dictionary.
    """
    # Find folder for each playlist
    try:
        with open('folder_mapping.json', 'r') as f:
            folder_mapping = json.load(f)
    except Exception as e:
        tqdm.write(f"Error loading folder mapping: {e}")
        return
    playlist_name = playlist_name.lower()
    for folder, playlists in folder_mapping.items():
        for key_substring in playlists:
            if key_substring in playlist_name:
                return folder
    print(f"Could not find folder for playlist {playlist_name}.")
    return None

def get_genres_by_artists(sp, artist_ids, cache):
    """
    Fetch genres for a list of artist IDs.
    """
    # Filter out artist IDs already present in the cache
    artist_ids_to_fetch = [artist_id for artist_id in artist_ids if artist_id not in cache]

    if not artist_ids_to_fetch:
        return {artist_id: cache[artist_id] for artist_id in artist_ids}
    
    genres_by_artist = {}

    # Fetch genres in batches
    for i in range(0, len(artist_ids_to_fetch), 50):  # Assuming 50 is the batch size limit
        batch = artist_ids_to_fetch[i:i+50]
        
        try:
            response = rate_limit_wrapper(sp.artists, batch)
            for artist_data in response.get('artists', []):
                artist_id = artist_data['id']
                genres = artist_data.get('genres', [])
                genres_by_artist[artist_id] = genres
                cache[artist_id] = genres
        except Exception as e:
            tqdm.write(f"Error fetching genres for batch starting with artist ID {batch[0]}. Error: {e}")
            # You could choose to exit or continue here, depending on your requirements.
            continue
    
    # Filling in genres for artists present in the cache
    for artist_id in artist_ids:
        if artist_id not in genres_by_artist:
            genres_by_artist[artist_id] = cache.get(artist_id, [])

    return genres_by_artist


def download_playlist_artwork(sp, playlist_id, save_directory='data/artworks'):
    """
    Download and save the artwork for a specified playlist.
    """
    # Fetch playlist details using the rate limit wrapper
    playlist_details = rate_limit_wrapper(sp.playlist, playlist_id)
    
    # Extract the highest resolution image (assuming images are sorted in descending order of resolution)
    image_url = playlist_details['images'][0]['url'] if playlist_details['images'] else None
    
    if not image_url:
        tqdm.write(f"No artwork found for playlist {playlist_id}")
        return

    try:
        # Download the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Ensure the directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the image
        with open(f"{save_directory}/{playlist_id}.jpg", 'wb') as file:
            for chunk in response.iter_content(8192):
                file.write(chunk)
    except:
        tqdm.write(f"Error downloading artwork for playlist {playlist_id}")

def save_to_playlists_csv(data):
    """Save playlist data to the main playlists.csv using pandas."""
    df = pd.DataFrame([data])
    filepath = 'data/playlists.csv'

    # If the file exists, append without headers. Else, create a new one with headers.
    if pd.io.common.file_exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

def save_to_individual_playlist_csv(playlist_id, track_data):
    """Save track data to the respective playlist's CSV using pandas."""
    df = pd.DataFrame([track_data])
    filepath = f'data/playlists/{playlist_id}.csv'

    # If the file exists, append without headers. Else, create a new one with headers.
    if pd.io.common.file_exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


def save_to_summary_csv():
    """Aggregate all playlist CSVs and save to the summary.csv."""
    all_data = []
    
    # Loop through individual playlist CSVs
    for csv_file in os.listdir('data/playlists'):
        csv_path = os.path.join('data/playlists', csv_file)
        data_df = pd.read_csv(csv_path)
        data_df['playlist_id'] = csv_file[:-4]  # Remove the .csv extension
        all_data.append(data_df)

    # Concatenate all dataframes
    summary_df = pd.concat(all_data)
    
    # Add a folder column
    summary_df['folder'] = [get_folder_by_playlist(playlist_name) for playlist_name in summary_df['playlist']]
    summary_df['color'] = [color_mapping[folder] for folder in summary_df['folder']]
    summary_df.to_csv('data/summary.csv', index=False)
    return summary_df

def process_playlists(sp):
    """Fetch and process playlists from Spotify."""
    # Check if cached CSV file is available
    if os.path.exists("data/playlists.csv"):
        tqdm.write("Cached CSV file found. Loading data...")
        cached_data = pd.read_csv("data/playlists.csv")
        playlists_to_process = cached_data.to_dict('records')
        return playlists_to_process

    tqdm.write("Fetching user playlists from Spotify API...")

    playlists_to_process = []

    try:
        results = rate_limit_wrapper(sp.current_user_playlists)
        playlists = results['items']

        # Paginate through results if there are more playlists
        while results.get('next'):
            results = rate_limit_wrapper(sp.next, results)
            playlists.extend(results['items'])

    except Exception as e:
        tqdm.write(f"Error fetching playlists: {e}")
        return playlists_to_process

    # Process each playlist and save relevant details with a progress bar
    for playlist in tqdm(playlists, desc="Processing Playlists", ncols=100):  # ncols=100 just to make the bar span the console more widely
        try:
            playlist_data = {
                'name': playlist['name'],
                'description': playlist.get('description', ''),  # Gets the description or defaults to an empty string if not available
                'id': playlist['id'],
                'number_of_tracks': playlist['tracks']['total'],  # Assuming 'tracks' field contains 'total'
            }
            save_to_playlists_csv(playlist_data)
            playlists_to_process.append(playlist_data)
            download_playlist_artwork(sp, playlist_data['id'])
        except Exception as e:
            tqdm.write(f"Error processing playlist {playlist['name']}: {e}")
            continue

    return playlists_to_process

def get_audio_features(sp, df):
    df = df.copy()
    features_list = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    track_ids = df['track_id'].tolist()

    # Paginate through track IDs if there are more than 100:
    if len(track_ids) > 99:
        tqdm.write("More than 99 tracks found. Fetching audio features in batches...")
        found = []

        for i in tqdm(range(0, len(track_ids), 99)):
            # Using wrapper to get the audio features, no need to handle rate limiting inside this loop
            current_track_ids = track_ids[i:i+99]
            batch_features = rate_limit_wrapper(lambda ids: sp.audio_features(ids), current_track_ids)
            if batch_features is not None:
                found.extend(batch_features)
            else:
                # Handle the error, perhaps by logging or raising an exception.
                tqdm.write(f"Error fetching audio features for track_ids: {current_track_ids}")

    else:
        tqdm.write("Fetching audio features...")
        found = rate_limit_wrapper(lambda ids: sp.audio_features(ids), track_ids)

    for feature in features_list:
        if found:
            # Ensure the feature exists for each track, else set it as NaN
            df[feature] = [f.get(feature, np.nan) for f in found]
        else:
            df[feature] = np.nan

    return df


def process_tracks_for_playlist(sp, playlist_id, playlist_name, genre_cache, fetch_genres=True, fetch_audio_features=True):
    """
    Process tracks for a given playlist. This includes fetching track details and the track's features.
    """
    tqdm.write(f"Fetching tracks for playlist {playlist_name}...")
    all_tracks = []

    results = rate_limit_wrapper(lambda pid: sp.playlist_tracks(pid), playlist_id)
    all_tracks.extend(results.get('items', []))
    
    tqdm.write(f"Total initial tracks: {len(all_tracks)}")

    # Paginate through results if there are more tracks
    while results.get('next'):
        results = rate_limit_wrapper(lambda pid: sp.next(pid), results)
        all_tracks.extend(results.get('items', []))
        tqdm.write(f"Total tracks after pagination: {len(all_tracks)}")

    track_data_for_csv = []

    for track in tqdm(all_tracks, desc=f"Processing tracks for {playlist_name}"):
        try:
            track_info = track.get('track')
            if not track_info:  # Check if track_info is None or empty
                continue  # Skip this loop iteration and move to the next track
            artist_info_list = track_info.get('artists', [{}])
            artist_info = artist_info_list[0] # Safely get the first item
            artist_id = artist_info.get('id')

            to_add = {
                'track_name': track_info.get('name'),
                'track_id': track_info.get('id'),
                'artist_name': artist_info.get('name'),
                'artist_id': artist_id,
                'date_added': track.get('added_at'),
                'playlist': playlist_name,
                'playlist_id': playlist_id
            }

            track_data_for_csv.append(to_add)
            
        except spotipy.SpotifyException as e:
            if e.http_status == 401:
                tqdm.write("Token has expired or is invalid. Refreshing token...")
                sp.refresh_access_token()  # Refresh the access token
                continue  # Move on to the next track

            tqdm.write(f"Error processing track {track_info.get('name')}: {e}")
            continue

    df = pd.DataFrame(track_data_for_csv)

    if fetch_audio_features:
        # Batch request audio features for all tracks:
        df = get_audio_features(sp, df)
    
    if fetch_genres:
        # Get genres for all artists:
        genre_cache = {}
        tqdm.write("Fetching genres for artists...")
        unique_artist_ids = df['artist_id'].unique().tolist()
        genres_by_artist = get_genres_by_artists(sp, unique_artist_ids, genre_cache)

        # Update the genre cache
        genre_cache.update(genres_by_artist)

        # Map genres to the DataFrame
        df['genre'] = df['artist_id'].map(genres_by_artist)

    df.to_csv(f'data/playlists/{playlist_id}.csv', index=False)
    return genre_cache


def download_data(sp, fetch_genres=True, fetch_audio_features=True, exclude_playlists=['Discover Weekly', 'Release Radar', 'Liked from Radio', 'GENU WIN', 'On Repeat']):
    """Downloads and processes the Spotify data."""
    # Directory setup logic
    try:
        os.makedirs('data/playlists', exist_ok=True)
    except Exception as e:
        tqdm.write(f"Error creating directories: {e}")
        return

    if fetch_genres:
        # Load genre cache
        try:
            genre_cache_df = pd.read_csv('data/genre_cache.csv')
            genre_cache = dict(zip(genre_cache_df['artist_id'], genre_cache_df['genres']))
        except FileNotFoundError:
            tqdm.write("No existing genre cache found. Initializing empty cache.")
            genre_cache = {}
    else:
        genre_cache = {}

    # Process playlists
    playlists_to_process = process_playlists(sp)
    
    # Process tracks for each playlist
    for playlist in playlists_to_process:
        playlist_id = playlist['id']
        playlist_name = playlist['name']
        if playlist_name not in exclude_playlists:
            # Check that we haven't already processed this playlist
            if os.path.exists(f"data/playlists/{playlist_id}.csv"):
                playlist_df = pd.read_csv(f"data/playlists/{playlist_id}.csv")
                cols_to_check = playlist_df.columns.tolist()
                if ((fetch_audio_features) and ('valence' not in cols_to_check) | (fetch_genres) and ('genre' not in cols_to_check)):
                    print(f"Reprocessing playlist {playlist_name}...")
                    process_tracks_for_playlist(sp, playlist_id, playlist_name, genre_cache, fetch_genres=fetch_genres, fetch_audio_features=fetch_audio_features)
                else:
                    tqdm.write(f"Playlist {playlist_name} already processed. Skipping...")
            else:
                genre_cache = process_tracks_for_playlist(sp, playlist_id, playlist_name, genre_cache, fetch_genres=fetch_genres, fetch_audio_features=fetch_audio_features)

    if fetch_genres:
        # Save genre cache to disk
        genre_cache_df = pd.DataFrame(list(genre_cache.items()), columns=['artist_id', 'genres'])
        genre_cache_df.to_csv('data/genre_cache.csv', index=False)

    # Save aggregated data to summary.csv
    df = save_to_summary_csv()

    tqdm.write("Data processing completed!")
    return df

def get_playlist_groups(df):
    playlist_groups = pd.DataFrame(df.groupby('playlist')['track'].count().reset_index())
    playlist_groups = playlist_groups.sort_values('folder')
    return playlist_groups

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},1)'

def color_to_hex(color_name):
    return color_mapping[color_name]

# Create a function that blends two colors together
def blend_colors(color_name1, color_name2):
    # Check if its already hex:
    if color_name1[0] == '#':
        color1 = color_name1
    else:
        color1 = color_to_hex(color_name1)
    if color_name2[0] == '#':
        color2 = color_name2
    else:
        color2 = color_to_hex(color_name2)
    r1, g1, b1 = [int(color1[i:i+2], 16) for i in (1, 3, 5)]
    r2, g2, b2 = [int(color2[i:i+2], 16) for i in (1, 3, 5)]
    r = int((r1 + r2) / 2)
    g = int((g1 + g2) / 2)
    b = int((b1 + b2) / 2)
    return f'#{r:02x}{g:02x}{b:02x}'

def get_timespan(added, span):
    if span == 'year':
        return pd.to_datetime(added.dt.year.astype(str) + '-07-01')
    elif span == 'month':
        return pd.to_datetime(added.dt.year.astype(str) + '-' + added.dt.month.astype(str) + '-15')
    elif span == 'week':
        # Strip time component and calculate the middle of the week
        mid_week = (added.dt.date - pd.to_timedelta(added.dt.dayofweek - 2, unit='d')).astype('datetime64[ns]')
        return mid_week
    elif span == 'day':
        return pd.to_datetime(added.dt.date) + pd.Timedelta(hours=12)



def make_timeline(df):
    data = []
    df['date_added'] = pd.to_datetime(df['date_added'])
    df.sort_values(by='date_added', inplace=True)
    
    for i, span in reversed(list(enumerate(['year','month','week','day']))):
        df[span] = get_timespan(df['date_added'], span)
        grouped = df.groupby(span).count().reset_index().sort_values(by=span).rename(columns={df.columns[0]: 'tracks added'})
        if span == 'day':
            grouped = grouped[grouped['tracks added'] > 0]
            trace = px.scatter(grouped, x=span, y='tracks added')
            trace.update_traces(marker=dict(color=f'rgba(255,255,255,{(1/(len(span)))*(i)})'))
        else:
            trace = px.line(grouped, x=span, y='tracks added')
            trace.update_traces(line=dict(color=f'rgba(255,255,255,{(1/(len(span)))*(i)})', width=i/2))
        data.append(trace.data[0])
        
    fig = go.Figure(data=data, layout={'width': 1200, 'height': 600, 'title': 'Tracks added over time'})
    fig.update_layout(yaxis_type="log")  # Adjust Y-axis to log scale
    fig.update_layout(autosize=False)
    fig.write_html('figs/timeline.html')
    fig.write_image('figs/timeline.png', scale=3)
    return fig


def make_folder_stacked_timeline(df, scale='month'):
    grouped = df.groupby([scale, 'folder']).agg({
            'date_added': 'size',  # Assuming 'date_added' is a column you can count.
            'color': 'first'
        }).reset_index()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing = 0.05)
    for folder in order:
        subset = grouped[grouped['folder'] == folder]

        if 'color' not in subset.columns:
            raise ValueError("The 'color' column is not in the subset DataFrame.")
        non_string_colors = subset[~subset['color'].apply(lambda x: isinstance(x, str))]['color']
        if not non_string_colors.empty:
            raise ValueError(f"Non-string values found in the 'color' column: {non_string_colors.unique()}")

        fig.add_trace(go.Scatter(
            x=subset[scale],
            y=subset['date_added'],
            name=folder,
            fill='tonexty',
            stackgroup='one',
            groupnorm='percent',
            fillcolor=hex_to_rgba(list(subset['color'])[0]),  # Setting fillcolor here
            line=dict(width=0, color='rgba(0,0,0,0)')
            ), row=2, col=1)
    totals = df.groupby([scale]).count().filter(['date_added']).reset_index()
    fig.add_trace(go.Bar(
            x=totals[scale],
            y=[t for t in totals['date_added']],
            name='total',
            marker=dict(color='white', opacity=1, line={'width':0}),
            ), row=1, col=1)
    fig.update_yaxes(title='Total Songs<br>Added', row=1)
    fig.update_yaxes(title='% Added to<br>Each Folder', row=2)
    fig.update_layout(width=800, height=600, title='Songs Added Over Time', showlegend=False, margin={'l':100, 'r': 40, 't':50, 'b': 30})
    fig.update_layout(autosize=False)
    fig.write_html('figs/stacked_timeline.html')
    fig.write_image('figs/stacked_timeline.png', scale=3)
    return fig

def make_folder_bars(df):
    df['folder'] = df['folder'].astype('category').cat.reorder_categories(order)
    
    # Group and count
    grouped = df.groupby(['folder','playlist_id']).agg({
        'date_added': 'count',
        'playlist': 'first'
    }).reset_index()
    
    # Assuming 'date_added' was the column being counted and is now named 'date_added_count'
    fig = px.bar(grouped, x='folder', y='date_added', color='folder', 
                 color_discrete_map=color_mapping, hover_name='playlist',
                 title='Number of tracks per playlist', 
                 labels={'folder': 'Folder', 'date_added': 'Number of tracks', 'playlist': 'Playlist'})
    
    fig.update_layout(showlegend=False, width=1000, height=400)
    fig.update_layout(autosize=False)
    fig.write_html('figs/folder_bars.html')
    fig.write_image('figs/folder_bars.png', scale=3)
    return fig

def compute_playlist_tracks(df):
    """
    Computes the set of tracks for each playlist in the dataframe.

    Parameters:
    - df: DataFrame with the tracks and playlists

    Returns:
    - A dictionary mapping playlist_id to its set of tracks
    """
    return {pid: set(df[df['playlist_id'] == pid]['track_id']) for pid in df['playlist_id'].unique()}

def create_shared_df(df=None, shared_data=None):
    """
    Computes shared tracks data for each unique pair of playlists and 
    creates a DataFrame from the shared tracks data, then sorts it.

    Parameters:
    - df: DataFrame with the tracks and playlists
    - shared_data: List of dictionaries with shared tracks data

    Returns:
    - A sorted DataFrame with shared tracks data
    """
    if shared_data is None:
        if df is None:
            raise ValueError("Either shared_data or df must be provided.")
        
        playlist_tracks = compute_playlist_tracks(df)
        shared_data = []
        unique_playlists = df['playlist_id'].unique()

        # Create mappings for playlist_id to color and name
        playlist_color_mapping = dict(df.drop_duplicates('playlist_id').set_index('playlist_id')['color'])
        playlist_name_mapping = dict(df.drop_duplicates('playlist').set_index('playlist_id')['playlist'])

        for i, pid_a in enumerate(unique_playlists):
            for j, pid_b in enumerate(unique_playlists):
                if j > i:
                    shared_tracks = list(playlist_tracks[pid_a] & playlist_tracks[pid_b])
                    shared_count = len(shared_tracks)

                    if shared_count > 0:
                        shared_data.append({
                            'playlist_id_a': pid_a,
                            'playlist_id_b': pid_b,
                            'color_a': playlist_color_mapping[pid_a],
                            'color_b': playlist_color_mapping[pid_b],
                            'playlist_name_a': playlist_name_mapping[pid_a],
                            'playlist_name_b': playlist_name_mapping[pid_b],
                            'shared_count': shared_count,
                            'shared_tracks': df[df['track_id'].isin(shared_tracks)]['track_name'].tolist()
                        })

    return pd.DataFrame(shared_data).sort_values(by='shared_count', ascending=False)


def prepare_matrix_df(shared_df=None, df=None, use='name'):
    """
    Prepare a matrix dataframe based on shared_df.

    Parameters:
    - shared_df: Dataframe containing shared tracks information
    - df: Original dataframe containing track and playlist information

    Returns:
    - matrix_df: Dataframe that represents a matrix of shared track counts between playlists
    """
    if shared_df is None:
        if df is None:
            raise ValueError("Either shared_df or df must be provided.")
        else:
            shared_df = create_shared_df(df)

    mirror_df = shared_df.copy()
    if use == 'name':
        i = 'playlist_name_a'
        j = 'playlist_name_b'
    elif use == 'id':
        i = 'playlist_id_a'
        j = 'playlist_id_b'
    mirror_df[i], mirror_df[j] = shared_df[j], shared_df[i]
    mirror_df = pd.concat([shared_df, mirror_df])
    matrix_df = mirror_df.pivot(index=j, columns=i, values='shared_count').fillna(0)
    return matrix_df

def plot_shared_count_heatmap(matrix_df=None, shared_df=None, df=None):
    """
    Plots a heatmap showing the shared count between playlists.

    Parameters:
    - matrix_df: Dataframe that represents a matrix of shared track counts between playlists
    - shared_df: Dataframe with shared track data between playlists
    - df: Original dataframe containing track and playlist information

    Returns:
    - fig: Figure object representing the heatmap
    """
    if matrix_df is None:
        if shared_df is None:
            if df is None:
                raise ValueError("Either matrix_df, shared_df or df must be provided.")
            else:
                shared_df = create_shared_df(df)
                matrix_df = prepare_matrix_df(shared_df)
        else:
            matrix_df = prepare_matrix_df(shared_df)

    matrix_df = matrix_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    distance_matrix = 1 / (matrix_df + 1)  # +1 to avoid division by zero
    linkage = sch.linkage(sch.distance.pdist(distance_matrix), method='average')
    dendro_order = list(sch.dendrogram(linkage, no_plot=True)['leaves'])

    sorted_matrix = matrix_df.iloc[dendro_order, dendro_order]
    y_labels = matrix_df.index[dendro_order]
    x_labels = matrix_df.columns[dendro_order]

    fig = px.imshow(sorted_matrix,
                    x=x_labels,
                    y=y_labels,
                    labels=dict(color="Shared Count", x="Playlists", y="Playlists"),
                    title="Shared Count between Playlists",
                    color_continuous_scale='Magma'
                    )
    fig.update_layout(autosize=False)
    fig.update_layout(width=1000, height=1000, showlegend=False)
    fig.write_html('figs/heatmap.html')
    fig.write_image('figs/heatmap.png', scale=3)
    return fig



def plot_thresholded_parcats(shared_df=None, df=None, threshold=0, show_folders=False, show_playlists=True):
    """
    Plots a Parcats (Parallel Categories) diagram based on a threshold.

    Parameters:
    - shared_df: Dataframe containing shared tracks information

    Returns:
    - fig: Figure object representing the Parcats diagram
    """
    if shared_df is None:
        if df is None:
            raise ValueError("Either shared_df or df must be provided.")
        else:
            shared_df = create_shared_df(df=df)
    thresheld = shared_df[shared_df['shared_count'] > threshold]
    # Blend colors based on the playlist colors in the color_lookup
    thresheld['gradient_color'] = [blend_colors(a_color, b_color) for a_color, b_color in zip(thresheld['color_a'], thresheld['color_b'])]
    thresheld['folder_a'] = [get_folder_by_playlist(a, df=df) for a in thresheld['playlist_name_a']]
    thresheld['folder_b'] = [get_folder_by_playlist(b, df=df) for b in thresheld['playlist_name_b']]

    dimensions = []
    if show_folders and show_playlists:
        dimensions.extend(['folder_a','playlist_name_a','playlist_name_b','folder_b'])
    elif show_folders:
        dimensions.extend(['folder_a','folder_b'])
    elif show_playlists:
        dimensions.extend(['playlist_name_a','playlist_name_b'])
    else:
        raise ValueError("Either show_folders or show_playlists must be True.")

    fig = go.Figure(go.Parcats(
        dimensions=[{'label': p, 'values': thresheld[p]} for p in dimensions],
        counts=thresheld['shared_count'],
        line={'color': thresheld['gradient_color'], 'shape': 'hspline'}
    ))
    fig.update_layout(width=1200, height=800)
    fig.update_layout(autosize=False)
    fig.write_html('figs/parcats.html')
    fig.write_image('figs/parcats.png', scale=3, width=1200, height=800)
    return fig

def plot_parallel_categories(shared_df=None, df=None, threshold=20):
    """
    Plots a Parallel Categories diagram for playlists.

    Parameters:
    - shared_df: Dataframe containing shared tracks information

    Returns:
    - fig: Figure object representing the Parallel Categories diagram
    """
    if shared_df is None:
        if df is None:
            raise ValueError("Either shared_df or df must be provided.")
        else:
            shared_df = create_shared_df(df=df)
    thresheld = shared_df[shared_df['shared_count'] > threshold]
    exploded_df = thresheld.explode('shared_tracks')
    fig = px.parallel_categories(exploded_df, dimensions=['playlist_name_a', 'playlist_name_b'], color='shared_count', color_continuous_scale='Viridis')
    fig.update_layout(height=600, width=700, margin={'l': 50, 'r': 100, 't': 50, 'b': 50}, showlegend=False, coloraxis_showscale=False)
    fig.update_layout(autosize=False)
    fig.write_html('figs/sankey.html')
    fig.write_image('figs/sankey.png', scale=3)
    return fig

def track_repeat_analysis(df):
    """
    Analyzes track repeats in the dataframe.

    Parameters:
    - df: Dataframe containing tracks, track names, and playlists information

    Returns:
    - fig: Figure object representing the track repeats analysis
    """
    track_repeats = df.groupby('track_id').count()
    track_repeats = track_repeats.filter(['playlist']).rename(columns={'playlist': 'count'}).sort_values('count', ascending=False)
    
    # Using merge/join to get track names for the repeated tracks
    track_names = df[['track_id', 'track_name']].drop_duplicates()
    track_repeats = track_repeats.merge(track_names, on='track_id', how='left')
    
    count_summary = track_repeats.groupby('count').count().reset_index().sort_values(by='count', ascending=False)
    fig = px.bar(count_summary, x='count', y='track_name', log_y=True)
    fig.update_layout(autosize=False, width=1000, height=500, title='Number of tracks repeated across playlists')
    fig.write_html('figs/repeat_bar.html')
    fig.write_image('figs/repeat_bar.png', scale=3)
    return fig
