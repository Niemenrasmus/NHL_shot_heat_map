import requests
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from PIL import Image
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.colors as mcolors
c = mcolors.ColorConverter().to_rgb
pos_cm = ListedColormap([c('#BCFA9E'),c('#80B665'),c('#32780F')]) # Color mapping for positive relative freq
neg_cm = ListedColormap([c('#8ED3E9'), c('#49BCE1'),c('#4984E1')]) # Color mapping for negative relative freq


# Pull game event data (shots, hits, faceoffs etc.) from NHL API given current season, 
# max num of games and season type (pre, regular, playoffs)
def pull_nhl_api_data(year, num_of_games, season_type):
    game_data = []
    # Pull the data from NHL API in json form and format as a list 
    for i in range(0,num_of_games):
        req = requests.get(url="http://statsapi.web.nhl.com/api/v1/game/{}{}{}/feed/live"
                            .format(year, season_type, str(i).zfill(4)))
        data = req.json()
        game_data.append(data)
    with open('./{}_event_data.pkl'.format(year), 'wb') as f:
        pickle.dump(game_data, f, pickle.HIGHEST_PROTOCOL)
    return game_data


# Function for getting all shot and goal coordinates
# If wanted data from all players, player_type should be 'all'
def get_shot_and_goal_coordinates(year, player_name):

    # Read data. If data from that season has already been stored as pickle, use that file. 
    # Else query the new file 
    try:
        with open('../NHL_heat_map/{}_event_data.pkl'.format(year), 'rb') as f:
            api_data = pickle.load(f)
    except FileNotFoundError:
        api_data = pull_nhl_api_data(year, 1290, '02')
    
    # Dictionaries to store coordinates for shots and goals 
    event_data = {}
    event_data['Shot'] = {}
    event_data['Shot']['x'] = []
    event_data['Shot']['y'] = []

    event_data['Goal'] = {}
    event_data['Goal']['x'] = []
    event_data['Goal']['y'] = []

    # Only need shots and goals
    stats = ['Shot', 'Goal']

    # Loop through each game in api_data
    for game in api_data:
        # Some games in the api_data do not contain live data, so check that the game contains label "liveData".
        # If it does not, go to next value in the loop
        if 'liveData' not in game:
            continue

        # Play by play data for the game
        play_by_play_data = game['liveData']['plays']['allPlays']
        
        # For each event in play to play data
        for event in play_by_play_data:
            # Check if this event should be checked 
            check_event = False

            # If player_name is all, all events will be checked
            if player_name == 'all':
                check_event = True
            
            # Else, check if the name of the player appears in the players of the event
            else:
                if 'players' in event:
                    for player in event['players']:
                        if player['player']['fullName'] in [player_name] \
                            and player['playerType'] in ['Shooter','Scorer']:
                            check_event = True

            # For shot and goal
            if check_event:
                for stat in stats:
                    # If event is shot or goal
                    if event['result']['event'] in [stat]:
                        # If event contains coordinates
                        if 'x' in event['coordinates']:
                            # Saving coordinates for the dictionary about the coordinates
                            event_data[stat]['x'].append(event['coordinates']['x'])
                            event_data[stat]['y'].append(event['coordinates']['y'])
    
    return event_data


# Function for flipping coordinates as players cant score on their own goal
# Returns normalized values as a list for coordinates in this order:
# all_shots_x_norm, all_shots_y_norm, all_goals_x_n orm, all_goals_y_norm
def normalize_coordinates(event_data):
    # Concat arrays for shots and goals
    all_shots_x = event_data['Shot']['x'] + event_data['Goal']['x']
    all_shots_y = event_data['Shot']['y'] + event_data['Goal']['y']

    # Normalize coordinates
    all_shots_x_norm = []
    all_shots_y_norm = []
    all_goals_x_norm = []
    all_goals_y_norm = []

    # All shots
    for i, j in enumerate(all_shots_x):
        if(all_shots_x[i] < 0):
            all_shots_x_norm.append(-all_shots_x[i])
            all_shots_y_norm.append(-all_shots_y[i])
        else:
            all_shots_x_norm.append(all_shots_x[i])
            all_shots_y_norm.append(all_shots_y[i])
    
    # Goals
    for i, j in enumerate(event_data['Goal']['x']):
        if(event_data['Goal']['x'][i] < 0):
            all_goals_x_norm.append(-event_data['Goal']['x'][i])
            all_goals_y_norm.append(-event_data['Goal']['y'][i])
        else:
            all_goals_x_norm.append(event_data['Goal']['x'][i])
            all_goals_y_norm.append(event_data['Goal']['y'][i])
    
    return [all_shots_x_norm, all_shots_y_norm, all_goals_x_norm, all_goals_y_norm]


def get_shot_and_goal_freq(all_shots_x_norm, all_shots_y_norm, all_goals_x_norm, all_goals_y_norm):
    # Coordinate values vary between X: -100 to 100 and Y: -42.5, 42.5
    # To keep the aspect ration correct we use square figure size

    x_binds = np.array([-100.0, 100.0])
    y_binds = np.array([-100.0, 100.0])

    extent = [x_binds[0],x_binds[1],y_binds[0],y_binds[1]]

    # Binning with 40 unit increments 
    gridsize= 30
    mincnt=0

    # Use hexbin function to bucket shot data into a 2D-historgram

    all_data_hex = plt.hexbin(all_shots_x_norm, all_shots_y_norm, gridsize=gridsize,
    extent=extent, mincnt=mincnt, alpha=0.0)

    # Extract bin coordinates and counts for shots
    all_data_bins = all_data_hex.get_offsets()
    all_data_shot_freq = all_data_hex.get_array()

    all_data_goals_hex = plt.hexbin(all_goals_x_norm, all_goals_y_norm, gridsize=gridsize,
    extent=extent, mincnt=mincnt, alpha=0.0)

    all_data_goal_freq = all_data_goals_hex.get_array()

    return all_data_shot_freq, all_data_goal_freq, all_data_bins


def plot_shot_frequency_and_efficiency(all_shots_x_norm, all_shots_y_norm, all_goals_x_norm, all_goals_y_norm):
    
    # Setting up grid to show where shots are taken in the ice
    # Coordinate values vary between X: -100 to 100 and Y: -42.5, 42.5
    # To keep the aspect ration correct we use square figure size

    all_data_shot_freq, all_data_goal_freq, all_data_bins = \
    get_shot_and_goal_freq(all_shots_x_norm, all_shots_y_norm, all_goals_x_norm, all_goals_y_norm)

    # Downloading picture of a hockey rink, and scale our coordinate points to match the rink

    # Set up new figure
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1,1,1)

    # Clean the figure and remove labelling
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.0)

    ax.set_xticklabels(labels = [""], fontsize = 16, alpha = 0.7, minor = False)
    ax.set_yticklabels(labels = [""], fontsize = 16, alpha = 0.7, minor = False)

    image = Image.open('../NHL_heat_map/rink_image.png')
    ax.imshow(image);width, height = image.size

    # Variables for scaling (Trial and error)
    scaling_x=width/100-1.2
    scaling_y=height/100+0.5
    x_trans=33
    y_trans=height/2

    S = 3.6*scaling_x

    # Plotting loop. Plot shots and goals with different colors
    for i, j in enumerate(all_data_bins):
        # Skip empty locations
        if all_data_shot_freq[i] <  1:
            continue
        # Normalizing shot freq between [0.1]
        norm_shot_freq = all_data_shot_freq[i]/max(all_data_shot_freq)

        # Scale hex based on shot freq
        rad = S * math.sqrt(norm_shot_freq)

        # Plot hex based on scalings made previously
        shot_hex = RegularPolygon((x_trans+j[0]*scaling_x, y_trans-j[1]*scaling_y), numVertices=6,
            radius=rad, facecolor = '#08DCF7', orientation=np.radians(0), alpha=0.5, edgecolor=None)
        
        ax.add_patch(shot_hex)

        # Normalizing shot freq between [0.1]
        norm_goal_freq = all_data_goal_freq[i]/max(all_data_goal_freq)

        # Scale hex based on shot freq
        rad = S * math.sqrt(norm_goal_freq)

        # Plot hex based on scalings made previously
        goal_hex = RegularPolygon((x_trans+j[0]*scaling_x, y_trans-j[1]*scaling_y), numVertices=6,
            radius=rad, facecolor = '#89FC52', orientation=np.radians(0), alpha=0.5, edgecolor=None)
        
        ax.add_patch(goal_hex)


def plot_player_shooting_efficiency(all_league_data, player_data):
    
    # Get efficiency for both the player and the league. This to get the relative efficiency  
    league_shot_freq, league_goal_freq, league_data_bins = get_shot_and_goal_freq(*all_league_data)
    player_shot_freq, player_goal_freq, player_data_bins = get_shot_and_goal_freq(*player_data)
    
    league_efficiency = []
    player_efficiency = []
    relative_efficiency = []

    for i in range(0,len(league_shot_freq)):
        if league_shot_freq[i]<2 or player_shot_freq[i]<2:
            continue
        league_efficiency.append(league_goal_freq[i]/league_shot_freq[i])
        player_efficiency.append(player_goal_freq[i]/player_shot_freq[i])
        relative_efficiency.append((player_goal_freq[i]/player_shot_freq[i])-(league_goal_freq[i]/league_shot_freq[i]))

    # Downloading picture of a hockey rink, and scale our coordinate points to match the rink

    # Set up new figure
    fig = plt.figure(figsize = (50, 50))
    ax = fig.add_subplot(1,1,1)

    # Clean the figure and remove labelling
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.0)

    ax.set_xticklabels(labels = [""], fontsize = 16, alpha = 0.7, minor = False)
    ax.set_yticklabels(labels = [""], fontsize = 16, alpha = 0.7, minor = False)

    image = Image.open('../NHL_heat_map/rink_image.png')
    ax.imshow(image);width, height = image.size

    # Variables for scaling (Trial and error)
    scaling_x=width/100-1.2
    scaling_y=height/100+0.5
    x_trans=33
    y_trans=height/2

    S = 3.6*scaling_x

    # Plotting loop
    for i, j in enumerate(player_data_bins):
        # Skip empty locations
        if player_shot_freq[i] <  1:
            continue

        # Normalizing shot freq between [0.1]
        norm_shot_freq = player_shot_freq[i]/max(player_shot_freq)

        # Scale hex based on shot freq
        rad = S * math.sqrt(norm_shot_freq)

        player_efficiency = player_goal_freq[i]/player_shot_freq[i]
        league_efficiency = league_goal_freq[i]/league_shot_freq[i]
        relative_efficiency = player_efficiency - league_efficiency

        if relative_efficiency<0:
            col = neg_cm(math.pow(-1*relative_efficiency, 0.1))
        else:
            col = pos_cm(math.pow(relative_efficiency, 0.1))

        # Plot hex based on scalings made previously
        shot_hex = RegularPolygon((x_trans+j[0]*scaling_x, y_trans-j[1]*scaling_y), numVertices=6,
            radius=rad, facecolor = col, orientation=np.radians(0), alpha=1, edgecolor=None)
        
        ax.add_patch(shot_hex)
    
    ax.set_xlim([0,width])
    ax.set_ylim([height, 0])
    for sp in ax.spines.values():
        sp.set_edgecolor('white')
    plt.grid(False)

