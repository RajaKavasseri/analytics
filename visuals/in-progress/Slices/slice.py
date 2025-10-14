import numpy as np
import pandas as pd

def slice_place(player, path, fh_bh, exclude_times=None):
    # Load the data
    events = pd.read_csv(path)
    events['pointWonBy'] = events.groupby('pointNumber')['pointWonBy'].bfill()
    events['isError'] = (events['isErrorWideR'] == 1) | (events['isErrorWideL'] == 1) | (events['isErrorNet'] == 1) | (events['isErrorLong'] == 1)
    
    # Filter for the player's slices and shots in rally. Note: it is counting some serves as slices, I could filter this out...?
    slice_place = events[(events['shotHitBy'] == player) & (events['isSlice'] == 1) & (events['shotInRally'] != 1) ][
        ['pointStartTime', 'shotHitBy', 'shotContactX', 'shotContactY', 'shotLocationX', 'shotLocationY',
         'pointWonBy', 'isWinner', 'shotFhBh', 'isError', 'isErrorNet', 'side']
    ].dropna(subset=['pointWonBy']).copy()
    
    # If exclude_times is provided, filter out those rows
    if exclude_times:
        slice_place = slice_place[~slice_place['pointStartTime'].isin(exclude_times)]

    # Flip shotContactX and shotContactY where necessary
    mask_bottom_half = (slice_place['shotLocationY'] < 0) & (slice_place['shotContactY'] > 0)
    mask_near_net = (slice_place['shotLocationY'] <= 50) & (slice_place['shotContactY'] > 0) & (slice_place['isErrorNet'] == 1)

    slice_place.loc[mask_bottom_half, 'shotContactX'] *= -1
    slice_place.loc[mask_bottom_half, 'shotLocationX'] *= -1
    slice_place.loc[mask_bottom_half & (slice_place['shotContactY'] > 0), 'shotContactY'] *= -1
    slice_place.loc[mask_bottom_half, 'shotLocationY'] = slice_place.loc[mask_bottom_half, 'shotLocationY'].abs()

    slice_place.loc[mask_near_net & ~mask_bottom_half, 'shotContactX'] *= -1
    slice_place.loc[mask_near_net & ~mask_bottom_half, 'shotLocationX'] *= -1
    slice_place.loc[mask_near_net & ~mask_bottom_half, 'shotContactY'] *= -1

    # Accounting for net error tagging discrepencies
    mask = (slice_place['shotLocationY'] != 0) & (slice_place['isErrorNet'] == 1)
    adjust_up = mask & (slice_place['shotLocationX'] <= slice_place['shotContactX'])
    adjust_down = mask & (slice_place['shotLocationX'] > slice_place['shotContactX'])

    slice_place.loc[adjust_up, 'shotLocationX'] += slice_place.loc[adjust_up, 'shotLocationY']
    slice_place.loc[adjust_down, 'shotLocationX'] -= slice_place.loc[adjust_down, 'shotLocationY']

    slice_place.loc[adjust_up, 'shotContactX'] += slice_place.loc[adjust_up, 'shotLocationY']
    slice_place.loc[adjust_down, 'shotContactX'] -= slice_place.loc[adjust_down, 'shotLocationY']
    slice_place.loc[adjust_up, 'shotLocationY'] = 0
    slice_place.loc[adjust_down, 'shotLocationY'] = 0

   # filter for bad x values (perhaps a shot played after the point - in some data I looked at, it counted a random
   # shot as x = -340 and had the trajectory going the wrong direction)
    slice_place = slice_place[(slice_place['shotLocationX'] >= -300) & (slice_place['shotLocationX'] <= 300)]

   # Additional filtering for fh_bh 

    if fh_bh != 'All':
        slice_place = slice_place[slice_place['shotFhBh'] == fh_bh]

    slice_place['fhBhFiltered'] = [fh_bh != 'All'] * len(slice_place)

    # Categorize into 'left', 'mid', 'right' based on shotLocationX
    slice_place['width'] = slice_place['shotLocationX'].apply(
        lambda x: 'left' if x <= -52.5 else 'mid' if -52.5 < x < 52.5 else 'right'
    )

    # Calculate count + win pct.
    distribution = slice_place.groupby('width').apply(
        lambda df: pd.Series({
            'freq': len(df),
            'win_percentage': int((df['pointWonBy'] == df['shotHitBy']).mean() * 100)
        })
    ).reset_index()

    max_win_percentage = distribution['win_percentage'].max()
    min_win_percentage = distribution['win_percentage'].min()

    # Assign 'max', 'min', or 'no' to the distribution based on win_percentage
    distribution['maxMin'] = distribution['win_percentage'].apply(
        lambda x: 'max' if x == max_win_percentage else 'min' if x == min_win_percentage else 'no'
    )

    # Convert win_percentage to string for display
    distribution['win_percentage'] = distribution['win_percentage'].astype(str) + '%'

    # Adjust x_mapping to match the width values
    x_mapping = {
        'left': {'x': -100},
        'mid': {'x': 0},
        'right': {'x': 100}
    }

    
    # Export the data as JSON
    return_place_json = slice_place.to_json(orient='records')
    return_place_dist_json = distribution.to_json(orient='records')

    with open(os.path.join(output_dir, 'slice_place.json'), 'w') as f:
        f.write(return_place_json)

    with open(os.path.join(output_dir, 'slice_place_dist.json'), 'w') as f:
        f.write(return_place_dist_json)
