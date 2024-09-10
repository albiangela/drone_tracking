import numpy as np
import pandas as pd
import os
import re 

def get_trex_settings_value(settings_file_path, variable):
    """
    Retrieves the value of a specified variable from a settings file.
    
    Args:
    settings_file_path (str): The path to the settings file.
    variable (str): The variable name to look for in the file.
    
    Returns:
    float: The value of the specified variable as a float, or None if not found.
    """
    var = None  # Initialize variable to store the value
    
    # Open the settings file for reading
    with open(settings_file_path, 'r') as file:
        # Read through each line in the file
        for line in file:
            # Check if the variable is in the current line
            if variable in line:
                # Assuming the line format is "variable=value"
                key, value = line.strip().split('=')
                # Check if the key matches the variable we are looking for
                if key.strip() == variable:
                    # Convert the value to float and assign it to var
                    var = float(value.strip())
                    break  # Exit the loop once the variable is found

    return var  # Return the found value or None if not found


# Function to parse individual SRT file
def parse_srt_file(srt_folder,filename):
    """
    Parses an SRT file to extract frame count, latitude, longitude, relative altitude, and absolute altitude.
    
    Args:
    filename (str): The name of the SRT file to parse (without the .SRT extension).

    Returns:
    DataFrame: A pandas DataFrame containing the extracted data.
    """
    # Construct the file path for the SRT file
    srt_file_path = f'{srt_folder}{filename}.SRT'
    
    # Check if the file exists
    if not os.path.exists(srt_file_path):
        print(f"File not found: {srt_file_path}")
        return []

    try:
        # Read the content of the SRT file
        with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
            srt_content = srt_file.read()
    except Exception as e:
        # Handle any errors that occur during file reading
        print(f"Error reading {srt_file_path}: {e}")
        return []

    # Regex pattern to extract frame count, latitude, longitude, relative altitude, and absolute altitude
    pattern = r'FrameCnt: (\d+).*?latitude: ([\d.-]+).*?longitude: ([\d.-]+).*?rel_alt: ([\d.-]+) abs_alt: ([\d.-]+)'
    # Find all matches in the SRT content
    matches = re.findall(pattern, srt_content, re.DOTALL)
    
    # Convert matches to a DataFrame
    df = pd.DataFrame(matches, columns=['FrameCnt', 'Latitude', 'Longitude', 'Rel_Alt', 'Abs_Alt'])
    # Convert relevant columns to float
    df[['FrameCnt', 'Latitude', 'Longitude', 'Rel_Alt', 'Abs_Alt']] = df[['FrameCnt', 'Latitude', 'Longitude', 'Rel_Alt', 'Abs_Alt']].astype(float)
    # Rename 'FrameCnt' column to 'frame'
    df.rename(columns={'FrameCnt':'frame'}, inplace=True)
    # Add the filename as a new column
    df['filename'] = filename
    return df