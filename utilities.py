import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from typing import List, Tuple, Union

COLORS = {'PRETTY_BLUE': '#3D6FFF',
          'PRETTY_RED': '#FF3D3D',
          'PRETTY_ORANGE': '#FF8E35',
          'PRETTY_PURPLE': '#BB58FF'
}

def plot_missing_data(dataframe: pd.DataFrame, 
                      nan_values: List[Union[int, float, str]] = None,
                      figsize: Tuple[int, int] = (10, 6),
                      color: str = COLORS['PRETTY_BLUE'],
                      filepath: str = None) -> pd.DataFrame:
    """
    Generate a summary of missing data in a DataFrame and visualize it.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input DataFrame for which missing data analysis will be performed.

    nan_values : List[int, float, str], optional
        The list of values to be considered as NaN. Defaults to None.

    figsize : Tuple[int, int], optional
        The size of the figure for the visualization. Defaults to (10, 6).

    color : str, optional
        The color for the visualization bars. Defaults to COLORS['PRETTY_BLUE'].

    filepath : str, optional
        The file path where the generated visualization will be saved. 
        If provided, the figure will be saved as a PNG file. Defaults to None.

    Returns:
    --------
    pandas DataFrame
        DataFrame containing columns 'Total Missing' and 'Percentage Missing',
        sorted by 'Percentage Missing' in descending order.

    This function takes a pandas DataFrame as input and calculates the total number
    and percentage of missing values for each column in the DataFrame. It creates
    a visualization using seaborn to display the percentage of missing values for
    columns with missing data. The function returns a DataFrame summarizing the 
    missing data statistics for all columns in the input DataFrame.
    """

    if nan_values is not None:
        total_missing = dataframe.isnull().sum()
        for nan_value in nan_values:
            total_missing += (dataframe == nan_value).sum()
    else:
        total_missing = dataframe.isnull().sum()

    percentage_missing = total_missing / dataframe.shape[0]
    
    missing_info_df = pd.DataFrame({'Total Missing': total_missing, 
                                    'Percentage Missing': percentage_missing})

    missing_info_df.sort_values(by='Percentage Missing', ascending=False, inplace=True)
    
    filtered_missing_info_df = missing_info_df[missing_info_df['Percentage Missing'] > 0]

    plt.figure(figsize=figsize)
    sns.barplot(y='Percentage Missing', 
                x=filtered_missing_info_df.index, 
                data=filtered_missing_info_df, 
                color=color)

    plt.title('Percentage of Missing Values by Feature')
    plt.ylabel('Percentage Missing')
    plt.xlabel('Feature')

    if filepath:
        if not filepath.endswith('.png'):
            filepath += '.png'
        plt.savefig(filepath, bbox_inches="tight")

    plt.show()

    return missing_info_df


def plot_groupby(dataframe: pd.DataFrame, 
                 group: str,
                 result_label: str,
                 figsize: Tuple[int, int] = (10, 6),
                 filepath: str = None) -> pd.DataFrame:
    """
    Generate a grouped bar plot based on DataFrame aggregation.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input DataFrame containing the data for analysis.

    group : str
        The column name by which the DataFrame will be grouped.

    result_label : str
        The column label for which statistics will be calculated and visualized.

    figsize : Tuple[int, int], optional
        The size of the figure for the visualization. Defaults to (10, 6).

    filepath : str, optional
        The file path where the generated visualization will be saved. 
        If provided, the figure will be saved as a PNG file. Defaults to None.

    Returns:
    --------
    pandas DataFrame
        DataFrame containing aggregated statistics based on the provided group
        column and the result_label.

    This function groups the input DataFrame based on a specified column ('group')
    and calculates aggregated statistics ('result_label') for each group. It creates
    a grouped bar plot using seaborn to visualize the calculated statistics. The 
    function returns a DataFrame summarizing the aggregated statistics for each group.
    """
    
    groups = dataframe.groupby(group)

    percentage_by_group = groups[result_label].mean() * 100
    number_by_group = groups[result_label].sum()

    total_by_group = groups.size()
    
    df = pd.DataFrame({
        'Group Percentage': percentage_by_group,
        'Total': total_by_group,
        'Grouped total': number_by_group
    })

    plt.figure(figsize= figsize)
    sns.barplot(y=f'{result_label.capitalize()} Percentage', x=df.index, data=df, color=COLORS['PRETTY_BLUE'])
    plt.title(f'{result_label.capitalize()} Percentage by {group.capitalize()} Category')
    plt.ylabel(f'{result_label.capitalize()} Percentage')
    plt.xlabel(f'{group.capitalize()} Category')

    if filepath:
        if not filepath.endswith('.png'):
            filepath += '.png'
        plt.savefig(filepath, bbox_inches= "tight")

    plt.show()

    return df

def plot_hist_discrete_feature(df: pd.DataFrame, 
                               column: str,
                               frequency: bool = False,
                               figsize: Tuple[int, int] = (10, 6),
                               filepath: str = None,
                               **kwargs) -> None:
    """
    Plot a histogram for a specified column in a DataFrame with customizable options.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    column (str):
        The name of the column to plot.

    frequency (bool, optional):
        If True, plot frequencies instead of counts. Defaults to False.

    figsize (Tuple[int, int], optional):
        The size of the figure (width, height) in inches. Defaults to (10, 6).

    filepath (str, optional):
        The file path to save the plot as an image. If provided, the figure will be saved as a PNG file.
        Defaults to None.

    **kwargs:
        Additional keyword arguments for customization (e.g., alpha, edgecolor, bins, color, etc.).

    Returns:
    --------
    None

    This function plots a histogram for the specified column in the DataFrame. It provides options
    to customize the appearance of the plot, such as figure size, colors, and transparency. If a file
    path is provided, the plot will be saved as an image in PNG format.
    """
    plt.figure(figsize=figsize)

    labels, counts = np.unique(df[column], return_counts=True)
    
    if frequency:
        total_observations = len(df[column])
        counts = counts / total_observations

    plt.bar(labels, counts, **kwargs)
    plt.gca().set_xticks(labels)

    if frequency:
        plt.title(f'Frequency Histogram of {column}')
        plt.ylabel('Frequency')
    else:
        plt.title(f'Histogram of {column}')
        plt.ylabel('Count')

    plt.xlabel('Values')

    if filepath:
        if not filepath.endswith('.png'):
            filepath += '.png'
        plt.savefig(filepath, bbox_inches="tight")

    plt.show()


def min_max_scale_column(df: pd.DataFrame,
                         column: str,
                         new_min: int,
                         new_max: int,
                         add_noise: bool = False,
                         noise_factor: float = 0.0,
                         convert_to_int: bool = False,
                         inplace: bool = True) -> pd.DataFrame:

    """
    Scale a specified column in a DataFrame using Min-Max scaling and provide optional features.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    column (str):
        The name of the column to scale.

    new_min (int):
        The desired minimum value after scaling.

    new_max (int):
        The desired maximum value after scaling.

    add_noise (bool, optional):
        If True, add random noise to the scaled values. Defaults to False.

    noise_factor (float, optional):
        The standard deviation of the random noise to be added. Defaults to 0.0.

    convert_to_int (bool, optional):
        If True, round the scaled values and convert them to integers. Defaults to False.

    inplace (bool, optional):
        If True, modify the DataFrame in place. If False, create a copy of the DataFrame.
        Defaults to True.

    Returns:
    --------
    pd.DataFrame or None:
        Returns the modified DataFrame if inplace is True, otherwise, returns a new DataFrame.

    This function scales the specified column using Min-Max scaling and provides optional features
    such as adding random noise and rounding to integers. The scaled values are rounded to the nearest
    half to ensure granularity. If specified, random noise is added, and the values can be converted
    to integers. The modified DataFrame is returned if inplace is True; otherwise, a new DataFrame
    is returned.
    """
    
    if not inplace:
        df = df.copy()

    # Extract the column values as a 2D array for MinMaxScaler
    values = df[[column]].values

    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=(new_min, new_max))

    scaled_values = scaler.fit_transform(values)
    rounded_values = np.round(scaled_values * 2) / 2

    # Flatten the scaled values array and assign it back to the DataFrame
    df[column] = rounded_values.flatten()

    # Add noise if specified
    if add_noise:
        noise = np.random.normal(scale=noise_factor, size=len(df))
        df[column] += noise


    # Convert to int if specified
    if convert_to_int:
        df[column] = df[column].round().astype(int)

    # Return the modified DataFrame if inplace is False
    if not inplace:
        return df

def categorize_column(df: pd.DataFrame,
                      column: str,
                      int_bins: list,
                      categorial_labels: list,
                      handle_nan: bool = False,
                      replace_original: bool = True,
                      inplace: bool = True) -> pd.DataFrame:
    """
    Categorize a numerical column in a DataFrame into specified bins and labels.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    column (str):
        The name of the column to categorize.

    int_bins (list):
        The bin edges for categorizing the numerical values in the column.

    categorial_labels (list):
        The labels corresponding to the bins for categorizing the values.

    handle_nan (bool, optional):
        If True, add a category 'Unknown' for NaN values. Defaults to False.

    replace_original (bool, optional):
        If True, replace the original numerical column after categorization. Defaults to True.

    inplace (bool, optional):
        If True, modify the DataFrame in place. If False, create a copy of the DataFrame.
        Defaults to True.

    Returns:
    --------
    pd.DataFrame or None:
        Returns the modified DataFrame if inplace is True, otherwise, returns a new DataFrame.

    This function categorizes the specified numerical column into bins with corresponding labels.
    Optionally, it can handle NaN values by adding a category 'Unknown'. The modified DataFrame is
    returned if inplace is True; otherwise, a new DataFrame is returned.
    """
    
    assert len(int_bins) == len(categorial_labels) + 1, "Length of int_bins and categorial_labels must match."

    if not inplace:
        df = df.copy()

    str_to_add = '' if replace_original else '_bin'


    # Categorize the numerical column into specified bins and labels
    df[column + str_to_add] = pd.cut(df[column], bins=int_bins, labels=categorial_labels, right=False)

    # Deal with NaN values if specified
    if handle_nan:
        df[column + str_to_add] = df[column + str_to_add].cat.add_categories('Unknown').fillna('Unknown')

    # Return the modified DataFrame if inplace is False
    if not inplace:
        return df
    
def plot_PCA(df: pd.DataFrame, 
             size: Tuple[int, int] = (10, 8), 
             filepath: str = None,
             **kwargs) -> np.ndarray:
    
    """
    Plot the Explained Variance Ratio by Principal Components using PCA on the DataFrame.

    Parameters:
    -----------
    df (pd.DataFrame):
        The DataFrame containing the data.

    size (Tuple[int, int], optional):
        The size of the plot (width, height). Defaults to (10, 8).

    filepath (str, optional):
        If provided, saves the plot as a PNG file at the specified filepath. Defaults to None.

    **kwargs:
        Additional keyword arguments to pass to the plt.bar function.

    Returns:
    --------
    np.ndarray:
        Returns the explained variance ratio of Principal Components obtained from PCA.

    This function performs Principal Component Analysis (PCA) on the DataFrame and plots
    the Explained Variance Ratio by Principal Components. It also returns the explained 
    variance ratio for each principal component obtained from PCA.
    """
    
    pca = PCA()
    pca.fit(df)

    explained_variance_ratio = pca.explained_variance_ratio_
    
    principal_components = range(1, len(explained_variance_ratio) + 1)
    
    plt.figure(figsize=size)

    plt.bar(principal_components, 
            explained_variance_ratio, 
            **kwargs)
    
    plt.xlabel('Principal Components', fontweight='bold')
    plt.ylabel('Explained Variance Ratio', fontweight='bold')
    plt.title('Explained Variance Ratio by Principal Component', fontweight='bold')
    
    plt.xticks(principal_components)

    if filepath:
        if not filepath.endswith('.png'):
            filepath += '.png'
        plt.savefig(filepath, bbox_inches="tight")

    plt.show()

    return explained_variance_ratio


def plot_cumulative_explained_variance(explained_variance_ratio: np.ndarray,
                                       size: Tuple[int, int] = (10, 8),
                                       filepath: str = None,
                                       **kwargs) -> None:
    """
    Plot the Cumulative Explained Variance by Principal Components.

    Parameters:
    -----------
    explained_variance_ratio (np.ndarray):
        The array containing the explained variance ratio for each principal component.

    size (Tuple[int, int], optional):
        The size of the plot (width, height). Defaults to (10, 8).

    filepath (str, optional):
        If provided, saves the plot as a PNG file at the specified filepath. Defaults to None.

    **kwargs:
        Additional keyword arguments to pass to the plt.plot function.

    Returns:
    --------
    None

    This function plots the Cumulative Explained Variance by Principal Components using the
    explained variance ratio array obtained from PCA.
    """

    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    principal_components = range(1, len(explained_variance_ratio) + 1)

    # Plot the cumulative explained variance
    plt.figure(figsize=size)
    plt.plot(principal_components, 
             cumulative_explained_variance, 
             **kwargs)

    plt.xlabel('Number of Principal Components', fontweight='bold')
    plt.ylabel('Cumulative Explained Variance', fontweight='bold')
    plt.title('Cumulative Explained Variance by Principal Components', fontweight='bold')

    plt.xticks(principal_components)

    plt.grid(True)

    if filepath:
        if not filepath.endswith('.png'):
            filepath += '.png'
        plt.savefig(filepath, bbox_inches="tight")

    plt.show()