import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        'Survival Percentage': percentage_by_group,
        'Total Passengers': total_by_group,
        'Survived Passengers': number_by_group
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