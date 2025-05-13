import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def select_visualization_columns(df, llm):
    """
    Intelligently select the best columns for visualization based on data types.
    
    Args:
        df: DataFrame containing the data
        llm: Language model for analyzing column selection
        
    Returns:
        tuple: (x_axis_column, y_axis_column)
    """
    columns = df.columns.tolist()
    
    # Prioritize numeric columns for y-axis and categorical for x-axis manually 
    # if the LLM selection fails
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Default fallback values
    default_x = categorical_cols[0] if categorical_cols else columns[0]
    default_y = numeric_cols[0] if numeric_cols else (columns[1] if len(columns) > 1 else columns[0])
    
    visual_columns = f'''
    You are an expert data analyst specializing in data visualization. Your task is to intelligently identify the most appropriate columns for creating meaningful visualizations from the provided dataset.

    Dataset columns: {columns}

    ## Column Selection Guidelines:

    ### For x-axis (categorical/temporal axis):
    - PRIORITIZE columns that represent:
      - Categories (names, product types, regions, departments)
      - Time dimensions (dates, months, years, timestamps)
      - Discrete entities (customer IDs, store locations)
      - Ordinal values (rankings, ratings, priorities)
      - TEXT or VARCHAR columns that contain identifying information
    
    ### For y-axis (measure/value axis):
    - PRIORITIZE columns that represent:
      - Numerical quantities that can be aggregated (sales, counts, amounts)
      - Metrics, measurements, or KPIs
      - Percentages, ratios, or rates
      - Values that would make sense to analyze trends for
      - MUST be numeric columns, never text/categorical values
    
    ### IMPORTANT RULES:
    1. The y-axis MUST be a numeric column suitable for aggregation or measurement
    2. Text/string/varchar columns should almost always be on the x-axis
    3. Names of people or entities should ALWAYS be on the x-axis, never the y-axis
    4. If presented with names and numeric values, put names on x-axis and numeric values on y-axis
    5. When in doubt, choose the most meaningful numeric metric for y-axis

    ### Selection Logic:
    1. First, identify definite y-axis columns (numerical columns that represent measures)
    2. Then, identify definite x-axis columns (categorical/temporal columns that provide context)
    3. Choose the combination that would create the most insightful visualization

    Provide your analysis as a JSON object with exactly these fields:
    ```json
    {
    "x_axis": "SELECTED_X_AXIS_COLUMN",
    "y_axis": "SELECTED_Y_AXIS_COLUMN"
    }
    ```
    Where:
    - SELECTED_X_AXIS_COLUMN is the name of the column best suited for the x-axis (categorical/names/identifiers)
    - SELECTED_Y_AXIS_COLUMN is the name of the column best suited for the y-axis (numerical/metrics)
    '''
            
    try:
        columns_to_visual = llm.invoke(visual_columns)
        visual_data = json.loads(columns_to_visual.content)
        x_axis = visual_data.get("x_axis")
        y_axis = visual_data.get("y_axis")
        
        # Validate that x and y are actually in the dataframe
        if x_axis not in df.columns:
            x_axis = default_x
        if y_axis not in df.columns:
            y_axis = default_y
            
        # Ensure y is numeric when possible
        if y_axis not in numeric_cols and numeric_cols:
            y_axis = numeric_cols[0]
            
    except (json.JSONDecodeError, AttributeError) as e:
        x_axis = default_x
        y_axis = default_y
    
    return x_axis, y_axis

def create_line_chart(df, x_axis, y_axis):
    """Create a line chart with the given x and y axes."""
    try:
        # Limit to a reasonable number of data points
        chart_data = df.head(500) if len(df) > 500 else df
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check if x_axis is numeric or categorical
        if pd.api.types.is_numeric_dtype(chart_data[x_axis]):
            ax.plot(chart_data[x_axis], chart_data[y_axis], marker='o', linestyle='-', color='green')
        else:
            # Sort data if categorical
            if len(chart_data) > 15:  # Only pick top 15 if too many categories
                # Get top values by the y_axis metric
                top_categories = chart_data.groupby(x_axis)[y_axis].sum().nlargest(15).index
                filtered_data = chart_data[chart_data[x_axis].isin(top_categories)]
                ax.plot(filtered_data[x_axis], filtered_data[y_axis], marker='o', linestyle='-', color='green')
            else:
                ax.plot(chart_data[x_axis], chart_data[y_axis], marker='o', linestyle='-', color='green')
        
        # Set labels and grid
        ax.set_xlabel(x_axis.upper())
        ax.set_ylabel(y_axis.upper())
        ax.set_title('Line Chart')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if needed
        if not pd.api.types.is_numeric_dtype(chart_data[x_axis]) and len(chart_data[x_axis].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating line chart: {e}")
        return None

def create_bar_chart(df, x_axis, y_axis):
    """Create a bar chart with the given x and y axes."""
    try:
        # Limit to a reasonable number of data points
        chart_data = df.head(500) if len(df) > 500 else df
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For categorical x_axis, we might want to aggregate the data
        if not pd.api.types.is_numeric_dtype(chart_data[x_axis]):
            if len(chart_data[x_axis].unique()) > 15:  # Too many categories
                # Get top 15 categories by the y_axis metric
                aggregated = chart_data.groupby(x_axis)[y_axis].sum().nlargest(15)
                aggregated.plot(kind='bar', ax=ax, color='skyblue')
            else:
                chart_data.plot(x=x_axis, y=y_axis, kind='bar', ax=ax, color='skyblue')
        else:
            # For numeric x_axis, create bins
            chart_data.plot(x=x_axis, y=y_axis, kind='bar', ax=ax, color='skyblue')
        
        # Set labels and grid
        ax.set_xlabel(x_axis.upper())
        ax.set_ylabel(y_axis.upper())
        ax.set_title('Bar Chart')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        return None

def create_area_chart(df, x_axis, y_axis):
    """Create an area chart with the given x and y axes."""
    try:
        # Limit to a reasonable number of data points
        chart_data = df.head(500) if len(df) > 500 else df
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check if x_axis is numeric or categorical
        if pd.api.types.is_numeric_dtype(chart_data[x_axis]):
            ax.fill_between(chart_data[x_axis], chart_data[y_axis], alpha=0.5, color='green')
            ax.plot(chart_data[x_axis], chart_data[y_axis], color='darkgreen')
        else:
            # Filter if too many categories
            if len(chart_data[x_axis].unique()) > 15:
                # Get top values by the y_axis metric
                top_categories = chart_data.groupby(x_axis)[y_axis].sum().nlargest(15).index
                filtered_data = chart_data[chart_data[x_axis].isin(top_categories)]
                
                # Ensure proper ordering
                cat_order = filtered_data.groupby(x_axis)[y_axis].sum().sort_values().index
                filtered_data = filtered_data.set_index(x_axis).loc[cat_order].reset_index()
                
                ax.fill_between(range(len(filtered_data)), filtered_data[y_axis], alpha=0.5, color='green')
                ax.plot(range(len(filtered_data)), filtered_data[y_axis], color='darkgreen')
                plt.xticks(range(len(filtered_data)), filtered_data[x_axis])
            else:
                # Ensure proper ordering for fewer categories
                cat_order = chart_data.groupby(x_axis)[y_axis].sum().sort_values().index
                ordered_data = chart_data.set_index(x_axis).loc[cat_order].reset_index()
                
                ax.fill_between(range(len(ordered_data)), ordered_data[y_axis], alpha=0.5, color='green')
                ax.plot(range(len(ordered_data)), ordered_data[y_axis], color='darkgreen')
                plt.xticks(range(len(ordered_data)), ordered_data[x_axis])
        
        # Set labels and grid
        ax.set_xlabel(x_axis.upper())
        ax.set_ylabel(y_axis.upper())
        ax.set_title('Area Chart')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels if needed
        if not pd.api.types.is_numeric_dtype(chart_data[x_axis]) and len(chart_data[x_axis].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating area chart: {e}")
        return None