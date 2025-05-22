import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_visualization(df, x_axis, y_axis, visual_type="bar"):
    """
    Create the appropriate visualization based on the specified type and axes.
    
    Args:
        df: DataFrame containing the data
        x_axis: Column name to use for x-axis
        y_axis: Column name to use for y-axis
        visual_type: Type of visualization to create
        
    Returns:
        fig: Matplotlib figure object
    """
    # Limit to a reasonable number of data points
    chart_data = df.head(500) if len(df) > 500 else df
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # Handle different visualization types
        if visual_type == "line":
            create_line_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "bar":
            create_bar_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "area":
            create_area_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "scatter":
            create_scatter_plot(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "histogram":
            create_histogram(chart_data, y_axis, ax)
            
        elif visual_type == "box":
            create_box_plot(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "pie":
            create_pie_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "heatmap":
            create_heatmap(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "bubble":
            create_bubble_chart(chart_data, x_axis, y_axis, ax)
            
        elif visual_type == "grouped_bar":
            create_grouped_bar(chart_data, x_axis, y_axis, ax)
            
        else:
            # Default to bar chart if unsupported visualization type
            create_bar_chart(chart_data, x_axis, y_axis, ax)
        
        # Common chart formatting
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel(y_axis.replace('_', ' ').title())
        ax.set_title(f'{visual_type.replace("_", " ").title()} Chart: {y_axis.replace("_", " ").title()} by {x_axis.replace("_", " ").title()}')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating {visual_type} chart: {e}")
        # Create a fallback visualization
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Could not create visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def create_line_chart(df, x_axis, y_axis, ax):
    """Create a line chart with the given x and y axes."""
    # Handle datetime x-axis
    if pd.api.types.is_datetime64_any_dtype(df[x_axis]):
        df = df.sort_values(by=x_axis)
        ax.plot(df[x_axis], df[y_axis], marker='o', linestyle='-', color='#1f77b4')
    
    # Handle numeric x-axis
    elif pd.api.types.is_numeric_dtype(df[x_axis]):
        df = df.sort_values(by=x_axis)
        ax.plot(df[x_axis], df[y_axis], marker='o', linestyle='-', color='#1f77b4')
    
    # Handle categorical x-axis
    else:
        # If too many categories, aggregate and take top N
        if df[x_axis].nunique() > 15:
            grouped = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False).head(15)
            ax.plot(grouped.index, grouped.values, marker='o', linestyle='-', color='#1f77b4')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        else:
            grouped = df.groupby(x_axis)[y_axis].mean()
            ax.plot(grouped.index, grouped.values, marker='o', linestyle='-', color='#1f77b4')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_bar_chart(df, x_axis, y_axis, ax):
    """Create a bar chart with the given x and y axes."""
    # If too many categories, aggregate and take top N
    if not pd.api.types.is_numeric_dtype(df[x_axis]) and df[x_axis].nunique() > 15:
        grouped = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False).head(15)
        grouped.plot(kind='bar', ax=ax, color='#2ca02c')
    elif pd.api.types.is_numeric_dtype(df[x_axis]) and df[x_axis].nunique() > 15:
        # For numeric x with many values, create bins
        bins = min(15, int(df[x_axis].nunique() / 5) + 1)
        df['binned'] = pd.cut(df[x_axis], bins=bins)
        grouped = df.groupby('binned')[y_axis].mean()
        grouped.plot(kind='bar', ax=ax, color='#2ca02c')
    else:
        # Direct plotting for manageable number of categories
        grouped = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False)
        grouped.plot(kind='bar', ax=ax, color='#2ca02c')
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def create_area_chart(df, x_axis, y_axis, ax):
    """Create an area chart with the given x and y axes."""
    # Handle datetime or numeric x-axis
    if pd.api.types.is_datetime64_any_dtype(df[x_axis]) or pd.api.types.is_numeric_dtype(df[x_axis]):
        df = df.sort_values(by=x_axis)
        ax.fill_between(df[x_axis], df[y_axis], alpha=0.5, color='#9467bd')
        ax.plot(df[x_axis], df[y_axis], color='#7f5e91')
    
    # Handle categorical x-axis
    else:
        # If too many categories, aggregate and take top N
        if df[x_axis].nunique() > 15:
            grouped = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False).head(15)
            x_vals = range(len(grouped))
            ax.fill_between(x_vals, grouped.values, alpha=0.5, color='#9467bd')
            ax.plot(x_vals, grouped.values, color='#7f5e91')
            ax.set_xticks(x_vals)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        else:
            grouped = df.groupby(x_axis)[y_axis].mean()
            x_vals = range(len(grouped))
            ax.fill_between(x_vals, grouped.values, alpha=0.5, color='#9467bd')
            ax.plot(x_vals, grouped.values, color='#7f5e91')
            ax.set_xticks(x_vals)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_scatter_plot(df, x_axis, y_axis, ax):
    """Create a scatter plot with the given x and y axes."""
    # Ensure both axes are numeric
    if not pd.api.types.is_numeric_dtype(df[x_axis]):
        raise ValueError(f"X-axis column '{x_axis}' must be numeric for scatter plots")
    if not pd.api.types.is_numeric_dtype(df[y_axis]):
        raise ValueError(f"Y-axis column '{y_axis}' must be numeric for scatter plots")
    
    # Create scatter plot
    ax.scatter(df[x_axis], df[y_axis], alpha=0.6, color='#ff7f0e', edgecolor='k', linewidth=0.5)
    
    # Add best fit line
    try:
        z = np.polyfit(df[x_axis], df[y_axis], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[x_axis].min(), df[x_axis].max(), 100)
        ax.plot(x_range, p(x_range), "r--", alpha=0.8)
    except:
        pass  # Skip trend line if there's an error
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_histogram(df, column, ax):
    """Create a histogram for the specified column."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric for histograms")
    
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color='#d62728')
    ax.set_xlabel(column.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)

def create_box_plot(df, x_axis, y_axis, ax):
    """Create a box plot with the given x and y axes."""
    if not pd.api.types.is_numeric_dtype(df[y_axis]):
        raise ValueError(f"Y-axis column '{y_axis}' must be numeric for box plots")
    
    # If too many categories on x-axis, limit to top N by average y value
    if not pd.api.types.is_numeric_dtype(df[x_axis]) and df[x_axis].nunique() > 10:
        top_categories = df.groupby(x_axis)[y_axis].mean().sort_values(ascending=False).head(10).index
        filtered_df = df[df[x_axis].isin(top_categories)]
        sns.boxplot(x=x_axis, y=y_axis, data=filtered_df, ax=ax, palette='viridis')
    else:
        sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax, palette='viridis')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

def create_pie_chart(df, x_axis, y_axis, ax):
    """Create a pie chart with the given x and y axes."""
    # Aggregate data for pie chart
    grouped = df.groupby(x_axis)[y_axis].sum()
    
    # If too many categories, show top N and group others
    if grouped.shape[0] > 7:
        top = grouped.nlargest(6)
        others = pd.Series({'Others': grouped.sum() - top.sum()})
        combined = pd.concat([top, others])
        
        # Create pie chart with top 6 categories + "Others"
        combined.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, 
                         colors=plt.cm.tab10.colors, shadow=False)
    else:
        # Create pie chart with all categories
        grouped.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, 
                        colors=plt.cm.tab10.colors, shadow=False)
    
    ax.set_ylabel('')  # Remove y-axis label for pie chart

def create_heatmap(df, x_axis, y_axis, ax):
    """Create a heatmap with the given x and y axes."""
    # If both axes are categorical
    if not pd.api.types.is_numeric_dtype(df[x_axis]) and not pd.api.types.is_numeric_dtype(df[y_axis]):
        # Create pivot table with count of occurrences
        pivot_data = pd.crosstab(df[y_axis], df[x_axis])
        
        # Limit categories if too many
        if pivot_data.shape[0] > 15 or pivot_data.shape[1] > 15:
            # For columns (x-axis)
            if pivot_data.shape[1] > 15:
                top_cols = pivot_data.sum().nlargest(15).index
                pivot_data = pivot_data[top_cols]
            
            # For rows (y-axis)
            if pivot_data.shape[0] > 15:
                top_rows = pivot_data.sum(axis=1).nlargest(15).index
                pivot_data = pivot_data.loc[top_rows]
        
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='g', ax=ax)
        
    # If one axis is categorical and one is numeric
    elif not pd.api.types.is_numeric_dtype(df[x_axis]) or not pd.api.types.is_numeric_dtype(df[y_axis]):
        # Determine which is categorical and which is numeric
        cat_col = x_axis if not pd.api.types.is_numeric_dtype(df[x_axis]) else y_axis
        num_col = y_axis if cat_col == x_axis else x_axis
        
        # Create pivot table with mean of numeric column
        pivot_data = df.pivot_table(index=cat_col, values=num_col, aggfunc='mean')
        
        # Limit categories if too many
        if pivot_data.shape[0] > 15:
            pivot_data = pivot_data.nlargest(15, num_col)
            
        # Create heatmap as a 1D visualization
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f', ax=ax)
        
    # If both axes are numeric, create correlation heatmap
    else:
        corr = df[[x_axis, y_axis]].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

def create_bubble_chart(df, x_axis, y_axis, ax):
    """Create a bubble chart with the given x and y axes."""
    # Ensure both axes are numeric
    if not pd.api.types.is_numeric_dtype(df[x_axis]) or not pd.api.types.is_numeric_dtype(df[y_axis]):
        raise ValueError(f"Both axes must be numeric for bubble charts")
    
    # Find a suitable column for bubble size
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    size_col = None
    
    # Try to find a numeric column different from x_axis and y_axis
    for col in numeric_cols:
        if col != x_axis and col != y_axis:
            size_col = col
            break
    
    # If no third numeric column is found, use y_axis values for size
    if size_col is None:
        size_col = y_axis
    
    # Normalize size to make bubbles proportional
    sizes = df[size_col].values
    size_multiplier = 100
    if sizes.min() == sizes.max():
        bubble_sizes = np.ones_like(sizes) * size_multiplier
    else:
        bubble_sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * size_multiplier) + 10
    
    # Create scatter plot with varying bubble sizes
    scatter = ax.scatter(df[x_axis], df[y_axis], s=bubble_sizes, alpha=0.6,
                         c=df[y_axis], cmap='viridis', edgecolor='k', linewidth=0.5)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=y_axis.replace('_', ' ').title())
    
    # Add size legend
    if size_col != y_axis:
        ax.text(0.95, 0.05, f'Bubble size: {size_col}', transform=ax.transAxes,
                ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.grid(True, linestyle='--', alpha=0.7)

def create_grouped_bar(df, x_axis, y_axis, ax):
    """Create a grouped bar chart with the given x and y axes."""
    # Need a third categorical column to group by
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Find a suitable categorical column different from x_axis
    group_col = None
    for col in categorical_cols:
        if col != x_axis and df[col].nunique() < 7:  # Limit to columns with few categories
            group_col = col
            break
    
    # If no suitable grouping column is found, create a basic bar chart
    if group_col is None:
        create_bar_chart(df, x_axis, y_axis, ax)
        return
    
    # If too many categories on x-axis, limit to top N
    if df[x_axis].nunique() > 8:
        top_categories = df.groupby(x_axis)[y_axis].mean().nlargest(8).index
        filtered_df = df[df[x_axis].isin(top_categories)]
    else:
        filtered_df = df
    
    # Create pivot table for grouped bar chart
    pivot_data = filtered_df.pivot_table(index=x_axis, columns=group_col, values=y_axis, aggfunc='mean')
    
    # Plot grouped bar chart
    pivot_data.plot(kind='bar', ax=ax)
    
    # Add legend and formatting
    ax.legend(title=group_col.replace('_', ' ').title())
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
