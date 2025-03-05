import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Import our data processing functions
from data_processing import (prepare_gtd_data, perform_pca, perform_kmeans, 
                            find_elbow_point, get_top_features)

# Initialize the Dash app
app = dash.Dash(__name__, title="Global Terrorism Database Visualization")
server = app.server  # For deployment

# Load and preprocess the data
# In app.py, update the data loading and processing section:

# Load and preprocess the data
try:
    file_path = "data/globalterrorismdb.xlsx"
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 1: Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Found {len(numerical_cols)} numerical columns")
    
    # Step 2: Filter to recent years
    if 'iyear' in df.columns:
        df_filtered = df[df['iyear'] >= 2016].copy()
        print(f"Filtered to recent years: {df_filtered.shape[0]} incidents")
    else:
        df_filtered = df.copy()
    
    # Step 3: Sample to manageable size if needed
    if len(df_filtered) > 10000:
        df_sample = df_filtered.sample(10000, random_state=42)
        print(f"Sampled to 10000 incidents")
    else:
        df_sample = df_filtered
    
    # Step 4: Select columns for analysis
    if len(numerical_cols) > 20:
        # First check for columns with missing values
        missing_counts = df_sample[numerical_cols].isnull().sum()
        # Prioritize columns with fewer missing values and higher variance
        valid_cols = [col for col in numerical_cols if missing_counts[col] / len(df_sample) < 0.3]
        print(f"Identified {len(valid_cols)} columns with less than 30% missing values")
        
        # Calculate variance for columns with acceptable missing value rates
        variances = df_sample[valid_cols].var()
        top_cols = variances.nlargest(20).index.tolist()
        print(f"Selected top 20 columns with highest variance")
        
        # Use these columns for analysis
        analysis_cols = top_cols
    else:
        analysis_cols = numerical_cols
    
    # Step 5: Create a clean dataframe for analysis
    df_clean = df_sample[analysis_cols].copy()
    
    # Step 6: Handle missing values for selected columns
    for col in analysis_cols:
        if df_clean[col].isnull().sum() > 0:
            print(f"Replacing missing values in column: {col}")
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Step 7: Check for any remaining issues
    if df_clean.isnull().sum().sum() > 0:
        print(f"Warning: Still have {df_clean.isnull().sum().sum()} missing values after cleaning")
        df_clean = df_clean.fillna(0)  # Last resort: fill any remaining with zeros
    
    print(f"Final cleaned data shape: {df_clean.shape}")
    
    # Perform PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    print(f"Data standardized, shape: {scaled_data.shape}")
    
    # Continue with PCA and clustering as before
    pca, pca_result, explained_variance, loadings, scaled_data = perform_pca(df_clean)
    inertia, kmeans_models = perform_kmeans(scaled_data)
    
    # Find elbow points
    elbow_idx_dim = find_elbow_point(explained_variance)
    elbow_idx_k = find_elbow_point(inertia)
    
    # Get feature names for display
    feature_names = analysis_cols
    
    print(f"PCA and clustering complete. Recommended dimensions: {elbow_idx_dim}, K: {elbow_idx_k}")
    
except Exception as e:
    print(f"Error processing data: {e}")
    # Create placeholder data if loading fails
    df = pd.DataFrame()
    df_clean = pd.DataFrame()
    pca, pca_result, explained_variance, loadings, scaled_data = None, None, [], [], None
    inertia, kmeans_models = [], {}
    elbow_idx_dim, elbow_idx_k = 3, 3
    feature_names = []
    analysis_cols = []

# Define the layout
app.layout = html.Div([
    html.H1("Global Terrorism Database Analysis", 
            style={'textAlign': 'center', 'margin': '20px', 'color': '#2c3e50'}),
    
    html.Div([
        html.H2("PCA and Clustering Analysis", style={'margin': '10px', 'color': '#34495e'}),
        
        html.Div([
            # Left column for PCA scree plot
            html.Div([
                html.H3("PCA Eigenvalues (Scree Plot)", 
                        style={'textAlign': 'center', 'color': '#3498db'}),
                dcc.Graph(id='scree-plot'),
                html.P("Click on a bar to select intrinsic dimensionality",
                      style={'textAlign': 'center', 'fontStyle': 'italic'})
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'boxShadow': '0px 0px 5px #ccc'}),
            
            # Right column for K-means elbow plot
            html.Div([
                html.H3("K-Means Clustering (Elbow Plot)", 
                        style={'textAlign': 'center', 'color': '#e74c3c'}),
                dcc.Graph(id='elbow-plot'),
                html.P("Click on a point to select K value",
                      style={'textAlign': 'center', 'fontStyle': 'italic'})
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'padding': '10px', 'boxShadow': '0px 0px 5px #ccc'})
        ], style={'margin': '20px 0'}),
        
        # Biplot section
        html.Div([
            html.H3("PCA Biplot - First Two Principal Components", 
                    style={'textAlign': 'center', 'color': '#2ecc71'}),
            dcc.Graph(id='biplot'),
            html.P("Points colored by cluster. Vectors show feature directions.",
                  style={'textAlign': 'center', 'fontStyle': 'italic'})
        ], style={'margin': '30px 0', 'padding': '10px', 'boxShadow': '0px 0px 5px #ccc'}),
        
        # Top PCA components table
        html.Div([
            html.H3("Top Features by PCA Loading", 
                    style={'textAlign': 'center', 'color': '#9b59b6'}),
            dash_table.DataTable(
                id='pca-loadings-table',
                columns=[
                    {'name': 'Feature', 'id': 'feature'},
                    {'name': 'Importance Score', 'id': 'score'},
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ], style={'margin': '30px 0', 'padding': '10px', 'boxShadow': '0px 0px 5px #ccc'}),
        
        # Scatterplot matrix
        html.Div([
            html.H3("Scatterplot Matrix of Top Features", 
                    style={'textAlign': 'center', 'color': '#f39c12'}),
            dcc.Graph(id='scatterplot-matrix'),
            html.P("Points colored by cluster ID. Displays relationships between top features.",
                  style={'textAlign': 'center', 'fontStyle': 'italic'})
        ], style={'margin': '30px 0', 'padding': '10px', 'boxShadow': '0px 0px 5px #ccc'}),
        
        # Store components for state management
        dcc.Store(id='selected-dim', data=elbow_idx_dim if len(explained_variance) > 0 else 3),
        dcc.Store(id='selected-k', data=elbow_idx_k if len(inertia) > 0 else 3),
    ], style={'max-width': '1200px', 'margin': '0 auto', 'padding': '20px'})
])

# Callback for scree plot
@app.callback(
    Output('scree-plot', 'figure'),
    Output('selected-dim', 'data'),
    Input('scree-plot', 'clickData'),
    State('selected-dim', 'data')
)
def update_scree_plot(click_data, current_dim):
    if len(explained_variance) == 0:
        # Return empty plot if no data
        return {}, current_dim
    
    # Determine selected dimension
    selected_dim = current_dim
    if click_data is not None:
        selected_dim = click_data['points'][0]['pointNumber'] + 1
    
    # Create scree plot
    fig = go.Figure()
    
    # Add bars for each component (limit to first 20 for readability)
    max_components = min(20, len(explained_variance))
    
    for i, var in enumerate(explained_variance[:max_components]):
        if i+1 == selected_dim:
            fig.add_trace(go.Bar(
                x=[i+1], y=[var], 
                marker_color='rgba(50, 171, 96, 0.7)',
                name=f'PC{i+1}'
            ))
        else:
            fig.add_trace(go.Bar(
                x=[i+1], y=[var], 
                marker_color='rgba(55, 83, 109, 0.5)',
                name=f'PC{i+1}'
            ))
    
    # Add cumulative variance line
    cumulative = np.cumsum(explained_variance[:max_components])
    fig.add_trace(go.Scatter(
        x=list(range(1, max_components + 1)),
        y=cumulative,
        mode='lines+markers',
        name='Cumulative',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Explained Variance by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        yaxis=dict(
            tickformat='.1%',
            range=[0, max(1, max(cumulative) * 1.1)]
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        showlegend=False,
        clickmode='event+select'
    )
    
    return fig, selected_dim

# Callback for elbow plot
@app.callback(
    Output('elbow-plot', 'figure'),
    Output('selected-k', 'data'),
    Input('elbow-plot', 'clickData'),
    State('selected-k', 'data')
)
# def update_elbow_plot(click_data, current_k):
#     if len(inertia) == 0:
#         # Return empty plot if no data
#         return {}, current_k
    
#     # Determine selected k
#     selected_k = current_k
#     if click_data is not None:
#         # For a bar chart, we need to extract the k value differently
#         selected_k = click_data['points'][0]['x']
    
#     # Create elbow plot as a bar chart
#     fig = go.Figure()
    
#     # Create list of k values
#     k_values = list(range(1, len(inertia) + 1))
    
#     # Add bars for each k value
#     for i, (k, inert) in enumerate(zip(k_values, inertia)):
#         if k == selected_k:
#             fig.add_trace(go.Bar(
#                 x=[k], y=[inert], 
#                 marker_color='rgba(231, 76, 60, 0.7)',  # Red for selected bar
#                 name=f'K={k}'
#             ))
#         else:
#             fig.add_trace(go.Bar(
#                 x=[k], y=[inert], 
#                 marker_color='rgba(52, 152, 219, 0.5)',  # Blue for other bars
#                 name=f'K={k}'
#             ))
    
#     # Add a line to help guide the eye (optional, can be removed if you prefer just bars)
#     fig.add_trace(go.Scatter(
#         x=k_values, y=inertia,
#         mode='lines',
#         line=dict(color='gray', dash='dot'),
#         showlegend=False
#     ))
    
#     fig.update_layout(
#         title='K-Means Elbow Plot - Inertia vs. Number of Clusters',
#         xaxis_title='Number of Clusters (K)',
#         yaxis_title='Inertia (Within-Cluster Sum of Squares)',
#         xaxis=dict(
#             tickmode='linear',
#             tick0=1,
#             dtick=1
#         ),
#         showlegend=False,
#         clickmode='event+select'
#     )
    
#     return fig, selected_k
def update_elbow_plot(click_data, current_k):
    if len(inertia) == 0:
        # Return empty plot if no data
        return {}, current_k
    
    # Determine selected k
    selected_k = current_k
    if click_data is not None:
        selected_k = click_data['points'][0]['x']
    
    # Create elbow plot
    fig = go.Figure()
    
    # Create list of k values
    k_values = list(range(1, len(inertia) + 1))
    
    # Add main points
    for i, (k, inert) in enumerate(zip(k_values, inertia)):
        if k == selected_k:
            fig.add_trace(go.Scatter(
                x=[k], y=[inert],
                mode='markers',
                marker=dict(size=12, color='red'),
                name=f'K={k}'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[k], y=[inert],
                mode='markers',
                marker=dict(size=8, color='blue'),
                name=f'K={k}'
            ))
    
    # Add line connecting points
    fig.add_trace(go.Scatter(
        x=k_values, y=inertia,
        mode='lines',
        line=dict(color='gray'),
        showlegend=False
    ))
    
    fig.update_layout(
        title='K-Means Elbow Plot - Inertia vs. Number of Clusters',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Inertia (Within-Cluster Sum of Squares)',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        showlegend=False,
        clickmode='event+select'
    )
    
    return fig, selected_k

# Callback for PCA loadings table and biplot
# Callback for PCA loadings table and biplot
# Callback for PCA loadings table and biplot
@app.callback(
    [Output('pca-loadings-table', 'data'),
     Output('biplot', 'figure')],
    [Input('selected-dim', 'data'),
     Input('selected-k', 'data')]
)

def update_pca_visualizations(selected_dim, selected_k):
    if len(loadings) == 0 or len(feature_names) == 0:
        # Return empty data if no data
        return [], {}
    
    # Get top features based on loadings
    top_features, top_scores, top_indices = get_top_features(
        loadings, feature_names, selected_dim, n_top_features=4
    )
    
    # Create table data
    table_data = [
        {'feature': feature, 'score': f"{score:.4f}"} 
        for feature, score in zip(top_features, top_scores)
    ]
    
    # Create biplot with cleaner style
    fig = go.Figure()
    
    # Add scatter points
    if selected_k > 1 and kmeans_models and selected_k in kmeans_models:
        # Color by cluster if k-means has been run
        clusters = kmeans_models[selected_k].predict(scaled_data)
        
        # Plot each cluster - use a more distinct color palette
        colors = px.colors.qualitative.D3  # Using a clearer color scheme
        
        for cluster_id in range(selected_k):
            cluster_mask = (clusters == cluster_id)
            
            # Skip empty clusters
            if not any(cluster_mask):
                continue
                
            fig.add_trace(go.Scatter(
                x=pca_result[cluster_mask, 0],
                y=pca_result[cluster_mask, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.7
                ),
                name=f'Cluster {cluster_id + 1}'
            ))
    else:
        # Simple scatter if no clustering
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            marker=dict(size=8),
            name='Data Points'
        ))
    
    # Calculate the magnitude of loadings for scaling
    feature_importance = np.sqrt(loadings[0, :]**2 + loadings[1, :]**2)
    
    # Find top 6 most important features
    top_n = 6
    top_indices = np.argsort(feature_importance)[::-1][:top_n]
    
    # Calculate scaling based on data range
    x_range = np.max(pca_result[:, 0]) - np.min(pca_result[:, 0])
    y_range = np.max(pca_result[:, 1]) - np.min(pca_result[:, 1])
    axis_range = max(x_range, y_range)
    
    # Use a larger scale factor like in the example
    scale_factor = 0.3 * axis_range
    
    # Add vectors for top features only - with cleaner styling
    for i in top_indices:
        feature = feature_names[i]
        
        # Scale the loadings
        x_end = loadings[0, i] * scale_factor
        y_end = loadings[1, i] * scale_factor
        
        # Skip if vector is too small
        if np.sqrt(x_end**2 + y_end**2) < 0.05 * scale_factor:
            continue
            
        # Add the line without text (more like the example)
        fig.add_trace(go.Scatter(
            x=[0, x_end],
            y=[0, y_end],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))
        
        # Add text at the end of the vector
        fig.add_annotation(
            x=x_end,
            y=y_end,
            text=feature,
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            borderpad=2
        )
    
    # Add dotted lines at x=0 and y=0 (like in the example)
    fig.add_shape(
        type="line",
        x0=np.min(pca_result[:, 0])*1.1,
        y0=0,
        x1=np.max(pca_result[:, 0])*1.1,
        y1=0,
        line=dict(color="black", width=1, dash="dot"),
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=np.min(pca_result[:, 1])*1.1,
        x1=0,
        y1=np.max(pca_result[:, 1])*1.1,
        line=dict(color="black", width=1, dash="dot"),
    )
    
    # Update layout for a cleaner look
    fig.update_layout(
        title='PCA Biplot - First Two Principal Components',
        xaxis_title='PC1',
        yaxis_title='PC2',
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
        width=800
    )
    
    # Set axes ranges to be equal (important for vector representation)
    max_range = max(
        abs(np.min(pca_result[:, 0])), abs(np.max(pca_result[:, 0])),
        abs(np.min(pca_result[:, 1])), abs(np.max(pca_result[:, 1]))
    )
    fig.update_xaxes(range=[-max_range*1.1, max_range*1.1])
    fig.update_yaxes(range=[-max_range*1.1, max_range*1.1])
    
    return table_data, fig
# Callback for scatterplot matrix
@app.callback(
    Output('scatterplot-matrix', 'figure'),
    [Input('selected-dim', 'data'),
     Input('selected-k', 'data')]
)
def update_scatterplot_matrix(selected_dim, selected_k):
    if df_clean.empty or len(loadings) == 0:
        # Return empty figure if no data
        return {}
    
    # Get top features
    top_features, _, top_indices = get_top_features(
        loadings, feature_names, selected_dim, n_top_features=4
    )
    
    # Create a dataframe with just these features
    top_data = df_clean.iloc[:, top_indices].copy()
    
    # Add cluster information if k > 1
    if selected_k > 1 and kmeans_models and selected_k in kmeans_models:
        clusters = kmeans_models[selected_k].predict(scaled_data)
        top_data['Cluster'] = clusters
        color_col = 'Cluster'
    else:
        color_col = None
    
    # Create scatterplot matrix
    fig = px.scatter_matrix(
        top_data,
        dimensions=top_features,
        color=color_col,
        labels={col: col for col in top_features},
        title="Scatterplot Matrix of Top Features"
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=900
    )
    
    # Update traces
    fig.update_traces(diagonal_visible=False)
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)