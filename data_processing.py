import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def prepare_gtd_data(df):
    """
    Prepare GTD data for analysis by selecting and processing relevant columns
    """
    # Check for common numerical columns in GTD
    numerical_cols = []
    
    # Try to find numerical columns
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)
    
    # Ensure we have enough columns (at least 10)
    if len(numerical_cols) < 10:
        print("Warning: Few numerical columns found. Results may not be optimal.")
    else:
        print(f"Found {len(numerical_cols)} numerical columns.")
    
    # Limit to first 20 numerical columns to prevent performance issues
    if len(numerical_cols) > 20:
        numerical_cols = numerical_cols[:20]
        print(f"Limited to 20 numerical columns for performance.")
    
    # Handle missing values - replace with column means
    df_clean = df[numerical_cols].copy()
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    return df_clean, numerical_cols

def perform_pca(df):
    """
    Perform PCA on the dataframe with proper data cleaning
    """
    # Check for any remaining NaN values
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Found {df.isnull().sum().sum()} NaN values before PCA")
        # Fill any remaining NaNs with column means
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
    
    # Scale the data
    scaler = StandardScaler()
    try:
        scaled_data = scaler.fit_transform(df)
    except Exception as e:
        print(f"Error in scaling: {e}")
        # Last resort - replace any problematic values
        df_fixed = df.copy()
        for col in df.columns:
            df_fixed[col] = pd.to_numeric(df[col], errors='coerce')
            df_fixed[col] = df_fixed[col].fillna(df_fixed[col].mean())
        scaled_data = scaler.fit_transform(df_fixed)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Get the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Get the loadings
    loadings = pca.components_
    
    return pca, pca_result, explained_variance, loadings, scaled_data

def find_elbow_point(values):
    """
    Find the elbow point in a curve (for scree plot or kmeans)
    """
    if len(values) <= 2:
        return 1
    
    # Calculate differences (first derivative)
    y_diff = np.diff(values)
    
    # For scree plot (decreasing values)
    if values[0] > values[-1]:
        # Find where the decrease starts to level off
        idx = 0
        for i in range(len(y_diff) - 1):
            if abs(y_diff[i]) > abs(y_diff[i + 1]) * 3:  # significant change in slope
                idx = i + 1
                break
        
        # Default to component where we reach ~70% explained variance for PCA
        if "elbow_idx" not in locals():
            cum_var = np.cumsum(values)
            idx = np.argmax(cum_var >= 0.7 * cum_var[-1]) + 1
    else:
        # For kmeans (increasing values), using second derivative
        y_diff2 = np.diff(y_diff)
        idx = np.argmax(np.abs(y_diff2)) + 2
    
    # Ensure the index is within bounds
    return min(max(1, idx), len(values) - 1)

def get_top_features(loadings, feature_names, n_components, n_top_features=4):
    """
    Get the top features based on squared sum of PCA loadings
    """
    # Ensure n_components is within bounds
    n_components = min(n_components, loadings.shape[0])
    
    # Calculate squared sum of loadings for each feature
    squared_loadings = loadings[:n_components, :]**2
    sum_squared_loadings = np.sum(squared_loadings, axis=0)
    
    # Get the indices of top features
    top_indices = np.argsort(sum_squared_loadings)[::-1][:n_top_features]
    
    # Get the feature names and their scores
    top_features = [feature_names[i] for i in top_indices]
    top_features_scores = sum_squared_loadings[top_indices]
    
    return top_features, top_features_scores, top_indices

def perform_kmeans(data, k_range=range(1, 11)):
    """
    Perform K-means clustering for a range of k values
    """
    inertia = []
    models = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        models[k] = kmeans
    
    return inertia, models