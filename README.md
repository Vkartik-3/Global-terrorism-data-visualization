# Global Terrorism Database Analysis

![GitHub last commit](https://img.shields.io/github/last-commit/Vkartik-3/gtd-visualization?style=flat-square)
![GitHub license](https://img.shields.io/github/license/Vkartik-3/gtd-visualization?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-latest-blue?style=flat-square&logo=plotly&logoColor=white)

An interactive data visualization dashboard for exploring patterns in global terrorism through Principal Component Analysis (PCA) and k-means clustering.

![Description of image](https://github.com/Vkartik-3/Global-terrorism-data-visualization/blob/main/images/pca%3Akmeans.jpeg)
![Description of image](https://github.com/Vkartik-3/Global-terrorism-data-visualization/blob/main/images/scatterplot.jpeg)
![Description of image](https://github.com/Vkartik-3/Global-terrorism-data-visualization/blob/main/images/fearures%20top.jpeg)
![Description of image](https://github.com/Vkartik-3/Global-terrorism-data-visualization/blob/main/images/biplot.jpeg)


## 📋 Table of Contents

- [Overview](#overview)
- [Data Source](#data-source)
- [Features](#features)
- [Key Insights](#key-insights)
- [Implementation Details](#implementation-details)
- [Installation and Usage](#installation-and-usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🔭 Overview

This project implements an interactive visualization dashboard for analyzing the Global Terrorism Database (GTD) through dimensionality reduction and clustering techniques. It allows users to explore patterns and relationships in terrorism data through Principal Component Analysis (PCA) and k-means clustering.

The visualization provides interactive tools to:
- Determine optimal dimensionality through PCA
- Identify the most influential features in the dataset
- Discover natural groupings using k-means clustering
- Explore relationships between key terrorism attributes

## 📊 Data Source

The project uses the Global Terrorism Database (GTD), a comprehensive open-source database of terrorist events worldwide:

- **Source**: [START Consortium](https://www.start.umd.edu/gtd/) (National Consortium for the Study of Terrorism and Responses to Terrorism)
- **Scope**: Global terrorist incidents from 1970-2020
- **Size**: 200,000+ incidents with 135 attributes
- **Focus Area**: For this analysis, we sampled 10,000 records from 2016-2020 and selected 20 key numerical features

## ✨ Features

### 1. PCA Visualization
- **Scree Plot**: Interactive bar chart of eigenvalues with cumulative variance overlay
- **Dimensionality Selection**: Click functionality to select optimal number of components
- **Biplot**: Visualization of data points on first two principal components with feature vector overlays
- **Reference Lines**: Clear x=0 and y=0 axes for better interpretation

### 2. Feature Importance Analysis
- **Top Features Algorithm**: Identifies attributes with highest PCA loadings based on selected dimensionality
- **Dynamic Feature Table**: Shows most influential features with importance scores
- **Scatterplot Matrix**: Displays relationships between top 4 features

### 3. Clustering Implementation
- **K-means Elbow Plot**: Visualizes inertia vs. k values to identify optimal cluster count
- **Interactive Cluster Selection**: Allows users to select k by clicking on the elbow plot
- **Consistent Cluster Coloring**: Maintains color scheme across all visualizations

### 4. Interactive Coordination
- All visualizations are linked through callbacks
- Selections in one view update all other components
- Provides a cohesive analytical environment

## 🔍 Key Insights

Analysis of the Global Terrorism Database revealed several interesting patterns:

1. **Dimensionality Reduction**: Approximately 11 principal components capture ~80% of the variance in terrorism data.

2. **Most Influential Features**:
   - Day of incident (`iday`): 0.87 importance score
   - Number of perpetrators captured (`nperpcap`): 0.83 importance score
   - Property damage indicator (`property`): 0.61 importance score
   - Event identifier (`eventid`): 0.52 importance score

3. **Optimal Clustering**: The elbow method identified 7 distinct clusters of terrorism incidents, representing different terrorism typologies.

4. **Cluster Characteristics**:
   - Distinct separation in temporal patterns (incident days) across clusters
   - Varying patterns in perpetrator captures, suggesting different operational success rates
   - Clear differentiation in property damage patterns
   - Notable outliers representing unusual terrorism events

## 🛠️ Implementation Details

### Data Processing Pipeline
1. Loading and filtering GTD dataset to recent years (2016-2020)
2. Selection of numerical attributes suitable for PCA
3. Handling missing values by column mean imputation
4. Standardization of features for equal weighting in PCA
5. Computation of PCA components, eigenvalues, and explained variance
6. Implementation of k-means clustering for k=1...10 with inertia calculation

### Visualization Enhancements
1. **Improved Biplot**:
   - Reference lines at x=0 and y=0 to delineate quadrants
   - Strategic feature vector display to reduce visual clutter
   - Increased visibility of feature labels
   - Balanced axis scale for proper representation

2. **Scatterplot Matrix Optimization**:
   - Dynamic updating based on selected dimensionality
   - Clear visual separation of clusters
   - Consistent color scheme with other visualizations

3. **Elbow Plot Interface**:
   - Clear indication of optimal k value
   - Interactive selection capability
   - Visual feedback on selection

## 💻 Installation and Usage

### Prerequisites
- Python 3.8+
- Plotly
- NumPy, Pandas, Scikit-learn
- Modern web browser (Chrome, Firefox, Safari)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Vkartik-3/gtd-visualization.git
cd gtd-visualization
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

### Usage Guide

1. **Exploring PCA**:
   - View the scree plot to understand variance distribution
   - Click on a bar to select number of components to retain
   - Examine the biplot to understand feature relationships

2. **Understanding Feature Importance**:
   - After selecting dimensionality, check the Feature Importance table
   - Explore the scatterplot matrix to see relationships between key features

3. **Investigating Clusters**:
   - View the elbow plot to determine optimal number of clusters
   - Click on a point to select k value
   - Observe how clusters are distributed across different visualizations

## 📁 Project Structure

```
Global-terrorism-data-visualization/
├── __pycache__/                # Python cache directory
├── assets/                     # Assets directory
│   └── styles.css              # CSS styling file
├── data/                       # Data directory
│   ├── .DS_Store               # macOS directory file
│   └── globalterrorismdb.xlsx  # Original GTD dataset
├── app.py                      # Main application file
├── data_processing.py          # Data preprocessing and analysis functions
└── README.md                   # Project documentation
```

## 🚀 Technologies Used

- **Data Processing**: Python, NumPy, Pandas
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly
- **Web Framework**: Flask
- **Development Tools**: Git, VS Code

## 🔮 Future Improvements

1. **Temporal Analysis**: Add time-based visualizations to examine trends over years
2. **Geographical Integration**: Incorporate maps to visualize spatial patterns
3. **Feature Engineering**: Explore additional derived features for enhanced insights
4. **Alternative Clustering**: Implement DBSCAN or hierarchical clustering for comparison
5. **Real-time Updates**: Create pipeline for incorporating new GTD data as it's released

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- START Consortium for providing the Global Terrorism Database
- CSE-564 Visualization course at Stony Brook University
- [Plotly](https://plotly.com/) for interactive visualization capabilities
- [Scikit-learn](https://scikit-learn.org/) for machine learning implementations

---

Developed by Kartik Kirankumar Vadhawana (116740869) as part of CSE-564 Visualization coursework at Stony Brook University.
