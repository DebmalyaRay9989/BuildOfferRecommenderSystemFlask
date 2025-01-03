


import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.express as px
import json
import plotly
import logging

# Initialize Flask App
app = Flask(__name__)

# Define feature columns globally
features = ['total_rech_amt', 'total_rech_data', 'vol_4g', 'vol_5g', 'arpu']

# Set up logging for better error tracking
logging.basicConfig(level=logging.DEBUG)

# Load Data (with error handling)
def load_data():
    try:
        file_path = './data/telecom_customers.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)

        # Validate essential columns
        required_columns = ['Customer ID', 'offer', 'Churn Value', 'total_rech_amt', 'total_rech_data', 'vol_4g', 'vol_5g', 'arpu']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Drop rows with missing essential data
        df = df.dropna(subset=['Customer ID', 'offer', 'Churn Value'])

        # Handle missing numerical values (using median for better robustness)
        df['total_rech_amt'].fillna(df['total_rech_amt'].median(), inplace=True)
        df['total_rech_data'].fillna(df['total_rech_data'].median(), inplace=True)

        return df, None  # Return dataframe and None for error message
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, str(e)  # Return None and the error message

# Normalize data for similarity computation
def normalize_data(df, selected_features):
    try:
        scaler = StandardScaler()
        df[selected_features] = scaler.fit_transform(df[selected_features])
        return df
    except Exception as e:
        logging.error(f"Error normalizing data: {e}")
        raise

# DBSCAN Clustering Function
def perform_dbscan_clustering(df, selected_features, eps, min_samples):
    # Normalize the selected features before clustering
    df = normalize_data(df.copy(), selected_features)
    
    # Perform DBSCAN clustering on the selected features
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(df[selected_features])
    
    return df

# Visualize Clusters using Plotly
def plot_dbscan_clusters(df, x_col='total_rech_amt', y_col='total_rech_data'):
    # Scatter plot for DBSCAN clusters
    fig = px.scatter(df, x=x_col, y=y_col, color='Cluster',
                     title="DBSCAN Clustering of Customers",
                     labels={'total_rech_amt': 'Total Recharge Amount', 'total_rech_data': 'Total Recharge Data'},
                     color_continuous_scale='Viridis')

    fig.update_layout(template="plotly_dark")
    return fig

@app.route('/dbscan', methods=['GET', 'POST'])
def dbscan():
    # Load data
    df, error_message = load_data()
    if df is None:
        return render_template('error.html', error_message=error_message)

    # Default values for DBSCAN parameters
    eps = 0.5
    min_samples = 5
    selected_features = features  # Default selected features if none are chosen

    # Handle form submission
    if request.method == 'POST':
        # Get the selected features and DBSCAN parameters
        selected_features = request.form.getlist('features')
        
        # Validate that features are selected
        if not selected_features:
            return render_template('dbscan.html', 
                                   error_message="Please select at least one feature for clustering.", 
                                   df=df,
                                   features=features, 
                                   selected_features=selected_features,
                                   eps=eps, 
                                   min_samples=min_samples)

        # Get DBSCAN parameters from the form
        eps = float(request.form['eps'])
        min_samples = int(request.form['min_samples'])

        # Perform DBSCAN clustering
        df = perform_dbscan_clustering(df, selected_features, eps, min_samples)
        
        # Visualize DBSCAN clustering
        fig = plot_dbscan_clusters(df)
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Pass the dataframe to the template as a list of dictionaries for easier rendering
        if df is not None and not df.empty:
            df_html = df.head(10).to_dict(orient='records')
        else:
            df_html = []

        return render_template('dbscan.html', 
                               fig_json=fig_json, 
                               df=df_html,
                               features=features, 
                               selected_features=selected_features,
                               eps=eps, 
                               min_samples=min_samples)

    # Render the form page with the default clustering view
    df_html = df.to_dict(orient='records')
    return render_template('dbscan.html', 
                           df=df_html, 
                           features=features, 
                           selected_features=selected_features,
                           eps=eps, 
                           min_samples=min_samples)


# Recommend offers based on similarity
def get_recommended_offers(df, customer_id, distance_func, n, selected_features):
    try:
        # Normalize data if needed
        df = normalize_data(df.copy(), selected_features)
        
        # Extract customer vector
        customer_row = df[df['Customer ID'] == customer_id]
        if customer_row.empty:
            raise ValueError("Customer ID not found in the dataset.")
        
        x = customer_row[selected_features].values[0]
        X = df[selected_features].values

        # Compute similarity based on the selected distance metric
        if distance_func == 'euclidean':
            distances = euclidean_distances(X, x.reshape(1, -1)).flatten()
        elif distance_func == 'manhattan':
            distances = manhattan_distances(X, x.reshape(1, -1)).flatten()
        elif distance_func == 'cosine':
            distances = 1 - cosine_similarity(X, x.reshape(1, -1)).flatten()
        else:
            raise ValueError('Invalid distance function specified.')

        # Find the n most similar customers
        most_similar_indices = distances.argsort()[:n]

        # Get offers of similar customers
        similar_customers = df.iloc[most_similar_indices]
        recommended_offers = similar_customers['offer'].value_counts().head(n).index.tolist()

        return recommended_offers, similar_customers, distances[most_similar_indices]
    except Exception as e:
        logging.error(f"Error recommending offers: {e}")
        return [], None, []

# Route for the home page
@app.route('/')
def index():
    # Load data
    df, error_message = load_data()
    if df is None:
        return render_template('error.html', error_message=error_message)
    
    # Extract customer IDs for the dropdown menu
    customer_ids = df['Customer ID'].unique()
    
    return render_template('index.html', customer_ids=customer_ids)

# Route for recommending offers
@app.route('/recommend', methods=['POST'])
def recommend():
    # Load data
    df, error_message = load_data()
    if df is None:
        return render_template('error.html', error_message=error_message)

    # Get form data
    customer_id = request.form['customer_id']
    distance_func = request.form['distance_func']
    n = int(request.form['n_customers'])
    selected_features = request.form.getlist('features')

    # Validate input
    if not selected_features:
        return render_template('index.html', customer_ids=df['Customer ID'].unique(), error_message="Please select at least one feature for similarity.")

    # Get recommended offers
    recommended_offers, similar_customers, distances = get_recommended_offers(
        df, customer_id, distance_func, n, selected_features)

    # If no similar customers found
    if not recommended_offers:
        return render_template('error.html', error_message="No similar customers found based on the selected parameters.")
    
    # Convert results to JSON for Plotly visualization
    offer_counts = similar_customers['offer'].value_counts()
    offer_counts_json = offer_counts.to_json()

    # Offer counts plot using Plotly
    fig = px.bar(
        offer_counts,
        x=offer_counts.index,
        y=offer_counts.values,
        title="Offer Counts Recommended to Similar Customers",
        labels={'x': 'Offer', 'y': 'Count'},
        color=offer_counts.values,
        color_continuous_scale='Blues'
    )

    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('recommend.html', 
                           recommended_offers=recommended_offers, 
                           similar_customers=similar_customers.to_html(classes="table table-striped"), 
                           offer_counts_json=offer_counts_json, 
                           offer_counts_plot=fig_json)  # Pass offer counts plot instead of similarity plot

if __name__ == '__main__':
    app.run(debug=True)









