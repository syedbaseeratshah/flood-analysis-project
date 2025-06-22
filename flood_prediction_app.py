import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Flood Prediction Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .warning {
        color: #FF5722;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E0E0E0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 5px solid #9E9E9E;
    }
    .prediction-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F5F5F5;
        margin-bottom: 1rem;
        border: 1px solid #E0E0E0;
    }
    .flood-alert {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and other persistent data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
    
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
    
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'flood_threshold' not in st.session_state:
    st.session_state.flood_threshold = None
    
if 'sequence_length' not in st.session_state:
    st.session_state.sequence_length = 7  # Default sequence length for model input


# Function to load and preprocess data
def load_and_process_data(file):
    """Load and perform initial processing on dataset"""
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check for required columns
        required_columns = [
            'Time', 'Discharge(m^3 s^-1)', 'Precip(mm h^-1)', 'PET(mm h^-1)',
            'SM(%)', 'Groundwater (mm)', 'Fast Flow(mm*1000)',
            'Slow Flow(mm*1000)', 'Base Flow(mm*1000)'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert 'Time' to datetime
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Convert string 'nan' to np.nan
        df = df.replace('nan', np.nan)
        
        # Convert all numeric columns to float
        for col in df.columns:
            if col != 'Time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Sort by time
        df = df.sort_values('Time')
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        return df, processed_df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None


# Function to preprocess data for model input (similar to notebook)
def preprocess_data(df):
    """Comprehensive preprocessing and feature engineering"""
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Set Time as index for easier time-based operations
    data.set_index('Time', inplace=True)
    
    # Handle missing values
    for col in data.columns:
        if col != 'Observed(m^3 s^-1)':  # Leave observed discharge as is
            # Interpolate missing values with a limit on consecutive values
            data[col] = data[col].interpolate(method='time', limit=3)
    
    # Forward fill any remaining gaps
    data.fillna(method='ffill', inplace=True)
    
    # Create date-based features
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfYear'] = data.index.dayofyear
    
    # Cyclical encoding of month and day of year to capture seasonality
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 366)
    data['Day_cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 366)
    
    # Create lag features
    # Precipitation lags
    for lag in [1, 2, 3, 7]:
        data[f'Precip_lag_{lag}'] = data['Precip(mm h^-1)'].shift(lag)
    
    # Discharge lags (autoregressive component)
    for lag in [1, 2, 3]:
        data[f'Discharge_lag_{lag}'] = data['Discharge(m^3 s^-1)'].shift(lag)
    
    # Create rolling statistics
    data['Precip_sum_7d'] = data['Precip(mm h^-1)'].rolling(window=7).sum()
    data['Discharge_mean_7d'] = data['Discharge(m^3 s^-1)'].rolling(window=7).mean()
    
    # Fill NaNs after feature creation
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate flood threshold (95th percentile)
    flood_threshold = data['Discharge(m^3 s^-1)'].quantile(0.95)
    st.session_state.flood_threshold = flood_threshold
    
    # Create binary flood labels
    data['Flood_Peak'] = (data['Discharge(m^3 s^-1)'] > flood_threshold).astype(int)
    
    return data


# Function to load models
def load_models():
    """Load the saved TensorFlow models"""
    try:
        # Check if models exist in the expected paths
        model_paths = {
            'classification': 'g1_classification_model.h5',
            'regression': 'g1_regression_model.h5'
        }
        
        models = {}
        
        for model_type, path in model_paths.items():
            if os.path.exists(path):
                models[model_type] = load_model(path)
                st.sidebar.success(f"✅ {model_type.capitalize()} model loaded successfully")
            else:
                # For demo purposes, create dummy models if files don't exist
                st.sidebar.warning(f"⚠️ {model_type.capitalize()} model file not found. Using a dummy model.")
                
                # Create dummy model for demo purposes
                input_shape = (st.session_state.sequence_length, 11)  # Sequence length and features
                
                # Simple LSTM model with similar architecture
                dummy_model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
                    tf.keras.layers.LSTM(32, return_sequences=False),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid' if model_type == 'classification' else None)
                ])
                
                # Compile the model
                if model_type == 'classification':
                    dummy_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    dummy_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                models[model_type] = dummy_model
        
        st.session_state.models_loaded = True
        return models
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


# Function to create sequence data for prediction
def create_sequence_data(df, sequence_length=7):
    """Create sequence data for model input"""
    # Select feature columns (similar to your notebook)
    important_features = [
        'Discharge(m^3 s^-1)', 'Precip(mm h^-1)', 'PET(mm h^-1)', 'SM(%)',
        'Groundwater (mm)', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
        'Precip_lag_1', 'Discharge_lag_1'
    ]
    
    # Ensure these features exist in the dataframe
    features = [col for col in important_features if col in df.columns]
    
    # Get the most recent data points
    recent_data = df[features].tail(sequence_length).values
    
    # Reshape for model input (batch_size, sequence_length, n_features)
    X = np.array([recent_data])
    
    return X


# Function to make predictions
def make_predictions(X, models):
    """Make flood and discharge predictions using the loaded models"""
    if not models:
        st.error("Models not loaded properly")
        return None
    
    try:
        # Get discharge prediction
        discharge_pred = models['regression'].predict(X)[0][0]
        
        # Get flood probability
        flood_prob = models['classification'].predict(X)[0][0]
        
        # Check if prediction exceeds threshold
        exceeds_threshold = discharge_pred > st.session_state.flood_threshold
        
        return {
            'discharge': discharge_pred,
            'flood_probability': flood_prob,
            'exceeds_threshold': exceeds_threshold
        }
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None


# Function to generate future predictions
def predict_future_days(df, models, days=7):
    """Generate predictions for the next n days"""
    if not models:
        st.error("Models not loaded")
        return None
    
    # Get the last date in the dataset
    last_date = df.index[-1]
    
    # Create a copy of the original data for predictions
    prediction_df = df.copy()
    
    # Storage for predictions
    predictions = []
    
    for i in range(1, days + 1):
        # Create sequence data from the most recent window
        X = create_sequence_data(prediction_df)
        
        # Make prediction
        pred = make_predictions(X, models)
        
        if not pred:
            continue
        
        # Calculate next date
        next_date = last_date + pd.Timedelta(days=i)
        
        # Create a new row with the prediction results
        new_row = pd.Series({
            'Discharge(m^3 s^-1)': pred['discharge'],
            'Flood_Peak': 1 if pred['exceeds_threshold'] else 0
        }, name=next_date)
        
        # Add date features
        new_row['Year'] = next_date.year
        new_row['Month'] = next_date.month
        new_row['Day'] = next_date.day
        new_row['DayOfYear'] = next_date.dayofyear
        new_row['Month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        new_row['Month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        new_row['Day_sin'] = np.sin(2 * np.pi * next_date.dayofyear / 366)
        new_row['Day_cos'] = np.cos(2 * np.pi * next_date.dayofyear / 366)
        
        # Append the new row to the prediction dataframe
        prediction_df = pd.concat([prediction_df, pd.DataFrame([new_row])])
        
        # Update lag features for the new row
        for lag in [1, 2, 3, 7]:
            if f'Precip_lag_{lag}' in prediction_df.columns:
                try:
                    prediction_df.loc[next_date, f'Precip_lag_{lag}'] = prediction_df['Precip(mm h^-1)'].shift(lag).loc[next_date]
                except:
                    # If lag is larger than available data, use the last known value
                    prediction_df.loc[next_date, f'Precip_lag_{lag}'] = 0
        
        for lag in [1, 2, 3]:
            if f'Discharge_lag_{lag}' in prediction_df.columns:
                try:
                    prediction_df.loc[next_date, f'Discharge_lag_{lag}'] = prediction_df['Discharge(m^3 s^-1)'].shift(lag).loc[next_date]
                except:
                    # If lag is larger than available data, use the last known value
                    prediction_df.loc[next_date, f'Discharge_lag_{lag}'] = prediction_df['Discharge(m^3 s^-1)'].iloc[-2]
        
        # Append prediction results
        predictions.append({
            'date': next_date,
            'discharge': pred['discharge'],
            'flood_probability': pred['flood_probability'],
            'exceeds_threshold': pred['exceeds_threshold']
        })
    
    return predictions


# Function to analyze flood events
def analyze_flood_events(df):
    """Identify and analyze historical flood events"""
    # Get the flood threshold
    flood_threshold = st.session_state.flood_threshold
    
    # Create a copy of the dataframe with reset index
    flood_df = df.reset_index()
    
    # Identify flood events (periods where discharge exceeds threshold)
    flood_df['is_flood'] = flood_df['Discharge(m^3 s^-1)'] > flood_threshold
    
    # Group consecutive flood days
    flood_df['flood_group'] = (flood_df['is_flood'] != flood_df['is_flood'].shift()).cumsum()
    
    # Filter for flood periods
    flood_events = flood_df[flood_df['is_flood']].groupby('flood_group')
    
    # Analyze each flood event
    event_list = []
    
    for group, event_data in flood_events:
        event = {
            'start_date': event_data['Time'].min(),
            'end_date': event_data['Time'].max(),
            'duration': len(event_data),
            'peak_discharge': event_data['Discharge(m^3 s^-1)'].max(),
            'peak_date': event_data.loc[event_data['Discharge(m^3 s^-1)'].idxmax(), 'Time'],
            'total_precip': event_data['Precip(mm h^-1)'].sum(),
            'avg_soil_moisture': event_data['SM(%)'].mean()
        }
        event_list.append(event)
    
    return event_list, flood_threshold


# Function to analyze hysteresis
def analyze_hysteresis(df):
    """Analyze hydrological hysteresis in flood events"""
    # Get flood events
    events, flood_threshold = analyze_flood_events(df)
    
    # Sort events by peak discharge
    sorted_events = sorted(events, key=lambda x: x['peak_discharge'], reverse=True)
    
    # Take the top 3 events
    top_events = sorted_events[:3]
    
    # Get data for each event with surrounding days
    event_data = []
    
    for event in top_events:
        # Get 3 days before and after the event
        start_date = event['start_date'] - pd.Timedelta(days=3)
        end_date = event['end_date'] + pd.Timedelta(days=3)
        
        # Extract data for this time period
        event_df = df.reset_index()
        event_df = event_df[(event_df['Time'] >= start_date) & (event_df['Time'] <= end_date)]
        
        # Add time progression index
        event_df['time_index'] = range(len(event_df))
        
        event_data.append({
            'event_info': event,
            'data': event_df
        })
    
    return event_data


# Function to analyze correlations
def analyze_correlations(df):
    """Calculate correlations between key variables"""
    # Select key variables
    key_vars = [
        'Discharge(m^3 s^-1)', 'Precip(mm h^-1)', 'PET(mm h^-1)', 'SM(%)',
        'Groundwater (mm)', 'Fast Flow(mm*1000)', 'Slow Flow(mm*1000)',
        'Base Flow(mm*1000)'
    ]
    
    # Add lag variables if they exist
    for lag in [1, 7]:
        lag_var = f'Precip_lag_{lag}'
        if lag_var in df.columns:
            key_vars.append(lag_var)
    
    # Calculate correlation matrix
    correlation = df[key_vars].corr()
    
    # Get correlations with discharge
    discharge_corrs = correlation['Discharge(m^3 s^-1)'].sort_values(ascending=False)
    
    return discharge_corrs, correlation


# Function to analyze basin memory
def analyze_basin_memory(df):
    """Analyze basin memory by examining lag correlation patterns"""
    # Calculate correlation between discharge and lagged precipitation
    lag_corrs = []
    max_lag = 30  # Maximum lag to consider
    
    # Calculate correlations for different lags
    for lag in range(1, max_lag + 1):
        lag_var = f'Precip_lag_{lag}'
        
        if lag_var in df.columns:
            # Use existing lag column
            corr = df['Discharge(m^3 s^-1)'].corr(df[lag_var])
        else:
            # Calculate correlation manually
            corr = df['Discharge(m^3 s^-1)'].corr(df['Precip(mm h^-1)'].shift(lag))
        
        lag_corrs.append((lag, corr))
    
    # Find the lag with the highest correlation
    lags, corrs = zip(*lag_corrs)
    max_corr_lag = lags[np.argmax(corrs)]
    
    # Calculate the "memory half-life" - lag at which correlation drops to half of max
    max_corr = max(corrs)
    half_corr = max_corr / 2
    
    # Find the lag where correlation drops below half of max
    memory_half_life = None
    for i, corr in enumerate(corrs):
        if i > np.argmax(corrs) and corr < half_corr:
            memory_half_life = lags[i]
            break
    
    return {
        'lag_correlations': lag_corrs,
        'max_correlation_lag': max_corr_lag,
        'memory_half_life': memory_half_life
    }


# Function to create time series plot
def plot_time_series(df):
    """Create a time series plot of key variables"""
    # Create plotly figure with multiple y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add discharge line
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Discharge(m^3 s^-1)'],
            name="Discharge",
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add precipitation line
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Precip(mm h^-1)'],
            name="Precipitation",
            line=dict(color='green')
        ),
        secondary_y=True
    )
    
    # Add soil moisture
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['SM(%)'],
            name="Soil Moisture",
            line=dict(color='brown', dash='dash')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Time Series of Hydro-Meteorological Variables",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Discharge (m³/s)", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation (mm/h) / Soil Moisture (%)", secondary_y=True)
    
    return fig


# Function to plot flood events
def plot_flood_events(df, events, threshold):
    """Plot a time series with flood events highlighted"""
    # Create a copy of the dataframe with reset index
    flood_df = df.reset_index()
    
    # Add a column indicating if each point is during a flood
    flood_df['is_flood'] = flood_df['Discharge(m^3 s^-1)'] > threshold
    
    # Create the plot
    fig = go.Figure()
    
    # Add discharge line
    fig.add_trace(
        go.Scatter(
            x=flood_df['Time'],
            y=flood_df['Discharge(m^3 s^-1)'],
            name='Discharge',
            line=dict(color='blue')
        )
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=flood_df['Time'],
            y=[threshold] * len(flood_df),
            name=f'Flood Threshold ({threshold:.2f} m³/s)',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add precipitation as a bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=flood_df['Time'],
            y=flood_df['Precip(mm h^-1)'],
            name='Precipitation',
            marker=dict(color='lightgreen', opacity=0.5),
            yaxis='y2'
        )
    )
    
    # Highlight flood periods
    for event in events:
        fig.add_vrect(
            x0=event['start_date'],
            x1=event['end_date'],
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0
        )
    
    # Update layout
    fig.update_layout(
        title="Discharge Time Series with Flood Events",
        xaxis_title="Date",
        yaxis=dict(
            title="Discharge (m³/s)",
            side="left"
        ),
        yaxis2=dict(
            title="Precipitation (mm/h)",
            side="right",
            overlaying="y",
            range=[0, max(flood_df['Precip(mm h^-1)']) * 3]  # Scale for better visibility
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig


# Function to plot hysteresis
def plot_hysteresis(event_data):
    """Plot hysteresis loops for flood events"""
    if not event_data:
        return None
    
    # Create a figure with subplots for each event
    fig = make_subplots(
        rows=len(event_data), 
        cols=2,
        subplot_titles=[f"Event {i+1}: Hysteresis Loop" for i in range(len(event_data))] + 
                       [f"Event {i+1}: Time Series" for i in range(len(event_data))]
    )
    
    for i, event in enumerate(event_data):
        event_df = event['data']
        info = event['event_info']
        
        # Add scatter plot for hysteresis (precipitation vs discharge)
        fig.add_trace(
            go.Scatter(
                x=event_df['Precip(mm h^-1)'],
                y=event_df['Discharge(m^3 s^-1)'],
                mode='markers+lines',
                marker=dict(
                    color=event_df['time_index'],
                    colorscale='Viridis',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title="Time Progression") if i == 0 else None
                ),
                name=f"Event {i+1}"
            ),
            row=i+1, col=1
        )
        
        # Add time series for the event
        fig.add_trace(
            go.Scatter(
                x=event_df['Time'],
                y=event_df['Discharge(m^3 s^-1)'],
                mode='lines',
                name='Discharge',
                line=dict(color='blue')
            ),
            row=i+1, col=2
        )
        
        # Add precipitation on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=event_df['Time'],
                y=event_df['Precip(mm h^-1)'],
                name='Precipitation',
                marker=dict(color='green', opacity=0.5)
            ),
            row=i+1, col=2
        )
        
        # Highlight the flood period
        flood_period = event_df[event_df['is_flood'] == True]
        if not flood_period.empty:
            fig.add_vrect(
                x0=flood_period['Time'].min(),
                x1=flood_period['Time'].max(),
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=i+1, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=300 * len(event_data),
        title="Hysteresis Analysis for Major Flood Events",
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, len(event_data) + 1):
        fig.update_xaxes(title_text="Precipitation (mm/h)", row=i, col=1)
        fig.update_yaxes(title_text="Discharge (m³/s)", row=i, col=1)
        fig.update_xaxes(title_text="Date", row=i, col=2)
        fig.update_yaxes(title_text="Discharge (m³/s) / Precipitation (mm/h)", row=i, col=2)
    
    return fig


# Function to plot correlation analysis
def plot_correlations(discharge_corrs):
    """Plot correlation of variables with discharge"""
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add bars for each correlation
    corr_df = discharge_corrs.reset_index()
    corr_df.columns = ['Variable', 'Correlation']
    
    # Exclude self-correlation (discharge with discharge)
    corr_df = corr_df[corr_df['Variable'] != 'Discharge(m^3 s^-1)']
    
    # Clean variable names for display
    corr_df['Variable'] = corr_df['Variable'].apply(lambda x: x.split('(')[0].strip())
    
    # Sort by absolute correlation value
    corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
    
    # Create color scale based on correlation value
    colors = ['red' if c < 0 else 'blue' for c in corr_df['Correlation']]
    
    # Add bars
    fig.add_trace(
        go.Bar(
            y=corr_df['Variable'],
            x=corr_df['Correlation'],
            orientation='h',
            marker_color=colors
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Correlation with Discharge",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Variable",
        xaxis=dict(range=[-1, 1])
    )
    
    # Add a vertical line at x=0
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(corr_df) - 0.5,
        line=dict(color="black", width=1, dash="dash")
    )
    
    return fig


# Function to plot basin memory analysis
def plot_basin_memory(memory_data):
    """Plot basin memory analysis"""
    # Extract lag correlations
    lag_corrs = memory_data['lag_correlations']
    lags, corrs = zip(*lag_corrs)
    
    # Create the plot
    fig = go.Figure()
    
    # Add scatter plot with line
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=corrs,
            mode='markers+lines',
            marker=dict(size=8),
            line=dict(width=2),
            name='Correlation'
        )
    )
    
    # Add vertical line at max correlation lag
    max_corr_lag = memory_data['max_correlation_lag']
    fig.add_vline(
        x=max_corr_lag,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Correlation Lag: {max_corr_lag} days",
        annotation_position="top right"
    )
    
    # Add vertical line at memory half-life if it exists
    memory_half_life = memory_data['memory_half_life']
    if memory_half_life:
        fig.add_vline(
            x=memory_half_life,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Memory Half-life: {memory_half_life} days",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title="Basin Memory Analysis: Correlation of Lagged Precipitation with Discharge",
        xaxis_title="Lag (days)",
        yaxis_title="Correlation Coefficient",
        xaxis=dict(range=[0, max(lags) + 1])
    )
    
    return fig


# Function to plot prediction results
def plot_predictions(predictions, threshold):
    """Plot discharge predictions with flood probability"""
    # Create a dataframe from predictions
    pred_df = pd.DataFrame(predictions)
    
    # Create the plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add discharge prediction line
    fig.add_trace(
        go.Scatter(
            x=pred_df['date'],
            y=pred_df['discharge'],
            name='Predicted Discharge',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add flood probability line
    fig.add_trace(
        go.Scatter(
            x=pred_df['date'],
            y=pred_df['flood_probability'] * 100,  # Convert to percentage
            name='Flood Probability (%)',
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=pred_df['date'],
            y=[threshold] * len(pred_df),
            name=f'Flood Threshold ({threshold:.2f} m³/s)',
            line=dict(color='red', dash='dash')
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title="Discharge Predictions and Flood Probability",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Discharge (m³/s)", secondary_y=False)
    fig.update_yaxes(title_text="Flood Probability (%)", secondary_y=True, range=[0, 100])
    
    return fig


# Function to process user query
def process_query(query):
    """Process natural language query and respond with appropriate analysis"""
    # Convert query to lowercase for easier matching
    query_lower = query.lower()
    
    # Get the current dataset from session state
    df = st.session_state.processed_data
    
    if df is None:
        return "Please upload a dataset first before asking questions."
    
    # Check what kind of analysis is requested
    if any(term in query_lower for term in ['time series', 'show data', 'display data']):
        return {
            'type': 'time_series',
            'message': "Here's the time series visualization of your hydro-meteorological data."
        }
    
    elif any(term in query_lower for term in ['predict', 'forecast', 'next', 'future']):
        # Determine how many days to predict
        days = 7  # Default
        for num in ["1", "2", "3", "4", "5", "6", "7", "10", "14", "30"]:
            if num in query_lower.split() and "day" in query_lower:
                days = int(num)
                break
        
        return {
            'type': 'predictions',
            'days': days,
            'message': f"Here are the discharge predictions for the next {days} days."
        }
    
    elif any(term in query_lower for term in ['flood', 'event']):
        return {
            'type': 'flood_events',
            'message': "I've analyzed the flood events in your dataset."
        }
    
    elif any(term in query_lower for term in ['correlation', 'relationship']):
        return {
            'type': 'correlations',
            'message': "Here's the correlation analysis between discharge and other variables."
        }
    
    elif any(term in query_lower for term in ['hysteresis', 'loop']):
        return {
            'type': 'hysteresis',
            'message': "I've analyzed the hysteresis patterns in major flood events."
        }
    
    elif any(term in query_lower for term in ['basin memory', 'memory', 'lag']):
        return {
            'type': 'basin_memory',
            'message': "Here's the basin memory analysis showing how long precipitation events affect discharge."
        }
    
    elif any(term in query_lower for term in ['model', 'technique']):
        return {
            'type': 'model_info',
            'message': """
            This flood prediction system uses an LSTM (Long Short-Term Memory) neural network with attention mechanisms for both regression (predicting discharge values) and classification (predicting flood events).
            
            Key features of the model:
            - Uses a sequence of 7 days of hydro-meteorological data as input
            - Contains feature engineering such as lag variables and cyclical encoding of time
            - Achieves strong performance in identifying flood events above the 95th percentile threshold
            - Provides both discharge values and flood probability estimates
            """
        }
    
    else:
        # General information about the dataset
        return {
            'type': 'dataset_info',
            'message': f"""
            Here's a summary of your dataset:
            
            - Time range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
            - Number of records: {len(df)}
            - Average discharge: {df['Discharge(m^3 s^-1)'].mean():.2f} m³/s
            - Maximum discharge: {df['Discharge(m^3 s^-1)'].max():.2f} m³/s
            - Flood threshold (95th percentile): {st.session_state.flood_threshold:.2f} m³/s
            
            You can ask me to:
            - Show time series data
            - Predict discharge for future days
            - Identify flood events in the dataset
            - Analyze correlations between variables
            - Examine hysteresis patterns
            - Analyze basin memory effects
            """
        }


# Main app layout
def main():
    # Display app header
    st.markdown("<h1 class='main-header'>🌊 Flood Prediction Assistant</h1>", unsafe_allow_html=True)
    
    # Sidebar for dataset upload and model loading
    with st.sidebar:
        st.markdown("<h2 class='subheader'>Data & Models</h2>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Hydro-Meteorological CSV", type=['csv'])
        
        if uploaded_file is not None:
            # Load and process the data
            with st.spinner("Processing dataset..."):
                df, processed_df = load_and_process_data(uploaded_file)
                
                if df is not None and processed_df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.processed_data = processed_df
                    st.success(f"✅ Dataset loaded: {uploaded_file.name}")
                    
                    # Display dataset summary
                    st.markdown("### Dataset Summary")
                    st.write(f"Records: {len(df)}")
                    st.write(f"Time range: {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}")
                    st.write(f"Mean discharge: {df['Discharge(m^3 s^-1)'].mean():.2f} m³/s")
                    st.write(f"Max discharge: {df['Discharge(m^3 s^-1)'].max():.2f} m³/s")
                    
                    # Load models
                    st.markdown("### Models")
                    if st.button("Load Prediction Models"):
                        with st.spinner("Loading models..."):
                            st.session_state.models = load_models()
        
        # Information about the app
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses LSTM models with attention mechanisms to analyze hydro-meteorological data, predict discharge values, and identify potential flood events.
        
        Upload your CSV file in the format of the training data to get started.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.markdown("<h2 class='subheader'>Ask about your data</h2>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f"<div class='chat-message user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message bot-message'>{message['content']}</div>", unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Type your question here", key="user_query")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input
            })
            
            # Process the query
            with st.spinner("Analyzing..."):
                response = process_query(user_input)
                
                # Handle different response types
                if isinstance(response, dict):
                    # Add bot message to chat history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response['message']
                    })
                    
                    # Perform requested analysis
                    if response['type'] == 'time_series' and st.session_state.processed_data is not None:
                        fig = plot_time_series(st.session_state.processed_data)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif response['type'] == 'predictions' and st.session_state.processed_data is not None:
                        if not st.session_state.models_loaded:
                            st.warning("Please load the prediction models first.")
                        else:
                            days = response.get('days', 7)
                            predictions = predict_future_days(
                                st.session_state.processed_data, 
                                st.session_state.models, 
                                days=days
                            )
                            
                            if predictions:
                                # Display prediction results
                                fig = plot_predictions(predictions, st.session_state.flood_threshold)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show detailed prediction table
                                st.markdown("<h3>Detailed Predictions</h3>", unsafe_allow_html=True)
                                for pred in predictions:
                                    with st.container():
                                        cols = st.columns([1, 1, 1])
                                        with cols[0]:
                                            st.markdown(f"**{pred['date'].strftime('%Y-%m-%d')}**")
                                        with cols[1]:
                                            st.markdown(f"Discharge: **{pred['discharge']:.2f} m³/s**")
                                        with cols[2]:
                                            st.markdown(f"Flood Probability: **{pred['flood_probability']*100:.1f}%**")
                                        
                                        # Add alert if threshold exceeded
                                        if pred['exceeds_threshold']:
                                            st.markdown("<div class='flood-alert'>⚠️ POTENTIAL FLOOD ALERT: Discharge may exceed threshold</div>", unsafe_allow_html=True)
                                        
                                        st.markdown("---")
                    
                    elif response['type'] == 'flood_events' and st.session_state.processed_data is not None:
                        events, threshold = analyze_flood_events(st.session_state.processed_data)
                        
                        # Plot flood events
                        fig = plot_flood_events(st.session_state.processed_data, events, threshold)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display event details
                        st.markdown("<h3>Flood Event Details</h3>", unsafe_allow_html=True)
                        
                        if not events:
                            st.write("No flood events detected in this dataset.")
                        else:
                            for i, event in enumerate(sorted(events, key=lambda x: x['peak_discharge'], reverse=True)[:5]):
                                with st.expander(f"Event {i+1}: {event['start_date'].strftime('%Y-%m-%d')} to {event['end_date'].strftime('%Y-%m-%d')}"):
                                    st.write(f"Duration: {event['duration']} days")
                                    st.write(f"Peak Discharge: {event['peak_discharge']:.2f} m³/s on {event['peak_date'].strftime('%Y-%m-%d')}")
                                    st.write(f"Total Precipitation: {event['total_precip']:.1f} mm")
                                    st.write(f"Average Soil Moisture: {event['avg_soil_moisture']:.1f}%")
                    
                    elif response['type'] == 'correlations' and st.session_state.processed_data is not None:
                        discharge_corrs, correlation_matrix = analyze_correlations(st.session_state.processed_data)
                        
                        # Plot correlations
                        fig = plot_correlations(discharge_corrs)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display correlation matrix as a heatmap
                        st.markdown("<h3>Correlation Matrix</h3>", unsafe_allow_html=True)
                        fig = px.imshow(
                            correlation_matrix, 
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            origin='lower'
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif response['type'] == 'hysteresis' and st.session_state.processed_data is not None:
                        event_data = analyze_hysteresis(st.session_state.processed_data)
                        
                        if event_data:
                            # Plot hysteresis loops
                            fig = plot_hysteresis(event_data)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            ### Understanding Hysteresis Patterns
                            
                            Hysteresis loops show the relationship between precipitation and discharge during flood events.
                            
                            - **Clockwise loops** indicate that runoff responds quickly to precipitation, typical in watersheds with efficient drainage networks.
                            - **Counter-clockwise loops** suggest that soil moisture or groundwater processes delay the discharge response.
                            - The **width of the loop** indicates the strength of the hysteresis effect.
                            """)
                        else:
                            st.write("No significant flood events found for hysteresis analysis.")
                    
                    elif response['type'] == 'basin_memory' and st.session_state.processed_data is not None:
                        memory_data = analyze_basin_memory(st.session_state.processed_data)
                        
                        # Plot basin memory analysis
                        fig = plot_basin_memory(memory_data)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"""
                        ### Basin Memory Analysis Results
                        
                        - **Maximum correlation** occurs at a lag of **{memory_data['max_correlation_lag']} days**
                        - **Memory half-life**: {memory_data['memory_half_life'] if memory_data['memory_half_life'] else 'Not detected'} days
                        
                        This suggests that precipitation events continue to influence discharge for approximately {memory_data['max_correlation_lag'] * 2} days, with the strongest effect after {memory_data['max_correlation_lag']} days.
                        """)
                    
                    # Force a rerun to update the UI with the new chat message
                    st.rerun()
                else:
                    # Add bot message to chat history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response
                    })
                    
                    # Force a rerun to update the UI with the new chat message
                    st.rerun()
    
    with col2:
        # Examples and guidance
        st.markdown("<h2 class='subheader'>Example Questions</h2>", unsafe_allow_html=True)
        
        example_questions = [
            "Show me the time series data",
            "Predict discharge for the next 7 days",
            "What flood events are in this dataset?",
            "Show me the correlation between variables",
            "Analyze hysteresis patterns",
            "What is the basin memory effect?",
            "Tell me about your prediction model"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question}"):
                # Add the question to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': question
                })
                
                # Process the query
                with st.spinner("Analyzing..."):
                    response = process_query(question)
                    
                    # Add bot response to chat history
                    if isinstance(response, dict):
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': response['message']
                        })
                    else:
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': response
                        })
                
                # Force a rerun to update the UI
                st.rerun()
        
        # Guidance on data format
        st.markdown("<h2 class='subheader'>Data Requirements</h2>", unsafe_allow_html=True)
        st.markdown("""
        Your CSV should have these columns:
        - `Time`: Date/time
        - `Discharge(m^3 s^-1)`: River discharge
        - `Precip(mm h^-1)`: Precipitation
        - `PET(mm h^-1)`: Potential evapotranspiration
        - `SM(%)`: Soil moisture
        - `Groundwater (mm)`: Groundwater level
        - `Fast Flow(mm*1000)`: Fast flow component
        - `Slow Flow(mm*1000)`: Slow flow component
        - `Base Flow(mm*1000)`: Base flow component
        """)


if __name__ == "__main__":
    main()