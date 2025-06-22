import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os
import io
import base64
import json
import re
import requests
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Set up API configuration
TOGETHER_API_KEY = "ddffab75bf5985ae5d3adbaccacaad50eb08cbff7c25b7c907fb3e4b41183bde"  
LLM_PROVIDER = "together"  # Options: "together", "openai", "anthropic"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Default Together AI model

st.set_page_config(page_title="Flood Analysis", layout="wide")

# Add custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: #ffffff;
        border-radius: 0.5rem 0.5rem 0 0.5rem;
        align-self: flex-end;
        margin-left: 30%;
    }
    .chat-message.bot {
        background-color: #475063;
        border-radius: 0.5rem 0.5rem 0.5rem 0;
        border-left: 5px solid #1E88E5;
        margin-right: 30%;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 40px;
        max-height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding-left: 10px;
    }
    .streamlit-expanderContent {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .custom-card {
        border-radius: 0.5rem;
        background-color: #262730;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .custom-metric {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .custom-metric-label {
        font-size: 0.8rem;
        color: #9e9e9e;
    }
    .download-btn {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background-color: #0d6efd;
        color: white;
        border-radius: 0.5rem;
        text-decoration: none;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .download-btn:hover {
        background-color: #0b5ed7;
    }
    .thinking {
        font-style: italic;
        color: #9e9e9e;
        margin-bottom: 1rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-left: 0.5rem;
    }
    .lstm-badge {
        background-color: #6f42c1;
        color: white;
    }
    .rf-badge {
        background-color: #20c997;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to create download link
def get_download_link(df, filename, text):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a class="download-btn" href="{href}" download="{filename}">{text} 📥</a>'

# Function to create figure download link
def get_figure_download_link(fig, filename, text):
    """Generate a link to download the plotly figure as HTML"""
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    b64 = base64.b64encode(html_bytes).decode()
    href = f'data:text/html;base64,{b64}'
    return f'<a class="download-btn" href="{href}" download="{filename}">{text} 📥</a>'

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'dataset_summary' not in st.session_state:
    st.session_state.dataset_summary = None

if 'model_type' not in st.session_state:
    st.session_state.model_type = "random_forest"  # Default model type

# LSTM Data Preprocessing Functions

def preprocess_for_lstm(df, sequence_length=7):
    """Preprocess data for LSTM models following the notebook's approach"""
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Set Time as index for easier time-based operations
    if 'Time' in data.columns:
        data = data.set_index('Time').copy()
    
    # Handle missing values
    for col in data.columns:
        data[col] = data[col].interpolate(method='time', limit=3)
    
    # Forward fill any remaining gaps
    data.fillna(method='ffill', inplace=True)
    
    # Extract date features
    data['Month'] = data.index.month
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    
    # Create hydrological features
    data['Discharge_change'] = data['Discharge(m^3 s^-1)'].diff()
    
    # Create lag features
    for lag in [1, 2, 3, 7]:
        data[f'Precip_lag_{lag}'] = data['Precip(mm h^-1)'].shift(lag)
        data[f'Discharge_lag_{lag}'] = data['Discharge(m^3 s^-1)'].shift(lag)
    
    # Create window-based features
    data['Precip_sum_7d'] = data['Precip(mm h^-1)'].rolling(window=7).sum()
    data['Discharge_mean_7d'] = data['Discharge(m^3 s^-1)'].rolling(window=7).mean()
    
    # Create interaction features
    if 'SM(%)' in data.columns:
        data['Precip_x_SM'] = data['Precip(mm h^-1)'] * data['SM(%)'] / 100
    
    # Fill NaNs after feature creation
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Define the important features
    important_features = [
        'Discharge(m^3 s^-1)', 'Precip(mm h^-1)', 'PET(mm h^-1)', 
        'Discharge_lag_1', 'Discharge_lag_3', 'Precip_lag_1', 'Precip_lag_7',
        'Precip_sum_7d', 'Discharge_mean_7d', 'Month_sin', 'Month_cos'
    ]
    
    # Add soil moisture if available
    if 'SM(%)' in data.columns:
        important_features.extend(['SM(%)', 'Precip_x_SM'])
    
    # Use only features that exist in the dataframe
    feature_cols = [col for col in important_features if col in data.columns]
    
    # Reset the index to get Time back as a column
    data = data.reset_index()
    
    return data, feature_cols

def create_sequence_data(df, feature_cols, input_length=7, target_col='Discharge(m^3 s^-1)', target_ahead=1):
    """Create sequence data for LSTM prediction"""
    # Get the values for the selected features
    feature_data = df[feature_cols].values
    
    # Create empty arrays for sequences and targets
    X = []
    y = []
    
    # Create sequences
    for i in range(len(df) - input_length - target_ahead + 1):
        # Input sequence
        X.append(feature_data[i:i+input_length])
        
        # Target value
        y.append(df[target_col].iloc[i+input_length+target_ahead-1])
    
    return np.array(X), np.array(y)

def predict_with_lstm(df, flood_threshold, sequence_length=7, target_ahead=1):
    """Generate predictions using the LSTM models"""
    # Check if LSTM models are available
    if not ('classification_model' in st.session_state.model and 'regression_model' in st.session_state.model):
        return None, None, "LSTM models not available"
    
    try:
        # Process data
        processed_df, feature_cols = preprocess_for_lstm(df, sequence_length)
        
        # Ensure we have enough data
        if len(processed_df) < sequence_length + target_ahead:
            return None, None, "Not enough data for prediction"
        
        # Get the last sequence
        last_sequence = processed_df[feature_cols].values[-sequence_length:]
        
        # Reshape for LSTM input [samples, time steps, features]
        X_pred = last_sequence.reshape(1, sequence_length, len(feature_cols))
        
        # Get predictions from both models
        discharge_pred = st.session_state.model['regression_model'].predict(X_pred)[0][0]
        flood_prob = st.session_state.model['classification_model'].predict(X_pred)[0][0]
        
        return discharge_pred, flood_prob, None
    
    except Exception as e:
        return None, None, f"Error in LSTM prediction: {str(e)}"

# LLM Interface Functions
def call_llm_api(prompt, context=None, provider=LLM_PROVIDER, model=LLM_MODEL):
    """Call the LLM API with the given prompt and context"""
    if provider == "together":
        return call_together_api(prompt, context, model)
    elif provider == "openai":
        return call_openai_api(prompt, context, model)
    elif provider == "anthropic":
        return call_anthropic_api(prompt, context, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def call_together_api(prompt, context=None, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Call the Together AI API with the given prompt and context"""
    try:
        # Full prompt with context if provided
        if context:
            system_prompt = context
            full_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]"""
        else:
            full_prompt = f"""<s>[INST] You are a helpful assistant that specializes in hydrology and flood prediction analysis.

{prompt} [/INST]"""

        # API call
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.7,
            "repetition_penalty": 1.0
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/completions",
            headers=headers,
            json=payload
        )
        
        response_json = response.json()
        
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['text'].strip()
        else:
            error_msg = response_json.get('error', {}).get('message', 'Unknown error')
            return f"Error from Together AI: {error_msg}"
    
    except Exception as e:
        return f"Error calling Together AI API: {str(e)}"

def generate_template_response(query, df, flood_threshold):
    """Generate responses based on pre-defined templates"""
    query_lower = query.lower()
    
    if "average discharge" in query_lower or "mean discharge" in query_lower:
        return f"The average discharge in this dataset is {df['Discharge(m^3 s^-1)'].mean():.2f} m³/s, with a standard deviation of {df['Discharge(m^3 s^-1)'].std():.2f} m³/s."
    
    elif "flood threshold" in query_lower:
        return f"The flood threshold is {flood_threshold:.2f} m³/s, which corresponds to the 95th percentile of discharge values in this dataset."
    
    elif "how many flood events" in query_lower or "number of flood events" in query_lower:
        # Calculate flood events
        df_tmp = df.copy()
        df_tmp['is_flood'] = df_tmp['Discharge(m^3 s^-1)'] > flood_threshold
        df_tmp['flood_group'] = (df_tmp['is_flood'] != df_tmp['is_flood'].shift()).cumsum()
        flood_count = len(df_tmp[df_tmp['is_flood']].groupby('flood_group'))
        flood_days = df_tmp['is_flood'].sum()
        
        return f"There are {flood_count} distinct flood events in this dataset, comprising a total of {flood_days} days ({flood_days/len(df)*100:.1f}% of the record)."
    
    elif "largest flood event" in query_lower or "peak discharge" in query_lower:
        # Find largest flood
        max_discharge = df['Discharge(m^3 s^-1)'].max()
        max_date = df.loc[df['Discharge(m^3 s^-1)'].idxmax(), 'Time']
        
        return f"The largest flood event had a peak discharge of {max_discharge:.2f} m³/s on {max_date.strftime('%Y-%m-%d')}. This is {max_discharge/flood_threshold:.1f} times the flood threshold."
    
    elif "characteristics" in query_lower or "basin" in query_lower or "watershed" in query_lower:
        avg_discharge = df['Discharge(m^3 s^-1)'].mean()
        max_discharge = df['Discharge(m^3 s^-1)'].max()
        ratio = max_discharge / avg_discharge
        
        basin_type = "highly flashy" if ratio > 25 else "moderately flashy" if ratio > 10 else "relatively stable"
        
        # Create lag features to find correlation lag
        temp_df = df.copy()
        lag_corrs = []
        for lag in range(1, 20):
            temp_df[f'Precip_lag_{lag}'] = temp_df['Precip(mm h^-1)'].shift(lag)
            corr = temp_df['Discharge(m^3 s^-1)'].corr(temp_df[f'Precip_lag_{lag}'])
            lag_corrs.append((lag, corr))
        
        # Find max correlation lag
        max_corr_lag = max(lag_corrs, key=lambda x: x[1])[0]
        
        response = f"""
        Based on the dataset spanning from {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}, this basin shows the following hydrological characteristics:
        
        • Mean discharge: {avg_discharge:.2f} m³/s
        • Maximum discharge: {max_discharge:.2f} m³/s (peaking factor: {ratio:.1f})
        • Flood threshold (95th percentile): {flood_threshold:.2f} m³/s
        • Peak discharge to mean ratio: {ratio:.1f}, indicating a {basin_type} basin
        • Typical lag time between precipitation and discharge response: approximately {max_corr_lag} days
        
        The data suggests this is a {basin_type} catchment with moderate to rapid response to precipitation events.
        """
        return response
    
    elif "hysteresis" in query_lower or "loop" in query_lower:
        return """
        Hysteresis patterns reveal the complex relationship between precipitation and discharge during flood events:
        
        • Clockwise loops indicate quick runoff response, typical in watersheds with efficient drainage networks, impervious surfaces, or steep topography.
        
        • Counter-clockwise loops suggest delayed discharge response, typical in watersheds with significant storage capacity, permeable soils, or groundwater contribution.
        
        • Figure-8 patterns indicate complex watershed behavior with both quick and delayed responses.
        
        The width of the hysteresis loop reveals the strength of the memory effect - wider loops indicate stronger memory effects in the watershed.
        
        Try exploring the Hysteresis Analysis tab to visualize these patterns for specific flood events in your dataset.
        """
    
    elif "basin memory" in query_lower or "memory" in query_lower:
        # Create lag features to find correlation lag
        temp_df = df.copy()
        lag_corrs = []
        for lag in range(1, 20):
            temp_df[f'Precip_lag_{lag}'] = temp_df['Precip(mm h^-1)'].shift(lag)
            corr = temp_df['Discharge(m^3 s^-1)'].corr(temp_df[f'Precip_lag_{lag}'])
            lag_corrs.append((lag, corr))
        
        # Find max correlation lag
        max_corr_lag = max(lag_corrs, key=lambda x: x[1])[0]
        max_corr = max(lag_corrs, key=lambda x: x[1])[1]
        
        return f"""
        Basin memory refers to how long precipitation events influence discharge in this watershed.
        
        Based on lagged correlation analysis:
        
        • The strongest correlation occurs at a lag of {max_corr_lag} days (correlation coefficient: {max_corr:.3f})
        • This indicates that precipitation has its maximum effect on discharge after approximately {max_corr_lag} days
        • The memory effect continues for roughly {max_corr_lag * 2} days before diminishing substantially
        
        You can view the full basin memory analysis in the "Basin Memory Analysis" tab.
        """
    
    elif "predict" in query_lower or "forecast" in query_lower:
        if 'classification_model' in st.session_state.model and st.session_state.model_type == 'lstm':
            # Get LSTM prediction
            discharge_pred, flood_prob, error = predict_with_lstm(df, flood_threshold)
            
            if error:
                return f"Error generating forecast: {error}"
            
            return f"""
            LSTM Forecast for tomorrow:
            • Predicted discharge: {discharge_pred:.2f} m³/s
            • Flood probability: {flood_prob:.1%}
            • Exceeds threshold ({flood_threshold:.2f} m³/s): {'Yes' if discharge_pred > flood_threshold else 'No'}
            
            This prediction uses an advanced LSTM model with attention mechanism, trained to capture complex hydrological patterns including:
            • Basin memory effects (how past precipitation influences current discharge)
            • Hysteresis patterns (non-linear relationships between precipitation and discharge)
            • Seasonal patterns and soil moisture conditions
            
            For a detailed forecast visualization, use the {{forecast: days=7}} command.
            """
        elif st.session_state.model:
            return """
            I can generate discharge forecasts using the trained Random Forest model.
            
            The model uses the following features:
            • Precipitation and Discharge history (lag variables)
            • Rolling averages and sums
            • Seasonal factors
            
            To see a detailed forecast, check the '{{forecast: days=7}}' command which will generate a visual prediction.
            """
        else:
            return "To generate forecasts, you'll need to first train a prediction model using the 'Train Prediction Model' button in the sidebar, or load LSTM models if available."
    
    elif "correlation" in query_lower or "correlate" in query_lower:
        # Calculate basic correlations
        corr_columns = ['Discharge(m^3 s^-1)', 'Precip(mm h^-1)', 'PET(mm h^-1)']
        if 'SM(%)' in df.columns:
            corr_columns.append('SM(%)')
        
        corr = df[corr_columns].corr()
        discharge_corr = corr['Discharge(m^3 s^-1)'].drop('Discharge(m^3 s^-1)')
        
        top_corr = discharge_corr.abs().sort_values(ascending=False).index[0]
        top_corr_val = discharge_corr[top_corr]
        
        return f"""
        The variable most strongly correlated with discharge is {top_corr} with a correlation coefficient of {top_corr_val:.3f}.
        
        You can explore the full correlation analysis in the "Correlation Analysis" tab, which includes:
        • Correlation matrix heatmap
        • Bar chart of variables correlated with discharge
        • Scatter plots with trend lines
        • Lag correlation analysis
        """
    
    # Default response with dataset overview
    return f"""
    This dataset contains {len(df)} records from {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}.
    
    Key statistics:
    • Average discharge: {df['Discharge(m^3 s^-1)'].mean():.2f} m³/s
    • Maximum discharge: {df['Discharge(m^3 s^-1)'].max():.2f} m³/s
    • Flood threshold (95th percentile): {flood_threshold:.2f} m³/s
    
    You can ask questions about:
    • Basin characteristics and flood patterns
    • Correlation between variables
    • Hysteresis patterns in flood events
    • Basin memory and lagged effects
    • Forecasts and predictions
    """

def generate_dataset_context(df, flood_threshold):
    """Generate a context description of the dataset for the LLM"""
    
    # Basic dataset stats
    record_count = len(df)
    time_range = f"{df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}"
    mean_discharge = df['Discharge(m^3 s^-1)'].mean()
    max_discharge = df['Discharge(m^3 s^-1)'].max()
    
    # Identify flood events
    df_analysis = df.copy()
    df_analysis['is_flood'] = df_analysis['Discharge(m^3 s^-1)'] > flood_threshold
    df_analysis['flood_group'] = (df_analysis['is_flood'] != df_analysis['is_flood'].shift()).cumsum()
    flood_count = len(df_analysis[df_analysis['is_flood']].groupby('flood_group'))
    flood_days = df_analysis['is_flood'].sum()
    flood_percentage = (flood_days / record_count) * 100
    
    # Find largest flood event
    max_discharge_date = df.loc[df['Discharge(m^3 s^-1)'].idxmax(), 'Time'].strftime('%Y-%m-%d')
    
    # Column data
    columns = df.columns.tolist()
    
    # Check which model is being used
    model_type = st.session_state.model_type if 'model_type' in st.session_state else "random_forest"
    model_info = "using LSTM with attention mechanisms" if model_type == "lstm" else "using Random Forest regression"
    
    # Generate context
    context = f"""
You are a specialized hydrologist and flood prediction expert assistant. You're analyzing a specific hydro-meteorological dataset with the following characteristics:

DATASET SUMMARY:
- Records: {record_count}
- Time Range: {time_range}
- Average Discharge: {mean_discharge:.2f} m³/s
- Maximum Discharge: {max_discharge:.2f} m³/s on {max_discharge_date}
- Flood Threshold (95th percentile): {flood_threshold:.2f} m³/s
- Number of Flood Events: {flood_count}
- Flood Days: {flood_days} ({flood_percentage:.1f}% of record)
- Model Type: {model_type.upper()} {model_info}

AVAILABLE VARIABLES:
{', '.join(columns)}

When answering questions:
1. Use specific values from this dataset
2. Be concise but informative
3. Remember that "Discharge(m^3 s^-1)" is the main target variable
4. Explain hydrological concepts in clear terms
5. If you're not sure about a specific detail, stick to established hydrological principles
6. Suggest relevant analyses when appropriate (time series, correlation, hysteresis, basin memory)

Your audience is someone interested in understanding flood patterns and predictions based on this dataset.
    """
    
    return context

def extract_analysis_command(response):
    """Extract analysis commands from LLM response"""
    # Look for analysis commands in the format {{analysis_type: parameters}}
    pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(pattern, response)
    
    commands = []
    for match in matches:
        try:
            parts = match.split(':')
            if len(parts) >= 1:
                command_type = parts[0].strip()
                parameters = ':'.join(parts[1:]).strip() if len(parts) > 1 else ""
                commands.append({
                    "type": command_type,
                    "parameters": parameters
                })
        except Exception as e:
            print(f"Error parsing command: {match}, Error: {str(e)}")
    
    # Clean the response by removing the command patterns
    clean_response = re.sub(pattern, "", response)
    
    return clean_response, commands

def prepare_user_query_context(dataset_summary, query, analysis_history):
    """Prepare context for user query based on dataset summary and analysis history"""
    context = generate_dataset_context(dataset_summary['df'], dataset_summary['flood_threshold'])
    
    # Add information about what analyses have been done
    if analysis_history:
        context += "\n\nThe user has already explored the following analyses:\n"
        for analysis in analysis_history:
            context += f"- {analysis}\n"
    
    # Add instructions for special commands
    context += """
When appropriate, you can suggest specific analyses by including commands in your response using double curly braces. For example:

{{time_series}}
{{flood_events}}
{{correlation: Discharge(m^3 s^-1), Precip(mm h^-1)}}
{{hysteresis: flood_event_index}}
{{basin_memory}}
{{forecast: days=7}}

The application will process these commands and display the relevant visualizations.
"""
    
    return context

# Main layout
st.title("🌊 Advanced Flood Analysis and Prediction")

# Main content area with tabs
tab1, tab2 = st.tabs(["Analysis Dashboard", "LLM-Powered Chat Assistant"])

with tab1:
    if st.session_state.dataset_summary is not None:
        df = st.session_state.dataset_summary['df']
        flood_threshold = st.session_state.dataset_summary['flood_threshold']
        
        # Display basic info
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", len(df))
        col2.metric("Time Range", f"{df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}")
        col3.metric("Average Discharge", f"{df['Discharge(m^3 s^-1)'].mean():.2f} m³/s")
        
        # Calculate flood threshold (95th percentile)
        st.metric("Flood Threshold (95th percentile)", f"{flood_threshold:.2f} m³/s")
        
        # Show active model badge
        if st.session_state.model:
            if 'classification_model' in st.session_state.model:
                st.markdown(f"<div class='model-badge lstm-badge'>LSTM Model Active</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='model-badge rf-badge'>Random Forest Model Active</div>", unsafe_allow_html=True)
        
        # Analysis options
        analysis = st.selectbox(
            "Choose Analysis",
            ["Time Series", "Flood Events", "Correlation Analysis", "Hysteresis Analysis", "Basin Memory Analysis", "Forecasting"]
        )
        
        # Store current analysis in session state
        st.session_state.current_analysis = analysis
        
        # Add analysis to history if it's new
        if analysis not in st.session_state.analysis_history:
            st.session_state.analysis_history.append(analysis)
        
        if analysis == "Time Series":
            # Time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Time'], y=df['Discharge(m^3 s^-1)'], name="Discharge"))
            fig.add_trace(go.Scatter(x=df['Time'], y=df['Precip(mm h^-1)'], name="Precipitation", yaxis="y2"))
            
            # Add soil moisture if available
            if 'SM(%)' in df.columns:
                fig.add_trace(go.Scatter(x=df['Time'], y=df['SM(%)'], name="Soil Moisture", yaxis="y3", line=dict(dash='dash')))
            
            # Update layout
            fig.update_layout(
                title="Discharge and Precipitation Time Series",
                yaxis_title="Discharge (m³/s)",
                yaxis2=dict(title="Precipitation (mm/h)", overlaying="y", side="right"),
                yaxis3=dict(title="Soil Moisture (%)", overlaying="y", side="right", position=0.95),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_download_link(df, "time_series_data.csv", "Download Time Series Data"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_figure_download_link(fig, "time_series_plot.html", "Download Plot"), unsafe_allow_html=True)
            
            # Time window selection
            st.subheader("Zoom to Time Window")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", df['Time'].min().date())
            with col2:
                end_date = st.date_input("End Date", df['Time'].max().date())
            
            # Convert to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

            filtered_df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]
            
            if not filtered_df.empty:
                # Create zoomed plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=filtered_df['Time'], y=filtered_df['Discharge(m^3 s^-1)'], name="Discharge"))
                fig2.add_trace(go.Scatter(x=filtered_df['Time'], y=filtered_df['Precip(mm h^-1)'], name="Precipitation", yaxis="y2"))
                
                # Add soil moisture if available
                if 'SM(%)' in df.columns:
                    fig2.add_trace(go.Scatter(x=filtered_df['Time'], y=filtered_df['SM(%)'], name="Soil Moisture", yaxis="y3", line=dict(dash='dash')))
                
                # Update layout
                fig2.update_layout(
                    title=f"Time Series: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    yaxis_title="Discharge (m³/s)",
                    yaxis2=dict(title="Precipitation (mm/h)", overlaying="y", side="right"),
                    yaxis3=dict(title="Soil Moisture (%)", overlaying="y", side="right", position=0.95),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Download zoomed data
                st.markdown(get_download_link(filtered_df, f"time_series_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv", 
                                             "Download Filtered Data"), unsafe_allow_html=True)
            else:
                st.warning("No data in the selected time range.")
        
        elif analysis == "Flood Events":
            # Identify flood events
            df['is_flood'] = df['Discharge(m^3 s^-1)'] > flood_threshold
            
            # Plot with flood threshold
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Time'], y=df['Discharge(m^3 s^-1)'], name="Discharge"))
            fig.add_trace(go.Scatter(
                x=df['Time'], 
                y=[flood_threshold] * len(df),
                name="Flood Threshold",
                line=dict(dash="dash", color="red")
            ))
            
            # Add precipitation on secondary axis
            fig.add_trace(go.Bar(
                x=df['Time'],
                y=df['Precip(mm h^-1)'],
                name="Precipitation",
                opacity=0.5,
                yaxis="y2"
            ))
            
            # Update layout
            fig.update_layout(
                title="Discharge with Flood Threshold",
                yaxis_title="Discharge (m³/s)",
                yaxis2=dict(title="Precipitation (mm/h)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Count flood days
            flood_days = df['is_flood'].sum()
            st.metric("Flood Days", f"{flood_days} ({flood_days/len(df)*100:.1f}% of record)")
            
            # Group flood events
            df['flood_group'] = (df['is_flood'] != df['is_flood'].shift()).cumsum()
            
            # Get flood events
            flood_events = []
            for group, data in df[df['is_flood']].groupby('flood_group'):
                if len(data) >= 2:  # Only consider events with at least 2 days
                    flood_events.append({
                        'start': data['Time'].min(),
                        'end': data['Time'].max(),
                        'duration': len(data),
                        'peak': data['Discharge(m^3 s^-1)'].max(),
                        'peak_date': data.loc[data['Discharge(m^3 s^-1)'].idxmax(), 'Time'],
                        'total_precip': data['Precip(mm h^-1)'].sum(),
                        'group': group
                    })
            
            # Display top flood events
            if flood_events:
                st.subheader(f"Top Flood Events (Total: {len(flood_events)})")
                
                # Sort by peak discharge
                sorted_events = sorted(flood_events, key=lambda x: x['peak'], reverse=True)
                
                # Create a table
                event_df = pd.DataFrame(sorted_events[:10])
                event_df['start'] = event_df['start'].dt.strftime('%Y-%m-%d')
                event_df['end'] = event_df['end'].dt.strftime('%Y-%m-%d')
                event_df['peak_date'] = event_df['peak_date'].dt.strftime('%Y-%m-%d')
                event_df = event_df.rename(columns={
                    'start': 'Start Date',
                    'end': 'End Date',
                    'duration': 'Duration (days)',
                    'peak': 'Peak Discharge (m³/s)',
                    'peak_date': 'Peak Date',
                    'total_precip': 'Total Precipitation (mm)'
                }).drop(columns=['group'])
                
                st.dataframe(event_df, use_container_width=True)
                
                # Download options
                st.markdown(get_download_link(event_df, "flood_events.csv", "Download Flood Events"), unsafe_allow_html=True)
        
        elif analysis == "Correlation Analysis":
            # Correlation matrix
            corr_columns = ['Discharge(m^3 s^-1)', 'Precip(mm h^-1)', 'PET(mm h^-1)']
            
            # Add optional columns if they exist
            optional_columns = ['SM(%)', 'Groundwater (mm)', 'Fast Flow(mm*1000)', 'Slow Flow(mm*1000)', 'Base Flow(mm*1000)']
            for col in optional_columns:
                if col in df.columns:
                    corr_columns.append(col)
            
            # Create lag features for correlation analysis
            temp_df = df.copy()
            for lag in [1, 3, 7]:
                lag_col = f'Precip_lag_{lag}'
                temp_df[lag_col] = temp_df['Precip(mm h^-1)'].shift(lag)
                corr_columns.append(lag_col)
            
            # Remove NaN values
            temp_df = temp_df.dropna()
            
            # Calculate correlation
            corr = temp_df[corr_columns].corr()
            
            # Heatmap
            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            st.markdown(get_download_link(corr.reset_index(), "correlation_matrix.csv", "Download Correlation Matrix"), unsafe_allow_html=True)
            
            # Correlation with discharge
            st.subheader("Correlation with Discharge")
            discharge_corr = corr['Discharge(m^3 s^-1)'].sort_values(ascending=False).drop('Discharge(m^3 s^-1)')
            
            # Bar chart of correlations
            fig = px.bar(
                x=discharge_corr.index, 
                y=discharge_corr.values,
                labels={'x': 'Variable', 'y': 'Correlation Coefficient'},
                title="Variables Correlated with Discharge",
                color=discharge_corr.values,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            st.subheader("Explore Relationships")
            col1, col2 = st.columns(2)
            with col1:
                y_var = st.selectbox("Select variable to correlate with Discharge", corr_columns[1:])
            
            with col2:
                point_count = st.slider("Number of points to display (sampling)", 
                                       min_value=100, max_value=len(df), value=min(2000, len(df)))
            
            # Sample data for better performance
            if len(df) > point_count:
                sample_df = df.sample(point_count)
            else:
                sample_df = df
            
            # Create scatter plot
            fig = px.scatter(
                sample_df, x='Discharge(m^3 s^-1)', y=y_var,
                title=f"Discharge vs {y_var}",
                trendline="ols",
                opacity=0.6
            )
            
            # Add correlation coefficient to title
            corr_value = corr['Discharge(m^3 s^-1)'][y_var]
            fig.update_layout(title=f"Discharge vs {y_var} (Correlation: {corr_value:.3f})")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis == "Hysteresis Analysis":
            # Calculate flood threshold
            df['is_flood'] = df['Discharge(m^3 s^-1)'] > flood_threshold
            
            # Group flood events
            df['flood_group'] = (df['is_flood'] != df['is_flood'].shift()).cumsum()
            
            # Get flood events
            flood_events = []
            for group, data in df[df['is_flood']].groupby('flood_group'):
                if len(data) >= 3:  # Only consider events with at least 3 days
                    flood_events.append({
                        'start': data['Time'].min(),
                        'end': data['Time'].max(),
                        'peak': data['Discharge(m^3 s^-1)'].max(),
                        'group': group
                    })
            
            if flood_events:
                # Sort by peak discharge
                flood_events = sorted(flood_events, key=lambda x: x['peak'], reverse=True)
                
                # Select an event
                selected_event = st.selectbox(
                    "Select flood event",
                    [f"Event {i+1}: {event['start'].strftime('%Y-%m-%d')} to {event['end'].strftime('%Y-%m-%d')} (Peak: {event['peak']:.2f} m³/s)"
                     for i, event in enumerate(flood_events[:5])],
                    index=0
                )
                
                event_idx = int(selected_event.split(':')[0].replace("Event ", "")) - 1
                event_group = flood_events[event_idx]['group']
                
                # Get data for this event (plus 3 days before and after)
                event_start = flood_events[event_idx]['start'] - pd.Timedelta(days=3)
                event_end = flood_events[event_idx]['end'] + pd.Timedelta(days=3)
                event_data = df[(df['Time'] >= event_start) & (df['Time'] <= event_end)].copy()
                
                # Add time index
                event_data['time_idx'] = range(len(event_data))
                
                # Create two plots
                col1, col2 = st.columns(2)
                
                # Hysteresis plot
                fig1 = px.scatter(
                    event_data, x='Precip(mm h^-1)', y='Discharge(m^3 s^-1)',
                    color='time_idx', color_continuous_scale='Viridis',
                    title="Hysteresis Loop (Precipitation vs Discharge)",
                    labels={"time_idx": "Time Progression"}
                )
                
                # Add arrows to show direction
                for i in range(len(event_data)-1):
                    fig1.add_annotation(
                        x=event_data['Precip(mm h^-1)'].iloc[i+1],
                        y=event_data['Discharge(m^3 s^-1)'].iloc[i+1],
                        ax=event_data['Precip(mm h^-1)'].iloc[i],
                        ay=event_data['Discharge(m^3 s^-1)'].iloc[i],
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="red"
                    )
                
                col1.plotly_chart(fig1, use_container_width=True)
                
                # Time series plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=event_data['Time'], 
                    y=event_data['Discharge(m^3 s^-1)'],
                    name="Discharge"
                ))
                fig2.add_trace(go.Bar(
                    x=event_data['Time'],
                    y=event_data['Precip(mm h^-1)'],
                    name="Precipitation",
                    opacity=0.5
                ))
                
                # Add flood threshold
                fig2.add_trace(go.Scatter(
                    x=event_data['Time'],
                    y=[flood_threshold] * len(event_data),
                    name="Flood Threshold",
                    line=dict(dash="dash", color="red")
                ))
                
                fig2.update_layout(title="Event Time Series")
                col2.plotly_chart(fig2, use_container_width=True)
                
                # Download options
                col1.markdown(get_figure_download_link(fig1, "hysteresis_loop.html", "Download Hysteresis Plot"), unsafe_allow_html=True)
                col2.markdown(get_figure_download_link(fig2, "event_time_series.html", "Download Time Series Plot"), unsafe_allow_html=True)
                st.markdown(get_download_link(event_data, "event_data.csv", "Download Event Data"), unsafe_allow_html=True)
                
                # Explanation
                st.markdown("""
                ### Interpreting Hysteresis Patterns
                
                Hysteresis patterns reveal the complex relationship between precipitation and discharge during flood events:
                
                - **Clockwise loops** indicate that runoff responds quickly to precipitation. This is typical in watersheds with:
                    - Efficient drainage networks
                    - Impervious surfaces
                    - Steep topography
                
                - **Counter-clockwise loops** suggest delayed discharge response, typical in watersheds with:
                    - Significant storage capacity
                    - Permeable soils
                    - Groundwater contribution
                
                - **Figure-8 patterns** indicate complex watershed behavior with both quick and delayed responses.
                
                - **Loop width** reveals the strength of the hysteresis effect - wider loops indicate stronger memory effects in the watershed.
                """)
            else:
                st.warning("No significant flood events found for hysteresis analysis.")
                
        elif analysis == "Basin Memory Analysis":
            st.subheader("Basin Memory Analysis")
            
            # Create lag features
            temp_df = df.copy()
            lag_corrs = []
            
            # Calculate correlation for different lags
            max_lag = 30
            for lag in range(1, max_lag + 1):
                temp_df[f'Precip_lag_{lag}'] = temp_df['Precip(mm h^-1)'].shift(lag)
                # Calculate correlation between discharge and lagged precipitation
                corr = temp_df['Discharge(m^3 s^-1)'].corr(temp_df[f'Precip_lag_{lag}'])
                lag_corrs.append((lag, corr))
            
            # Convert to DataFrame for plotting
            lag_df = pd.DataFrame(lag_corrs, columns=['Lag (days)', 'Correlation'])
            
            # Create plot
            fig = px.line(
                lag_df, x='Lag (days)', y='Correlation',
                markers=True,
                title="Basin Memory: Correlation between Discharge and Lagged Precipitation"
            )
            
            # Find maximum correlation
            max_corr_idx = lag_df['Correlation'].argmax()
            max_corr_lag = lag_df.iloc[max_corr_idx]['Lag (days)']
            max_corr = lag_df.iloc[max_corr_idx]['Correlation']
            
            # Add vertical line at max correlation
            fig.add_vline(
                x=max_corr_lag,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Maximum correlation at lag: {max_corr_lag} days",
                annotation_position="top right"
            )
            
            # Calculate memory half-life
            half_corr = max_corr / 2
            memory_half_life = None
            for i in range(int(max_corr_lag), len(lag_df)):
                if lag_df.iloc[i]['Correlation'] < half_corr:
                    memory_half_life = lag_df.iloc[i]['Lag (days)']
                    break
            
            # Add vertical line at memory half-life if found
            if memory_half_life:
                fig.add_vline(
                    x=memory_half_life,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Memory half-life: {memory_half_life} days",
                    annotation_position="top right"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            st.markdown(get_download_link(lag_df, "basin_memory_data.csv", "Download Basin Memory Data"), unsafe_allow_html=True)
            st.markdown(get_figure_download_link(fig, "basin_memory_plot.html", "Download Plot"), unsafe_allow_html=True)
            
            # Display memory metrics
            col1, col2 = st.columns(2)
            col1.metric("Lag with Maximum Correlation", f"{max_corr_lag:.0f} days")
            col2.metric("Maximum Correlation", f"{max_corr:.3f}")
            
            if memory_half_life:
                col1, col2 = st.columns(2)
                col1.metric("Memory Half-life", f"{memory_half_life:.0f} days")
                col2.metric("Watershed Memory", f"~{memory_half_life * 2:.0f} days")
            
            st.markdown("""
            ### Interpreting Basin Memory Analysis
            
            Basin memory refers to how long precipitation events influence discharge in a watershed:
            
            - **Lag with maximum correlation** indicates the typical time delay between precipitation and peak discharge.
            
            - **Memory half-life** shows how long the influence of precipitation persists in the watershed.
            
            - **Shorter memory** (steep decline in correlation) suggests:
                - Efficient drainage
                - Limited storage capacity
                - Flashy hydrological response
            
            - **Longer memory** (gradual decline) suggests:
                - Significant groundwater contribution
                - Higher storage capacity
                - More regulated flows
            """)

        elif analysis == "Forecasting":
            st.subheader("Discharge Forecasting")
            
            if not st.session_state.model:
                st.warning("No model loaded. Please train a Random Forest model or load LSTM models from the sidebar.")
            else:
                # Forecast parameters
                col1, col2 = st.columns(2)
                with col1:
                    forecast_days = st.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=7)
                with col2:
                    use_lstm = st.checkbox("Use LSTM models (if available)", 
                                          value='classification_model' in st.session_state.model and st.session_state.model_type == 'lstm')
                
                if st.button("Generate Forecast"):
                    # Determine which model to use
                    if use_lstm and 'classification_model' in st.session_state.model:
                        # LSTM forecast
                        st.markdown("### LSTM Model Forecast")
                        
                        # Current data
                        current_df = df.copy()
                        
                        # Preprocess data
                        processed_df, feature_cols = preprocess_for_lstm(current_df)
                        
                        # Make predictions for each future day
                        forecast_results = []
                        current_sequence = None
                        
                        # Get the last valid sequence for initial prediction
                        last_sequence = processed_df[feature_cols].values[-7:]
                        current_sequence = last_sequence.reshape(1, 7, len(feature_cols))
                        
                        for i in range(forecast_days):
                            # Get predictions
                            discharge_pred = st.session_state.model['regression_model'].predict(current_sequence)[0][0]
                            flood_prob = st.session_state.model['classification_model'].predict(current_sequence)[0][0]
                            
                            # Create next date
                            next_date = current_df['Time'].max() + pd.Timedelta(days=i+1)
                            
                            # Store prediction
                            forecast_results.append({
                                'Date': next_date,
                                'Discharge': discharge_pred,
                                'Flood_Probability': flood_prob,
                                'Exceeds_Threshold': discharge_pred > flood_threshold
                            })
                            
                            # Update sequence for next prediction by rolling forward
                            # Remove oldest time step and add new prediction
                            
                            # For simplicity, we'll create an augmented sequence with predicted discharge
                            # In a real implementation, you would update all features properly
                            new_sequence = np.roll(current_sequence[0], -1, axis=0)
                            
                            # Update discharge value in the sequence
                            # This is a simplification - in practice you should update all features
                            discharge_idx = feature_cols.index('Discharge(m^3 s^-1)') if 'Discharge(m^3 s^-1)' in feature_cols else 0
                            new_sequence[-1, discharge_idx] = discharge_pred
                            
                            # Update current sequence for next prediction
                            current_sequence = new_sequence.reshape(1, 7, len(feature_cols))
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame(forecast_results)
                        
                        # Display forecast results
                        st.subheader("Forecast Results")
                        forecast_display = forecast_df.copy()
                        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                        forecast_display['Discharge (m³/s)'] = forecast_display['Discharge'].round(2)
                        forecast_display['Flood Probability'] = (forecast_display['Flood_Probability'] * 100).round(1).astype(str) + '%'
                        forecast_display['Exceeds Threshold'] = forecast_display['Exceeds_Threshold'].map({True: 'Yes', False: 'No'})
                        
                        st.dataframe(forecast_display[['Date', 'Discharge (m³/s)', 'Flood Probability', 'Exceeds Threshold']], use_container_width=True)
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Add historical data (last 30 days)
                        hist_data = df.iloc[-30:]
                        fig.add_trace(go.Scatter(
                            x=hist_data['Time'],
                            y=hist_data['Discharge(m^3 s^-1)'],
                            name='Historical Discharge',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Discharge'],
                            name='LSTM Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add flood probability as area
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Flood_Probability'] * forecast_df['Discharge'].max() * 0.2,
                            name='Flood Probability',
                            fill='tozeroy',
                            line=dict(color='rgba(128, 0, 128, 0.3)'),
                            yaxis="y2"
                        ))
                        
                        # Add flood threshold
                        fig.add_trace(go.Scatter(
                            x=pd.concat([hist_data['Time'], forecast_df['Date']]),
                            y=[flood_threshold] * (len(hist_data) + len(forecast_df)),
                            name='Flood Threshold',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{forecast_days}-Day Discharge Forecast (LSTM Model)",
                            xaxis_title="Date",
                            yaxis_title="Discharge (m³/s)",
                            yaxis2=dict(
                                title="Flood Probability",
                                overlaying="y",
                                side="right",
                                range=[0, 1],
                                showgrid=False
                            ),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download options
                        st.markdown(get_download_link(forecast_display, "lstm_forecast.csv", "Download Forecast"), unsafe_allow_html=True)
                        st.markdown(get_figure_download_link(fig, "lstm_forecast_plot.html", "Download Plot"), unsafe_allow_html=True)
                        
                    elif st.session_state.model:
                        # Random Forest forecast
                        st.markdown("### Random Forest Model Forecast")
                        
                        # Use the existing Random Forest model
                        model = st.session_state.model['model']
                        scaler = st.session_state.model['scaler']
                        features = st.session_state.model['features']
                        
                        # Create forecast dataframe starting from the last date
                        last_date = df['Time'].max()
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
                        
                        # Create a copy of the dataframe for forecasting
                        forecast_df = df.copy()
                        
                        # Generate predictions day by day
                        predictions = []
                        
                        for i in range(forecast_days):
                            next_date = last_date + pd.Timedelta(days=i+1)
                            
                            # Create lag features if needed
                            if not all(col in forecast_df.columns for col in features):
                                # Create lag features
                                for lag in [1, 2, 3, 7]:
                                    forecast_df[f'Precip_lag_{lag}'] = forecast_df['Precip(mm h^-1)'].shift(lag)
                                    forecast_df[f'Discharge_lag_{lag}'] = forecast_df['Discharge(m^3 s^-1)'].shift(lag)
                                
                                # Create rolling features
                                forecast_df['Precip_roll_7d'] = forecast_df['Precip(mm h^-1)'].rolling(window=7).sum()
                                forecast_df['Discharge_roll_7d'] = forecast_df['Discharge(m^3 s^-1)'].rolling(window=7).mean()
                                
                                # Add month feature for seasonality
                                forecast_df['Month'] = forecast_df['Time'].dt.month
                                forecast_df['Month_sin'] = np.sin(2 * np.pi * forecast_df['Month'] / 12)
                                forecast_df['Month_cos'] = np.cos(2 * np.pi * forecast_df['Month'] / 12)
                            
                            # Get the most recent row of data
                            latest_data = forecast_df.iloc[-1:].copy()
                            
                            # Set the date for the new row
                            latest_data['Time'] = next_date
                            latest_data['Month'] = next_date.month
                            latest_data['Month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
                            latest_data['Month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
                            
                            # Extract features for prediction
                            X_pred = latest_data[features].values
                            
                            # Scale features
                            X_pred_scaled = scaler.transform(X_pred)
                            
                            # Make prediction
                            pred = model.predict(X_pred_scaled)[0]
                            
                            # Create new row with prediction
                            latest_data['Discharge(m^3 s^-1)'] = pred
                            
                            # For simplicity, set precipitation to 0 (ideally would use weather forecast)
                            latest_data['Precip(mm h^-1)'] = 0
                            
                            # Append to forecast dataframe
                            forecast_df = pd.concat([forecast_df, latest_data])
                            
                            # Store prediction
                            predictions.append({
                                'Date': next_date,
                                'Discharge': pred,
                                'Exceeds_Threshold': pred > flood_threshold
                            })
                        
                        # Create forecast results dataframe
                        forecast_results = pd.DataFrame(predictions)
                        
                        # Display forecast
                        st.subheader("Forecast Results")
                        forecast_display = forecast_results.copy()
                        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                        forecast_display['Discharge (m³/s)'] = forecast_display['Discharge'].round(2)
                        forecast_display['Exceeds Threshold'] = forecast_display['Exceeds_Threshold'].map({True: 'Yes', False: 'No'})
                        
                        st.dataframe(forecast_display[['Date', 'Discharge (m³/s)', 'Exceeds Threshold']], use_container_width=True)
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Add historical data (last 30 days)
                        hist_data = df.iloc[-30:]
                        fig.add_trace(go.Scatter(
                            x=hist_data['Time'],
                            y=hist_data['Discharge(m^3 s^-1)'],
                            name='Historical Discharge',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_results['Date'],
                            y=forecast_results['Discharge'],
                            name='Random Forest Forecast',
                            line=dict(color='green')
                        ))
                        
                        # Add flood threshold
                        fig.add_trace(go.Scatter(
                            x=pd.concat([hist_data['Time'], forecast_results['Date']]),
                            y=[flood_threshold] * (len(hist_data) + len(forecast_results)),
                            name='Flood Threshold',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{forecast_days}-Day Discharge Forecast (Random Forest Model)",
                            xaxis_title="Date",
                            yaxis_title="Discharge (m³/s)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download options
                        st.markdown(get_download_link(forecast_display, "rf_forecast.csv", "Download Forecast"), unsafe_allow_html=True)
                        st.markdown(get_figure_download_link(fig, "rf_forecast_plot.html", "Download Plot"), unsafe_allow_html=True)
                
                # Add model comparison if both models are available
                if 'classification_model' in st.session_state.model and 'model' in st.session_state.model:
                    st.subheader("Model Comparison")
                    st.markdown("""
                    ### LSTM vs Random Forest Model Comparison
                    
                    **LSTM Model Strengths:**
                    - Better captures sequential patterns and temporal dependencies
                    - Accounts for hysteresis effects in flood events
                    - Provides flood probability estimates
                    - More accurate for larger flood events
                    
                    **Random Forest Strengths:**
                    - Faster training and prediction
                    - More interpretable (feature importance)
                    - Less prone to overfitting with limited data
                    - Generally robust for routine discharge predictions
                    
                    Choose LSTM for critical flood forecasting scenarios, and Random Forest for 
                    routine monitoring or when computational resources are limited.
                    """)
    with tab2:
        st.header("Chat with Flood Prediction Assistant")
    
        if st.session_state.dataset_summary is not None:
            df = st.session_state.dataset_summary['df']
            flood_threshold = st.session_state.dataset_summary['flood_threshold']
            
            # Display dataset info at top of chat
            st.info(f"Dataset: {st.session_state.dataset_summary['name']} | Records: {len(df)} | Time range: {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}")
            
            # Display active model badge
            if st.session_state.model:
                if 'classification_model' in st.session_state.model:
                    st.markdown(f"<div class='model-badge lstm-badge'>LSTM Model Active</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='model-badge rf-badge'>Random Forest Model Active</div>", unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"<div class='chat-message user'><div class='message'>{message['content']}</div></div>", unsafe_allow_html=True)
                elif message["role"] == "assistant":
                    st.markdown(f"<div class='chat-message bot'><div class='message'>{message['content']}</div></div>", unsafe_allow_html=True)
                elif message["role"] == "system":
                    st.markdown(f"<div class='thinking'>{message['content']}</div>", unsafe_allow_html=True)
            
            # User input
            user_input = st.text_input("Ask a question about your data:", key="user_query")
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.button("Ask", use_container_width=True)
            with col2:
                # Template questions
                st.markdown("**Try asking about:**")
            
            # Template question suggestions
            template_questions = [
                "Explain the hydrological characteristics of this basin",
                "What are the key drivers of flood events in this dataset?",
                "How does soil moisture affect discharge in this watershed?",
                "When is the watershed most susceptible to flooding?",
                "What's the typical lag between precipitation and peak discharge?",
                "Can you predict discharge for the next 7 days?",
                "What's the forecast for potential flood peaks?",
                "How do seasonal patterns affect flood risk?"
            ]
            
            # Display template questions as buttons across multiple columns
            cols = st.columns(2)
            for i, question in enumerate(template_questions):
                col_idx = i % 2
                if cols[col_idx].button(question, key=f"template_{i}"):
                    user_input = question
                    submit_button = True
            
            # Process user input
            if submit_button and user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Add thinking message
                thinking_message = {"role": "system", "content": "Thinking..."}
                st.session_state.messages.append(thinking_message)
                
                # Force refresh to show thinking message
                st.rerun()
            
            # Continue from previous run if there's a thinking message
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "system" and st.session_state.messages[-1]["content"] == "Thinking...":
                # Remove thinking message
                st.session_state.messages.pop()
                
                # Get the user's query (the message before "thinking")
                user_query = st.session_state.messages[-1]["content"]
                
                try:
                    # Try using the LLM API
                    context = prepare_user_query_context(
                        st.session_state.dataset_summary,
                        user_query,
                        st.session_state.analysis_history
                    )
                    
                    # Call LLM API with error handling
                    try:
                        response = call_llm_api(user_query, context, LLM_PROVIDER, LLM_MODEL)
                        clean_response, commands = extract_analysis_command(response)
                    except Exception as e:
                        # Fall back to template response if API fails
                        st.warning(f"LLM API call failed. Using template-based response instead.")
                        clean_response = generate_template_response(user_query, df, flood_threshold)
                        commands = []
                except:
                    # Fall back to template response if anything fails
                    clean_response = generate_template_response(user_query, df, flood_threshold)
                    commands = []
                
                # Add bot message
                st.session_state.messages.append({"role": "assistant", "content": clean_response})
                
                # Process any analysis commands
                if commands:
                    for command in commands:
                        command_type = command["type"]
                        parameters = command["parameters"]
                        
                        # Set current analysis for the next tab view
                        if command_type == "time_series":
                            st.session_state.current_analysis = "Time Series"
                            st.info("Time series analysis is available in the 'Analysis Dashboard' tab.")
                            
                        elif command_type == "flood_events":
                            st.session_state.current_analysis = "Flood Events"
                            st.info("Flood events analysis is available in the 'Analysis Dashboard' tab.")
                            
                        elif command_type == "correlation":
                            st.session_state.current_analysis = "Correlation Analysis"
                            st.info("Correlation analysis is available in the 'Analysis Dashboard' tab.")
                            
                        elif command_type == "hysteresis":
                            st.session_state.current_analysis = "Hysteresis Analysis"
                            st.info("Hysteresis analysis is available in the 'Analysis Dashboard' tab.")
                            
                        elif command_type == "basin_memory":
                            st.session_state.current_analysis = "Basin Memory Analysis"
                            st.info("Basin memory analysis is available in the 'Analysis Dashboard' tab.")
                            
                        elif command_type == "forecast" and st.session_state.model:
                            # Extract days parameter
                            days_match = re.search(r'days=(\d+)', parameters)
                            forecast_days = int(days_match.group(1)) if days_match else 7
                            
                            st.session_state.current_analysis = "Forecasting"
                            st.info(f"A {forecast_days}-day forecast is available in the 'Analysis Dashboard' tab.")
                
                # Force refresh to show the response
                st.rerun()
        else:
            st.info("Please upload a CSV file in the sidebar to begin analysis.")


# Sidebar for file upload and options
with st.sidebar:
    st.header("Data & Settings")
    
    # LLM Provider selection
    st.subheader("LLM Settings")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["together"],
        index=0
    )
    
    if llm_provider == "together":
        llm_model = st.selectbox(
            "Model",
            options=[
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "togethercomputer/llama-3-8b-instruct",
                "meta-llama/Llama-2-70b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.2"
            ],
            index=0
        )
    
    # Update global variables
    LLM_PROVIDER = llm_provider
    LLM_MODEL = llm_model
    
    # API Key input (hidden but showing the last 4 characters)
    api_key_display = "•" * 10 + TOGETHER_API_KEY[-4:] if TOGETHER_API_KEY else ""
    st.text_input("Together AI API Key", value=api_key_display, disabled=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Hydro-Meteorological CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Convert Time column to datetime
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time')
            
            # Display dataset info
            st.success(f"✅ Dataset loaded: {uploaded_file.name}")
            st.info(f"Records: {len(df)}")
            st.info(f"Time range: {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}")
            
            # Calculate and store flood threshold
            flood_threshold = df['Discharge(m^3 s^-1)'].quantile(0.95)
            
            # Store dataset summary for LLM context
            st.session_state.dataset_summary = {
                "df": df,
                "name": uploaded_file.name,
                "flood_threshold": flood_threshold
            }
            
            # Model selection
            st.markdown("---")
            st.header("Model Selection")
            
            model_type = st.selectbox(
                "Select prediction model type",
                options=["random_forest", "lstm"],
                index=0,
                format_func=lambda x: "Random Forest" if x == "random_forest" else "LSTM"
            )
            st.session_state.model_type = model_type
            
            # Model training section
            if model_type == "random_forest":
                if st.button("Train Random Forest Model"):
                    with st.spinner("Training Random Forest model..."):
                        # Feature engineering for model
                        model_df = df.copy()
                        
                        # Create lag features
                        for lag in [1, 2, 3, 7]:
                            model_df[f'Precip_lag_{lag}'] = model_df['Precip(mm h^-1)'].shift(lag)
                            model_df[f'Discharge_lag_{lag}'] = model_df['Discharge(m^3 s^-1)'].shift(lag)
                        
                        # Create rolling features
                        model_df['Precip_roll_7d'] = model_df['Precip(mm h^-1)'].rolling(window=7).sum()
                        model_df['Discharge_roll_7d'] = model_df['Discharge(m^3 s^-1)'].rolling(window=7).mean()
                        
                        # Add month feature for seasonality
                        model_df['Month'] = model_df['Time'].dt.month
                        model_df['Month_sin'] = np.sin(2 * np.pi * model_df['Month'] / 12)
                        model_df['Month_cos'] = np.cos(2 * np.pi * model_df['Month'] / 12)
                        
                        # Drop NaN values
                        model_df = model_df.dropna()
                        
                        # Select features
                        features = [
                            'Precip(mm h^-1)', 'PET(mm h^-1)', 
                            'Precip_lag_1', 'Precip_lag_3', 'Precip_lag_7',
                            'Discharge_lag_1', 'Discharge_lag_2', 'Discharge_lag_3',
                            'Precip_roll_7d', 'Discharge_roll_7d',
                            'Month_sin', 'Month_cos'
                        ]
                        
                        # Add optional features if they exist
                        if 'SM(%)' in model_df.columns:
                            features.append('SM(%)')
                        if 'Groundwater (mm)' in model_df.columns:
                            features.append('Groundwater (mm)')
                        
                        X = model_df[features]
                        y = model_df['Discharge(m^3 s^-1)']
                        
                        # Split data
                        train_size = int(len(X) * 0.8)
                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train random forest model
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        
                        # Evaluate model
                        y_pred = model.predict(X_test_scaled)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Store model and scaler in session state
                        st.session_state.model = {
                            'model': model,
                            'scaler': scaler,
                            'features': features,
                            'metrics': {
                                'mae': mae,
                                'r2': r2
                            }
                        }
                        st.session_state.model_type = "random_forest"
                        
                        st.success(f"Random Forest model trained successfully! MAE: {mae:.2f}, R²: {r2:.2f}")
            
            else:  # LSTM model
                if st.button("Load LSTM Models"):
                    with st.spinner("Loading LSTM models..."):
                        try:
                            # Check if LSTM models exist
                            if os.path.exists('g1_classification_model.h5') and os.path.exists('g1_regression_model.h5'):
                                class_model = tf.keras.models.load_model('g1_classification_model.h5')
                                reg_model = tf.keras.models.load_model('g1_regression_model.h5')
                                
                                # Load metadata if available
                                if os.path.exists('model_metadata.pkl'):
                                    with open('model_metadata.pkl', 'rb') as f:
                                        model_metadata = pickle.load(f)
                                else:
                                    model_metadata = {
                                        'flood_threshold_p95': flood_threshold,
                                        'input_sequence_length': 7
                                    }
                                
                                # Store models in session state
                                st.session_state.model = {
                                    'classification_model': class_model,
                                    'regression_model': reg_model,
                                    'metadata': model_metadata
                                }
                                st.session_state.model_type = "lstm"
                                
                                st.success("✅ LSTM models loaded successfully")
                            else:
                                # If models don't exist, explain upload process
                                st.error("LSTM models not found. Please ensure the following files exist in the app directory:")
                                st.code("g1_classification_model.h5\ng1_regression_model.h5\nmodel_metadata.pkl (optional)")
                                st.info("These models should be trained following the approach from your notebook.")
                        except Exception as e:
                            st.error(f"Error loading LSTM models: {str(e)}")
            
            # Display model status
            if st.session_state.model:
                if 'classification_model' in st.session_state.model:
                    st.info("✅ LSTM models loaded and ready for prediction")
                else:
                    # Display Random Forest model metrics
                    metrics = st.session_state.model['metrics']
                    st.markdown(f"**Random Forest model performance:**")
                    st.markdown(f"MAE: {metrics['mae']:.2f} m³/s")
                    st.markdown(f"R²: {metrics['r2']:.2f}")
            else:
                st.warning("No prediction model loaded. Please train or load a model.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.info("Please ensure your CSV file contains the required columns including 'Time' and 'Discharge(m^3 s^-1)'.")
            df = None