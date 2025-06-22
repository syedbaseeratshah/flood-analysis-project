# 🌊 Flood Peak Analysis & Prediction Project

A Streamlit web application for hydrological data analysis and flood peak prediction using machine learning models with an AI-powered chat interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)

## 🎯 What This App Does

- **📊 Interactive Analysis**: Time series visualization, flood event detection, correlation analysis, hysteresis patterns, and basin memory analysis
- **🤖 ML Predictions**: Random Forest and LSTM models for discharge forecasting
- **💬 AI Chat Interface**: LLM-powered assistant for natural language data queries using Together AI
- **📈 Visualizations**: Interactive Plotly charts with download capabilities
- **📋 Data Export**: CSV and HTML export functionality for analysis results

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **ML Models**: Scikit-learn (Random Forest), TensorFlow/Keras (LSTM)
- **Data Processing**: Pandas, NumPy
- **AI Integration**: Together AI API
- **Deployment**: Streamlit Cloud

## 📊 Data Requirements

Your CSV file needs these columns:

**Required:**
- `Time` - Date/timestamp (YYYY-MM-DD format)
- `Discharge(m^3 s^-1)` - Stream discharge
- `Precip(mm h^-1)` - Precipitation rate

**Optional (for enhanced analysis):**
- `PET(mm h^-1)` - Potential evapotranspiration
- `SM(%)` - Soil moisture
- `Groundwater (mm)` - Groundwater levels

## 🚀 Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/flood-analysis-dashboard.git
cd flood-analysis-dashboard
```

### 2. Create Virtual Environment
```bash
python -m venv flood_env
source flood_env/bin/activate  # Windows: flood_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys (Optional - for chat feature)
Create `.streamlit/secrets.toml`:
```toml
[api_keys]
together_ai = "your_together_ai_api_key_here"
```

### 5. Run Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 🎮 How to Use

1. **Upload Data**: Use sidebar to upload your CSV file
2. **Train Model**: Choose Random Forest or load pre-trained LSTM models
3. **Analyze**: Select from 6 analysis types in the main dashboard
4. **Chat**: Ask questions about your data in natural language
5. **Export**: Download results and visualizations

### Example Chat Queries:
- "What are the flood characteristics of this basin?"
- "Predict discharge for the next 7 days"
- "How does precipitation correlate with discharge?"

## 📁 Project Structure

```
flood-analysis-dashboard/
├── app.py                  # Streamlit application with RF
├── flood_prediction_app.py # Main Streamlit application with LSTM             
├── requirements.txt        # Python dependencies
├── Datasets/              # Sample data folder
├── LSTM model file/       # Pre-trained models
├── Visualizations/        # Generated plots
└── sample.ipynb          # Development notebook
```

## 🤖 ML Models

### Random Forest
- **Training**: Automatic with your data
- **Features**: Lag variables, rolling statistics, seasonal components
- **Output**: Discharge predictions + performance metrics

### LSTM (Pre-trained)
- **Files**: `g1_classification_model.h5`, `g1_regression_model.h5`
- **Capabilities**: Multi-step forecasting + flood probability
- **Architecture**: Sequence-to-sequence with attention

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
requests>=2.31.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
scipy>=1.10.0
python-dotenv>=1.0.0
```

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Add API keys in Streamlit Cloud secrets
4. Deploy!

### Docker
```bash
docker build -t flood-analysis .
docker run -p 8501:8501 flood-analysis
```

## 🔧 Troubleshooting

**Data Upload Issues:**
- Ensure CSV has required columns with exact names
- Check date format: YYYY-MM-DD
- Maximum file size: 200MB

**Model Training Fails:**
- Need minimum 100 data points
- Check for excessive missing values (>50%)
- Ensure discharge values are positive

**API Errors:**
- Verify Together AI API key in secrets
- Check internet connection
- App works without API (uses template responses)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the framework
- Together AI for LLM integration
- Open source ML libraries (TensorFlow, Scikit-learn)
- Hydrological research community

---

**Built for flood analysis and prediction using modern ML and AI techniques** 🌊
