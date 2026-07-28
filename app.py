import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocessing import SentiPreprocessor
from src.engine import SentiEngine

# Web shell framework parameter configs - Premium UI Layout
st.set_page_config(
    page_title="SentiOps Enterprise Dashboard", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# Custom High-End Cyberpunk Dark Glassmorphism Styling Injection
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: #cbd5e1 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 32px !important;
        font-weight: 700;
    }
    textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    .custom-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .pos-card {
        background: linear-gradient(90deg, rgba(22,101,52,0.3) 0%, rgba(22,101,52,0.8) 100%);
        border-left: 5px solid #22c55e;
    }
    .neg-card {
        background: linear-gradient(90deg, rgba(153,27,27,0.3) 0%, rgba(153,27,27,0.8) 100%);
        border-left: 5px solid #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title Structure Branding
st.markdown("""
    <div style='text-align: center; padding: 20px 0px 10px 0px;'>
        <h1 style='color: #ffffff; font-size: 42px; font-weight: 800; margin-bottom: 0;'>
            🧠 Senti<span style='color: #38bdf8;'>Ops</span> Enterprise
        </h1>
        <p style='color: #94a3b8; font-size: 16px; margin-top: 5px;'>
            Production-Grade Lightweight MLOps Pipeline For Local CPU Architectures
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Resource Memory Caching System
@st.cache_resource
def initialize_pipeline():
    preprocessor = SentiPreprocessor()
    engine = SentiEngine()
    return preprocessor, engine

try:
    preprocessor, engine = initialize_pipeline()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

tab1, tab2 = st.tabs(["💬 Real-Time Text Analysis", "📁 Enterprise Batch File Engine"])

# ---- TAB 1: SINGLE TEXT INFERENCE ----
with tab1:
    col_input, col_output = st.columns(2, gap="large")
    
    with col_input:
        st.markdown("### 📝 Input Feed")
        user_input = st.text_area(
            "Paste product feedback or customer review text below:", 
            height=200, 
            placeholder="Type your textual data context here..."
        )
        analyze_click = st.button("🚀 Execute Local Inference", use_container_width=True)
        
    with col_output:
        st.markdown("### 📊 Engine Analytics")
        if analyze_click:
            if user_input.strip() == "":
                st.warning("Input box empty! Please paste target text.")
            else:
                with st.spinner("Processing token distributions..."):
                    inputs = preprocessor.tokenize_and_pad([user_input])
                    results, latency, ram = engine.predict_sentiment(inputs)
                    res = results[0] # List array se pehla output safely extract kiya

                    if res['label'] == "POSITIVE":
                        st.markdown(f"""
                            <div class='custom-card pos-card'>
                                <span style='font-size: 14px; text-transform: uppercase;'>Predicted Status</span><br>
                                <span style='font-size: 26px; font-weight: 800;'>🟩 {res['label']} Sentiment</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='custom-card neg-card'>
                                <span style='font-size: 14px; text-transform: uppercase;'>Predicted Status</span><br>
                                <span style='font-size: 26px; font-weight: 800;'>🟥 {res['label']} Sentiment</span>
                            </div>
                            """, unsafe_allow_html=True)

                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Confidence Score", f"{res['confidence']}%")
                    m_col2.metric("Inference Latency", f"{latency * 1000:.2f} ms")
                    st.metric("Peak App Runtime RAM Usage", f"{ram:.2f} MB")
        else:
            st.info("Awaiting input data parameters. Enter text and trigger the inference layer engine.")

# ---- TAB 2: BATCH CSV FILE INFERENCE ----
with tab2:
    st.markdown("### 📁 Data File Ingestion System")
    uploaded_file = st.file_uploader(
        "Upload spreadsheet files containing multi-sequence arrays (CSV Format Required)", 
        type=["csv"]
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # SMART COLUMN MAPPER: Columns clean karke check karna
        original_cols = [str(c).lower().strip() for c in df.columns]
        df.columns = original_cols
        
        # Agar 'review' ya 'text' column milta hai to use standard 'text' naam de do
        if 'review' in df.columns:
            df.rename(columns={'review': 'text'}, inplace=True)
        elif 'sentence' in df.columns:
            df.rename(columns={'sentence': 'text'}, inplace=True)
        elif 'text' not in df.columns:
            # Fallback: Agar kuch nahi mila, to pehle column ko hi text maan lo (BOM issues code level safe bypass)
            df.rename(columns={df.columns[0]: 'text'}, inplace=True)
            
        with st.expander("👁️ View Source Document Raw Row Layout Preview"):
            st.dataframe(df.head(5), use_container_width=True)
            
        st.success("Target structural mapping verified successfully. Pipeline is armed.")
        
        # IMDB Dataset bohot bada hota hai (50k rows), isliye safer analysis ke liye top 100 rows use karenge
        max_rows = min(100, len(df))
        st.info(f"MLOps Optimization: Local CPU safety ke liye hum top {max_rows} rows ko process kar rahe hain.")
        
        if st.button("⚡ Trigger Massive Batch Analysis Pipeline", use_container_width=True):
            with st.spinner("Executing sequence optimizations over localized matrix layers..."):
                # Processing targeted limited slice
                df_slice = df.head(max_rows).copy()
                text_list = df_slice['text'].astype(str).tolist()
                
                inputs = preprocessor.tokenize_and_pad(text_list)
                results, latency, ram = engine.predict_sentiment(inputs)
                
                df_slice['sentiment_result'] = [r['label'] for r in results]
                df_slice['confidence_percentage'] = [r['confidence'] for r in results]
                
            st.markdown("### 🎉 Global Operational Telemetry")
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Rows Evaluated", f"{len(df_slice)} Records")
            kpi2.metric("Global Pipeline Latency", f"{latency:.3f} Seconds")
            kpi3.metric("Dedicated Processing Memory", f"{ram:.2f} MB")
            
            st.divider()
            
            chart_col, table_col = st.columns([1, 1.2], gap="large")
            
            with chart_col:
                st.markdown("#### 📈 Sentiment Composition Analytics")
                counts = df_slice['sentiment_result'].value_counts().reset_index()
                counts.columns = ['Sentiment', 'Count']
                
                fig = px.pie(
                    counts, values='Count', names='Sentiment', hole=0.5,
                    color='Sentiment',
                    color_discrete_map={'POSITIVE':'#22c55e', 'NEGATIVE':'#ef4444'}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    showlegend=True,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with table_col:
                st.markdown("#### 📝 Annotated Data Outputs Preview")
                st.dataframe(
                    df_slice[['text', 'sentiment_result', 'confidence_percentage']].head(10),
                    use_container_width=True
                )
                
                csv_output = df_slice.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download fully Annotated Enterprise Results File (CSV)",
                    data=csv_output,
                    file_name="sentiops_imdb_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
