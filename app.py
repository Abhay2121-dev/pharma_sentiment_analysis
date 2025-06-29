import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="AI Pharma Feedback Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #2E86AB, #A23B72);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.prediction-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.training-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'categories' not in st.session_state:
    st.session_state.categories = ['Question', 'Suggestion', 'Complaint']

class PharmaZeroShotClassifier:
    """Zero-shot pharmaceutical feedback classifier using BART-large-mnli"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.categories = st.session_state.categories
        
    def load_model(self):
        """Load the pre-trained model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            with st.spinner("🤖 Loading AI model... This may take a moment"):
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
                self.model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
                self.classifier = pipeline("zero-shot-classification", 
                                          model=self.model, 
                                          tokenizer=self.tokenizer,
                                          device=0 if torch.cuda.is_available() else -1)
            return True
        return True  # Model already loaded
    
    def create_sample_data(self):
        """Generate comprehensive sample pharmaceutical feedback data"""
        sample_data = {
            'feedback_text': [
                # Questions
                "Customer called asking about proper timing for taking CardioMax with their morning routine. They mentioned they usually take their blood pressure medication at 7 AM and wanted to know if there would be any interactions with our product.",
                "Patient inquired about the recommended dosage for DiabetCare when traveling across time zones. They're concerned about maintaining consistent blood sugar levels during their business trip to Europe.",
                "Healthcare provider asked about contraindications for PainRelief Plus in patients with kidney disease. They want to ensure safe prescribing for their elderly population.",
                "Pharmacy technician called to clarify storage requirements for CardioMax. They need to know the optimal temperature range and humidity conditions for their new storage facility.",
                "Patient's family member asked about potential side effects of DiabetCare in elderly patients. Their 85-year-old mother is starting the medication and they want to know what to watch for.",
                
                # Suggestions
                "During our product demonstration at the cardiology clinic, Dr. Martinez suggested developing a liquid formulation of CardioMax for elderly patients who have difficulty swallowing pills. This could significantly improve patient compliance.",
                "Clinical pharmacist at Regional Medical Center recommended creating a mobile app to help patients track their DiabetCare dosing schedule. They believe this would reduce medication errors and improve adherence.",
                "Nurse practitioner suggested developing educational materials in Spanish for PainRelief Plus. She mentioned that many of her patients struggle with the English-only instructions currently provided.",
                "Hospital system administrator proposed bulk packaging options for CardioMax to reduce waste and costs. They suggested 30-day supply bottles instead of the current 90-day bottles for better inventory management.",
                "Patient advocate recommended creating a patient assistance program for DiabetCare. Many uninsured patients are unable to afford the medication and may benefit from financial support options.",
                
                # Complaints
                "Customer filed a formal complaint after experiencing severe allergic reaction requiring emergency department treatment. Patient had no known allergies but developed hives within 30 minutes of taking DiabetCare tablet.",
                "Patient reported that PainRelief Plus caused severe nausea and vomiting that lasted for three days. They had to discontinue the medication and seek alternative pain management options.",
                "Pharmacy received multiple complaints about CardioMax tablets having an unusual bitter taste that wasn't present in previous batches. Patients are having difficulty taking their medication due to the taste.",
                "Healthcare provider filed complaint about delayed shipment of DiabetCare that caused several patients to miss doses. The delay occurred during a critical period when patients were adjusting their treatment regimens.",
                "Patient complained that PainRelief Plus packaging is difficult to open for people with arthritis. The child-resistant cap requires too much force and many elderly patients cannot access their medication.",
                
                # Additional Questions
                "Insurance coordinator called to verify the NDC number for CardioMax 50mg tablets. They need this information to process prior authorization requests for multiple patients.",
                "Clinical researcher asked about the mechanism of action for DiabetCare. They're conducting a comparative effectiveness study and need detailed pharmacological information.",
                "Patient asked if PainRelief Plus can be taken with ibuprofen for breakthrough pain. They're currently managing chronic back pain and need additional relief options.",
                
                # Additional Suggestions  
                "Geriatrician suggested developing a once-daily formulation of CardioMax. Current twice-daily dosing is challenging for patients with memory issues and medication compliance problems.",
                "Pain management specialist recommended creating a patch formulation of PainRelief Plus for patients who cannot tolerate oral medications due to gastrointestinal issues.",
                
                # Additional Complaints
                "Patient complained that DiabetCare caused unexpected weight gain of 15 pounds over two months. They're concerned about cardiovascular risks and want to discuss alternative treatments.",
                "Caregiver reported that their spouse became confused and disoriented after starting PainRelief Plus. The symptoms resolved after discontinuing the medication but they're seeking safer alternatives."
            ],
            'category': [
                'Question', 'Question', 'Question', 'Question', 'Question',  # Questions
                'Suggestion', 'Suggestion', 'Suggestion', 'Suggestion', 'Suggestion',  # Suggestions
                'Complaint', 'Complaint', 'Complaint', 'Complaint', 'Complaint',  # Complaints
                'Question', 'Question', 'Question',  # Additional Questions
                'Suggestion', 'Suggestion',  # Additional Suggestions
                'Complaint', 'Complaint'  # Additional Complaints
            ],
            'product_name': [
                'CardioMax', 'DiabetCare', 'PainRelief Plus', 'CardioMax', 'DiabetCare',
                'CardioMax', 'DiabetCare', 'PainRelief Plus', 'CardioMax', 'DiabetCare',
                'DiabetCare', 'PainRelief Plus', 'CardioMax', 'DiabetCare', 'PainRelief Plus',
                'CardioMax', 'DiabetCare', 'PainRelief Plus',
                'CardioMax', 'PainRelief Plus',
                'DiabetCare', 'PainRelief Plus'
            ],
            'urgency_level': [
                'Medium', 'Medium', 'High', 'Low', 'Medium',
                'Low', 'Low', 'Medium', 'Low', 'Medium',
                'High', 'High', 'Medium', 'High', 'Medium',
                'Low', 'Low', 'Medium',
                'Low', 'Medium',
                'Medium', 'High'
            ]
        }
        return pd.DataFrame(sample_data)
    
    def predict(self, text):
        """Make prediction on new text using zero-shot classification"""
        if self.classifier is None:
            self.load_model()
            
        # Run zero-shot classification
        results = self.classifier(
            text,
            candidate_labels=self.categories,
            multi_label=False
        )
        
        # Format results
        predicted_class = results['labels'][0]
        probabilities = {label: score for label, score in zip(results['labels'], results['scores'])}
        
        return predicted_class, probabilities

def show_home_page():
    """Display home page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>🚀 Zero-Shot Pharma Feedback Classifier</h2>
            <p>Advanced BART-based classification system for pharmaceutical feedback analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="training-card">
            <h3>⚡ Instant Classification</h3>
            <ul>
                <li>No training required</li>
                <li>Pre-trained BART model</li>
                <li>Zero-shot capabilities</li>
                <li>Multi-category analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h3>🔮 AI Predictions</h3>
            <ul>
                <li>Real-time classification</li>
                <li>Confidence scoring</li>
                <li>Probability distributions</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analytics</h3>
            <ul>
                <li>Sample dataset insights</li>
                <li>Batch processing</li>
                <li>Export capabilities</li>
                <li>Pharma-specific analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    ### 🎯 Getting Started
    1. **🔮 Make predictions**: Navigate to Single Prediction to classify feedback
    2. **📊 View analytics**: Explore the sample dataset in the Analytics section
    3. **📝 Process batches**: Upload CSV files in the Batch Processing section
    4. **⚙️ Customize**: Add or modify categories in the sidebar
    """)
    
    # Model info
    st.markdown("""
    ### 🔬 About the AI Model
    This application uses Facebook's **BART-large-mnli** model, which is:
    - Trained on the Multi-Genre Natural Language Inference (MNLI) dataset
    - Capable of zero-shot classification through natural language inference
    - State-of-the-art for text classification without task-specific training
    - Particularly effective for pharmaceutical domain analysis
    """)

def show_prediction_page():
    """Display prediction page"""
    st.header("🔮 Single Prediction")
    
    # Ensure classifier is initialized
    if st.session_state.classifier is None:
        st.session_state.classifier = PharmaZeroShotClassifier()
    
    classifier = st.session_state.classifier
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        if classifier.load_model():
            st.session_state.model_loaded = True
            st.success("✅ AI model loaded successfully!")
    
    # Category customization
    st.sidebar.subheader("🔧 Classification Categories")
    new_category = st.sidebar.text_input("Add new category:", "")
    if st.sidebar.button("Add Category") and new_category:
        if new_category not in st.session_state.categories:
            st.session_state.categories.append(new_category)
            classifier.categories = st.session_state.categories
            st.sidebar.success(f"✅ Added category: {new_category}")
    
    # Display current categories
    st.sidebar.write("**Current Categories:**")
    for i, category in enumerate(st.session_state.categories):
        cols = st.sidebar.columns([3, 1])
        cols[0].write(category)
        if cols[1].button("❌", key=f"del_{i}"):
            if len(st.session_state.categories) > 1:
                st.session_state.categories.remove(category)
                classifier.categories = st.session_state.categories
                st.sidebar.success(f"✅ Removed category: {category}")
                st.experimental_rerun()
            else:
                st.sidebar.error("❌ Need at least one category")
    
    # Input text area
    st.subheader("📝 Enter Pharmaceutical Feedback")
    user_input = st.text_area(
        "Type your feedback here:",
        placeholder="Customer called asking about proper timing for taking CardioMax with their morning routine...",
        height=150
    )
    
    # Prediction button
    if st.button("🔍 Classify Feedback", type="primary") and user_input:
        try:
            with st.spinner("🤖 Analyzing feedback using AI..."):
                prediction, probabilities = classifier.predict(user_input)
            
            # Display results
            st.markdown(f"""
            <div class="prediction-card">
                <h3>🎯 Prediction Result</h3>
                <h2 style="color: #FFD700;">{prediction}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence scores
            st.subheader("📊 Confidence Scores")
            
            # Create probability dataframe
            prob_df = pd.DataFrame(list(probabilities.items()), 
                                 columns=['Category', 'Confidence'])
            prob_df = prob_df.sort_values('Confidence', ascending=False)
            prob_df['Confidence'] = prob_df['Confidence'].round(4)
            
            # Display as metrics
            cols = st.columns(len(prob_df))
            for i, (_, row) in enumerate(prob_df.iterrows()):
                with cols[i]:
                    st.metric(
                        row['Category'], 
                        f"{row['Confidence']:.1%}",
                        delta=None
                    )
            
            # Confidence bar chart
            fig = px.bar(prob_df, x='Category', y='Confidence', 
                        title="Classification Confidence Scores",
                        color='Confidence',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence gauge for top prediction
            max_confidence = prob_df.iloc[0]['Confidence']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = max_confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Confidence: {prediction}"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def show_analytics_page():
    """Display analytics page"""
    st.header("📊 Data Analytics")
    
    # Ensure classifier is initialized
    if st.session_state.classifier is None:
        st.session_state.classifier = PharmaZeroShotClassifier()
    
    classifier = st.session_state.classifier
    
    # Load sample data
    df = classifier.create_sample_data()
    
    if df is not None:
        # Dataset overview
        st.subheader("📋 Sample Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Records", len(df))
        with col2:
            st.metric("🏷️ Categories", df['category'].nunique())
        with col3:
            avg_length = df['feedback_text'].str.len().mean()
            st.metric("📏 Avg Text Length", f"{avg_length:.0f}")
        with col4:
            st.metric("💊 Products", df['product_name'].nunique() if 'product_name' in df.columns else 'N/A')
        
        # Category distribution
        st.subheader("🏷️ Category Distribution")
        category_counts = df['category'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=category_counts.values, names=category_counts.index,
                           title="Feedback Categories Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(x=category_counts.index, y=category_counts.values,
                           title="Category Counts",
                           labels={'x': 'Category', 'y': 'Count'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Text length analysis
        st.subheader("📏 Text Length Analysis")
        df['text_length'] = df['feedback_text'].str.len()
        df['word_count'] = df['feedback_text'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df, x='text_length', 
                                  title="Distribution of Text Lengths",
                                  nbins=30)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(df, x='category', y='text_length',
                           title="Text Length by Category")
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Product analysis
        if 'product_name' in df.columns:
            st.subheader("💊 Product Analysis")
            product_category = pd.crosstab(df['product_name'], df['category'])
            
            fig_heatmap = px.imshow(product_category.values,
                                  x=product_category.columns,
                                  y=product_category.index,
                                  labels=dict(x="Category", y="Product", color="Count"),
                                  title="Product-Category Distribution")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Product urgency analysis
            if 'urgency_level' in df.columns:
                st.subheader("⚠️ Urgency Level Analysis")
                urgency_counts = df['urgency_level'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_urgency = px.pie(values=urgency_counts.values, 
                                        names=urgency_counts.index,
                                        title="Urgency Level Distribution")
                    st.plotly_chart(fig_urgency, use_container_width=True)
                
                with col2:
                    urgency_by_category = pd.crosstab(df['category'], df['urgency_level'])
                    fig_urgency_bar = px.bar(urgency_by_category,
                                           title="Urgency by Category",
                                           barmode='group')
                    st.plotly_chart(fig_urgency_bar, use_container_width=True)

def show_batch_processing_page():
    """Display batch processing page"""
    st.header("📝 Batch Processing")
    
    # Ensure classifier is initialized
    if st.session_state.classifier is None:
        st.session_state.classifier = PharmaZeroShotClassifier()
    
    classifier = st.session_state.classifier
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        if classifier.load_model():
            st.session_state.model_loaded = True
            st.success("✅ AI model loaded successfully!")
    
    # File uploader
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with feedback to classify:",
        type=["csv"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            if 'feedback_text' not in df.columns:
                st.error("❌ CSV file must contain a 'feedback_text' column")
                return
                
            # Show preview
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process button
            if st.button("🔍 Classify All Feedback", type="primary"):
                with st.spinner("🤖 Processing batch predictions using AI..."):
                    # Make predictions
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(df['feedback_text']):
                        status_text.text(f"Processing {i+1}/{len(df)} feedback items...")
                        progress_bar.progress((i+1)/len(df))
                        
                        pred, prob = classifier.predict(text)
                        predictions.append(pred)
                        probabilities.append(prob)
                    
                    # Add predictions to dataframe
                    df['predicted_category'] = predictions
                    
                    # Add probability columns
                    for label in classifier.categories:
                        df[f'prob_{label.lower()}'] = [p.get(label, 0.0) for p in probabilities]
                    
                    # Show results
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Results",
                        data=csv,
                        file_name=f"pharma_feedback_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Results summary
                    st.subheader("📊 Batch Results Summary")
                    pred_counts = df['predicted_category'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pred = px.pie(values=pred_counts.values, 
                                        names=pred_counts.index,
                                        title="Predicted Category Distribution")
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col2:
                        fig_pred_bar = px.bar(x=pred_counts.index, 
                                            y=pred_counts.values,
                                            title="Predicted Counts",
                                            labels={'x': 'Category', 'y': 'Count'})
                        st.plotly_chart(fig_pred_bar, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered Pharmaceutical Feedback Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Single Prediction", "📊 Analytics", "📝 Batch Processing"]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Single Prediction":
        show_prediction_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "📝 Batch Processing":
        show_batch_processing_page()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🔬 Model Information
    - **Model**: BART-large-mnli
    - **Approach**: Zero-shot Classification
    - **Categories**: Questions, Suggestions, Complaints
    - **Capabilities**: 
        - No training required
        - Real-time classification
        - Customizable categories
    """)

if __name__ == "__main__":
    main()