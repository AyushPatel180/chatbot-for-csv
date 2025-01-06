import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
import re

# Load environment variables
load_dotenv()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dataset Overview"
# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="mixtral-8x7b-32768"
    )

def extract_code_and_explanation(text):
    """Extract code and explanation from raw text response"""
    # Try to find code blocks
    code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    
    # If no code blocks found, try to find code based on common patterns
    if not code_blocks:
        # Look for patterns that might indicate code
        code_patterns = [
            r'# Calculate(.*?)(?=\n\n|$)',
            r'fig = px\.(.*?)(?=\n\n|$)',
            r'result = (.*?)(?=\n\n|$)',
            r'df\.(.*?)(?=\n\n|$)'
        ]
        
        for pattern in code_patterns:
            potential_code = re.findall(pattern, text, re.DOTALL)
            if potential_code:
                code_blocks = potential_code
                break
    
    # Extract explanation (text before the first code block or all text if no code block)
    explanation = text.split('```')[0] if '```' in text else text
    
    # Clean up the code
    code = '\n'.join(code_blocks).strip() if code_blocks else ''
    
    return {
        'code': code,
        'explanation': explanation.strip(),
        'visualization_type': 'plot' if 'fig' in code else 'text'
    }

def fallback_parser(text):
    """Fallback parser for when structured parsing fails"""
    try:
        # First try to parse as JSON
        try:
            return json.loads(text)
        except:
            pass
        
        # If JSON parsing fails, try to extract components manually
        return extract_code_and_explanation(text)
    except Exception as e:
        st.error(f"Fallback parsing failed: {str(e)}")
        return None

def generate_analysis(query, df_info):
    """Generate analysis using LangChain and Groq with improved parsing"""
    try:
        llm = get_llm()
        
        # Simplified prompt to reduce parsing issues
        simple_prompt = f"""Analyze the following dataset based on the query.
        
Dataset Information:
{df_info}

Query: {query}

Provide your response in the following format:
```python
# Your code here
# Make sure to use only the existing 'df' dataframe
# For visualizations, assign to 'fig'
# For calculations, assign to 'result'
```

Explanation: [Your explanation here]
"""
        
        # Get response from LLM
        response = llm.invoke([{"role": "user", "content": simple_prompt}])
        
        # Try different parsing approaches
        parsed_output = fallback_parser(response.content)
        
        if parsed_output and parsed_output.get('code'):
            # Clean up the code
            code = parsed_output['code']
            
            # Remove any file reading operations
            code = re.sub(r'pd\.read_csv\(.*?\)', 'df', code)
            
            # Ensure proper variable assignment
            if not (code.strip().endswith('fig') or code.strip().endswith('result')):
                if 'fig' in code:
                    code += '\nfig'
                else:
                    code += '\nresult'
            
            parsed_output['code'] = code
            return parsed_output
        else:
            st.error("Could not generate valid analysis code")
            return None
            
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        return None

def execute_code(code, df):
    """Execute generated code safely and return results"""
    try:
        # Create local namespace with required imports and dataframe
        local_namespace = {
            'pd': pd,
            'px': px,
            'go': go,
            'df': df,
            'result': None,
            'fig': None
        }
        
        # Execute code
        exec(code, local_namespace)
        
        # Check for results
        if local_namespace.get('fig') is not None:
            return {'type': 'plot', 'data': local_namespace['fig']}
        elif local_namespace.get('result') is not None:
            return {'type': 'text', 'data': local_namespace['result']}
        else:
            return {'type': 'error', 'data': 'No result or figure was generated'}
            
    except Exception as e:
        return {'type': 'error', 'data': f"Error executing code: {str(e)}"}
    


def render_dataset_overview():
    """Render the dataset overview tab"""
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        st.markdown("### 🔍 Column Details")
        col_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Values': df.count(),
            'Null Values': df.isnull().sum(),
            'Unique Values': df.nunique(),
            'Sample Values': [str(df[col].head(3).tolist()) for col in df.columns]
        })
        col_info.index.name = 'Column Name'
        st.dataframe(col_info, use_container_width=True)
        
        st.markdown("### 📑 Sample Data")
        st.dataframe(df.head(), use_container_width=True)
        
        # Basic statistics for numerical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            st.markdown("### 📈 Numerical Columns Statistics")
            st.dataframe(df[num_cols].describe(), use_container_width=True)

def render_query_interface():
    """Render the query interface tab"""
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("### 💭 Ask Questions About Your Data")
        st.markdown("""
        Type your question in natural language and get instant insights from your data.
        
        **Example questions:**
        - What's the distribution of [column_name]?
        - Show me the trend of [column_name] over time
        - Calculate the average [column_name] grouped by [another_column]
        - What's the correlation between [column1] and [column2]?
        """)
        
        query = st.text_area(
            "",
            key="query_input",
            placeholder="Enter your question here...",
            height=100
        )
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if query:
                with st.spinner("🤔 Analyzing your data..."):
                    # Get dataset info
                    df_info = f"""
                    Columns: {list(df.columns)}
                    Data types: {df.dtypes.to_dict()}
                    Shape: {df.shape}
                    """
                    
                    # Generate analysis
                    analysis_result = generate_analysis(query, df_info)
                    
                    if analysis_result:
                        # Display results in a clean format
                        st.markdown("---")
                        st.markdown("### 🎯 Analysis Results")
                        
                        # Display explanation in a highlighted box
                        st.info(analysis_result['explanation'])
                        
                        # Display generated code
                        with st.expander("🔧 View Generated Code", expanded=False):
                            st.code(analysis_result['code'], language='python')
                        
                        # Execute and display results
                        result = execute_code(analysis_result['code'], df)
                        
                        if result['type'] == 'plot':
                            st.markdown("### 📊 Visualization")
                            st.plotly_chart(result['data'], use_container_width=True)
                            st.caption("*Hover over the visualization to see more details*")
                        elif result['type'] == 'text':
                            st.markdown("### 📝 Analysis Output")
                            st.success("Here's what I found:")
                            st.write(result['data'])
                        else:
                            st.error(f"Error: {result['data']}")
            else:
                st.warning("Please enter a question about your data.")


def main():
    st.set_page_config(page_title="Data Analysis Assistant", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stTextInput>div>div>input {
            background-color: white;
        }
        .plot-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            background-color: white;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🔍 Smart Data Analysis Assistant")
    st.markdown("""
        Transform your data insights with natural language queries.
        Upload your CSV file and get instant analysis and visualizations.
    """)

    # File upload
    uploaded_file = st.file_uploader("📂 Choose your CSV file", type="csv", key="file_uploader")
    
    if uploaded_file is not None:
        try:
            # Only read the file if it's newly uploaded or changed
            file_contents = uploaded_file.getvalue()
            if st.session_state.df is None or st.session_state.get('last_file_contents') != file_contents:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.last_file_contents = file_contents
            
            # Create tabs
            tab1, tab2 = st.tabs(["📊 Dataset Overview", "❓ Query Data"])
            
            with tab1:
                render_dataset_overview()
                
            with tab2:
                render_query_interface()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.df = None  # Reset the session state on error
    else:
        st.info("👆 Please upload a CSV file to begin your analysis")

if __name__ == "__main__":
    main()