# Smart Data Analysis Assistant 🔍

A Streamlit-based intelligent data analysis tool that allows users to query datasets using natural language and get instant visualizations and insights. The application is specifically demonstrated using the Titanic dataset but can work with any CSV file.

## 🌟 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Interactive Visualizations**: Get instant visual insights using Plotly
- **Dataset Overview**: Automatically generated statistics and data profiling
- **Dynamic Analysis**: Real-time code generation and execution for custom analysis
- **User-Friendly Interface**: Clean, intuitive UI with expandable code sections

## 🛠️ Technology Stack

- **Frontend & App Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express, Plotly Graph Objects
- **LLM Integration**: LangChain with Groq (Mixtral-8x7b-32768)
- **Code Generation**: Custom parsing and execution system

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run main.py
```

## 📊 Usage

1. **Upload Data**:
   - Click the file upload button
   - Select your CSV file
   - The app will automatically generate an overview of your dataset

2. **Analyze Data**:
   - Navigate to the "Query Data" tab
   - Type your question in natural language
   - Click "Analyze" to get insights

3. **Example Queries**:
   - "How many males survived?"
   - "What percentage of survivors were females?"
   - "Show the age distribution of survivors"
   - "What was the ticket fare distribution?"

## 🏗️ Project Structure

```
├── main.py                # Main application file
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables (not tracked in git)
└── README.md             # Project documentation
```

## 🔧 Key Components

1. **Dataset Overview Module**:
   - Basic statistics
   - Column details
   - Data sample preview
   - Numerical column analysis

2. **Query Interface**:
   - Natural language input
   - Code generation
   - Result visualization
   - Error handling

3. **Analysis Engine**:
   - LLM integration
   - Code parsing
   - Safe execution environment
   - Dynamic visualization generation

## 🌐 Deployment

The application is deployed on Streamlit Cloud. You can access it at:
[Your Streamlit App URL]

## ⚠️ Limitations

- LLM responses may occasionally require refinement
- Complex queries might need reformulation
- Processing time varies with dataset size
- Memory constraints on very large datasets

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing framework
- Groq for LLM API access
- LangChain for the LLM integration tools
- The open-source community for various libraries used

## 📧 Contact

[Your Name] - [Your Email]

Project Link: [Your Repository URL]
