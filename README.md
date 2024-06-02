# CoinWise

CoinWise is a comprehensive tool that analyzes cryptocurrency white papers, websites, utility, sentiment, historical data, and technical moves. It generates detailed financial reports in PDF format. The project leverages various technologies and APIs to provide in-depth analysis and optimized runtime performance using Apache Spark's PySpark, asyncio, and aiohttp.

Features
White Paper Analysis: Evaluates the problem statement, project mechanics, tokenomics, factual reliability, and level of detail.
Website Analysis: Checks content quality, team information, transparency, technical and security features, and identifies red flags.
Utility Analysis: Assesses core functionality, ecosystem integration, token utility, real-world applications, technological innovation, market demand, economic incentives, security, regulatory compliance, and community support.
Sentiment Analysis: Analyzes news sentiment using NLP and sentiment scores, incorporating news from the current day to a month back.
Market Metrics and Technical Analysis: Fetches market metrics and performs technical analysis using historical data and technical indicators.
PDF Report Generation: Compiles the analysis into a comprehensive PDF report with graphs and detailed summaries.
Technologies Used
Python: Core programming language.
Apache Spark: Used for optimized runtime with PySpark.
aiohttp: For asynchronous HTTP requests.
asyncio: To handle asynchronous operations.
nltk: Natural Language Toolkit for text processing.
yfinance: Fetches historical market data.
talib: Technical analysis library.
matplotlib: For plotting graphs.
OpenAI API: GPT for generating detailed analysis.
FPDF: For generating PDF reports.
newspaper3k: For scraping and parsing news articles.
transformers: For sentiment analysis.
sentence-transformers: For semantic similarity calculations.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/CoinWise.git
cd CoinWise
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Set up API keys in your environment:

bash
Copy code
export NEWSAPI_API_KEY='your-newsapi-key'
export OPENAI_API_KEY='your-openai-key'
export COINMARKETCAP_API_KEY='your-coinmarketcap-key'
Usage
Run the main script to start the analysis:

bash
Copy code
python evaluate_crypto_optimized.py
The script will analyze the specified cryptocurrency and generate a detailed PDF report with the analysis results.

File Structure
evaluate_crypto_optimized.py: Main script for performing the analysis and generating the PDF report.
requirements.txt: List of dependencies required for the project.
README.md: Project documentation.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License.

Example Output
The final output is a PDF report containing:

Overall Score: A composite score out of 100 based on various metrics.
Market Metrics: Analysis of market cap, trading volume, supply, price changes, and technical indicators.
Sentiment Analysis: Summary of sentiment scores, frequency of positive vs. negative news, and impact of key events.
Utility and Use Case: Evaluation of core functionality, ecosystem integration, real-world applications, and more.
White Paper Review: Detailed analysis of the white paper content.
Website Analysis: Assessment of the cryptocurrency's website based on several criteria.
