import pandas as pd
import re 
import textwrap
import aiohttp
import asyncio
import nltk
import yfinance
import talib  as ta
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI
from newspaper import Article
from datetime import datetime
from transformers import pipeline, BertTokenizer
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF, XPos, YPos
from pandas.plotting import table
from pyspark.sql import SparkSession

# Set API keys
NEWSAPI_API_KEY = 'your-newsapi-key'
OPENAI_API_KEY = 'your-openai-key' 
COINMARKETCAP_API_KEY = 'your-coinmarketcap-key' 
# Initialize sentiment analysis pipeline and tokenizer
sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

# Initialize Sentence Transformer model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# List of relevant keywords and phrases
relevant_keywords = [
    "bullish", "bearish", "market trend", "price increase", "price decrease", "trading volume", 
    "market cap", "volatility", "sentiment", "investor confidence", "adoption", "liquidity", 
    "regulation", "legal", "ban", "approval", "compliance", "SEC", "lawsuit", "legislation", 
    "tax", "government policy", "blockchain", "smart contract", "scalability", "security", 
    "innovation", "upgrade", "fork", "decentralization", "protocol", "development", "partnership", 
    "collaboration", "alliance", "integration", "merger", "acquisition", "investment", 
    "funding", "venture capital", "institutional investment", "hack", "breach", "vulnerability", 
    "security audit", "risk", "attack", "cybersecurity", "safety", "adoption", "use case", 
    "application", "real-world", "utility", "payment", "transaction", "network", "platform", 
    "Bitcoin", "Ethereum", "Ripple", "Litecoin", "Cardano", "Binance Coin", "Solana", 
    "Polkadot", "Dogecoin", "Stablecoin", "inflation", "interest rate", "economic growth", 
    "recession", "financial crisis", "monetary policy", "hype", "FOMO", "FUD", "bull market", 
    "bear market", "market sentiment"
]

# APACHE SPARK, ASYNCIO AND AIOHTTP WAS ADDED TO OPTIMIZE RUN TIME
# Initialize Spark session
spark = SparkSession.builder.appName("CryptoAnalysis").getOrCreate()

async def fetch_url(session, url, params=None, headers=None):
    async with session.get(url, params=params, headers=headers) as response:
        return await response.json()
    
# Function to check website
def check_website_sync(crypto_name):
    query = '''
You are an AI that evaluates the investment potential of cryptocurrencies. Your goal is to provide a score out of 10 based on the provided market metrics. Use the following criteria to assign the score:

1. Content Quality
Clear Objectives: The website should clearly state the project's objectives and goals. It should explain what problem the project is solving and how it plans to achieve its goals.
Detailed Information: There should be detailed information about the project, including its technical aspects, use cases, and how it differentiates itself from competitors.
Whitepaper: A comprehensive whitepaper should be easily accessible and downloadable from the website. The whitepaper should be detailed and well-written, free from spelling and grammatical errors.
Blog/News Section: Regular updates through a blog or news section indicate ongoing development and transparency. Check the frequency and quality of the updates.
2. Team Information (VERY IMPORTANT, GIVE MORE IMPORTNCE THAN OTHERS WHILE CALCULATING FINAL SCORE)
Team Members: The website should have a dedicated section introducing the core team members, including their names, photos, roles, and professional backgrounds.
LinkedIn Profiles: Links to the LinkedIn profiles of the team members provide credibility and allow for further verification of their professional history.
Advisors and Partners: Information about advisors and strategic partners adds to the project's credibility.
3. Transparency and Legitimacy (VERY IMPORTANT, GIVE MORE IMPORTNCE THAN OTHERS WHILE CALCULATING FINAL SCORE)
Contact Information: There should be clear and accessible contact information, including an email address, physical address, and social media links.
Legal Information: Any legal disclaimers, terms of service, and privacy policies should be easily accessible and clearly stated.
Community Engagement: Links to active community channels (e.g., Telegram, Discord, Twitter, Reddit) show that the project engages with its community.
4. Technical and Security Features (VERY IMPORTANT, GIVE MORE IMPORTNCE THAN OTHERS WHILE CALCULATING FINAL SCORE)
Roadmap: A detailed roadmap showing past achievements and future plans. It should include timelines and milestones.
Tokenomics: Clear explanation of the tokenomics, including supply, distribution, and utility of the token.
Security Audits: Information about any security audits conducted on the project’s code, with links to audit reports if available.
Partnerships: Information about partnerships with reputable organizations adds credibility.
5. Red Flags
Spelling and Grammatical Errors: Numerous errors can indicate a lack of professionalism.
Vague Information: Lack of detailed information about the project, team, or roadmap can be a sign of a poorly developed or fraudulent project.
Overhyped Claims: Be wary of websites that make exaggerated claims about potential returns or use a lot of hype language without substantial backing.


For the Output produced I want you to Analyze the Website for {} and give me a rating out of 5 based on the factors and info above.
IMPORTANT: You are free to assign a higher weightage to more important factors. MAKE SURE TO INCLUDE STATISTICS, NUMBERS, DATA NAMES WHEN APPLICABLE
You should look at the investment from a long term perspective, an year or a few years in particular.

DO NOT USE UNNECESSARY WORDS

IT SHOULD BE IN THIS ORDER STRICTLY, AGAIN I SAY IT SHOULD GIVE ME THE OUTPUT IN THE FOLLOWING ORDER BELOW:
Overall score: total score/total = /5
Metric name score:
Metric component short summary:

Example:
Overall score: 3.7/5
1. Problem Statement: 7.5/10
Summary: Ethereum's ...

2. Execution: 8/10
Summary: Ethereum's ....
'''.format(crypto_name)
    extracted_content = ''
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a financial analyst and investment advisor, skilled in analyzing crpytocurrency data, formulating predicitions and providing investment advice."},
        {"role": "user", "content": query},
        {"role": "user", "content": extracted_content}
    ],
    max_tokens=2000, # Adjust based on how long you expect the answer to be
    temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
    )

    response = completion.choices[0]
    response = str(str(response))
    response = response[92:-58]
    response.replace(r"\\n", r"\n")

    num_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    del_text = ""
    modify_text = response.split(r"\n")

    for i in range(-1, -1*len(modify_text), -1):
        if modify_text[i] == '':
            continue
        if (modify_text[i][0] in num_string) or modify_text[i][0:7] == "Summary":
            break
        else:
            del_text = modify_text[i] + del_text
    response = response.replace(del_text, "")

    fraction = response.split(r"\n")[0].replace('Overall score: ', '').split('/')
    score = float(fraction[0]) / float(fraction[1])
    return response, score


# Function to fetch and analyze white paper
def analyze_white_paper_sync(crypto_name):
    query = '''
You are an AI that evaluates the investment potential of cryptocurrencies. Your goal is to provide a score out of 10 based on the provided market metrics. Use the following criteria to assign the score:

1. Problem Statement: What problem does the crypto project solve?
Understanding the problem the project addresses is crucial. This section usually appears in the introduction and helps you assess the project's market and target audience. Some projects provide data on market size, user base, and industry forecasts, while others focus on existing industry challenges and the necessity for a solution. Key elements to look for include:

Detailed description of the problem.
Market data and forecasts.
Identification of target user groups.
Explanation of why the problem is significant.

2. Project Mechanics: How does the project work?
This section outlines the technical aspects of the project, helping you understand its strengths, weaknesses, and how it stands against competitors. Points to consider include:

Consensus Mechanism: Proof of Work (PoW), Proof of Stake (PoS), etc.
Node Criteria: Requirements for running network nodes.
Fees: Transaction fees and other costs associated with using the platform.
Use Cases and dApps: Decentralized applications and real-world applications.
Scalability Solutions: Layer 1 vs. Layer 2 solutions.
For instance, if the project is a DeFi platform, look for detailed information on:

Automated Market Makers (AMM)
Liquidity pools and their management.

3. Tokenomics: Where does the value of the cryptocurrency lie?
Tokenomics is critical for understanding the economic model of the project. This section should provide answers to:

Type of Token: Coin or token being issued.
Supply Limits: Whether the token is inflationary or deflationary.
Distribution: Allocation for institutional and retail investors.
Utility: Actual use cases and utility once the project is live.
Lockup Periods: Vesting schedules and lockup periods for tokens.
Exchange Listings: Potential or confirmed listings on crypto exchanges.

4. Facts vs. Opinions: How reliable is the information?
A well-founded whitepaper should base its claims on data and research rather than opinions. Assess the credibility of the information by:

Verifying market size and industry data through external sources.
Noting phrases like "we think" or "in our opinion" and treating them with caution.
Checking for references to scientific research or case studies supporting their claims.

5. Level of Detail: Is the information vague or detailed?
The clarity and depth of information are crucial for trustworthiness. Look for:

Detailed explanations of technical terms and concepts.
Use of visual aids like charts, graphs, and equations.
Comprehensive coverage of key points without being overly vague or brief.
Clarity on the execution plan and milestones.

For the Output produced I want you to Analyze the White paper for {} and give me a rating out of 15 based on the factors and info above.
IMPORTANT: You are free to assign a higher weightage to more important factors. MAKE SURE TO INCLUDE STATISTICS, NUMBERS, DATA NAMES WHEN APPLICABLE
You should look at the investment from a long term perspective, an year or a few years in particular.

IT SHOULD BE IN THIS ORDER STRICTLY, AGAIN I SAY IT SHOULD GIVE ME THE OUTPUT IN THE FOLLOWING ORDER BELOW, I DONT WANT ANY UNNECESSARY WORDS:
Overall score: total score/total = /15
Metric name score:
Metric component short summary:

Example:
Overall score: 12.4/15
1. Problem Statement: 7.5/10
Summary: Ethereum's whitepaper clearly outlines the problem ...

2. Execution: 8/10
Summary: Ethereum's ....
'''.format(crypto_name)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a financial analyst and investment advisor, skilled in analyzing crpytocurrency data, formulating predicitions and providing investment advice."},
        {"role": "user", "content": query},
    ],
    max_tokens=2000, # Adjust based on how long you expect the answer to be
    temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
    )

    response = completion.choices[0]
    response = str(str(response))
    response = response[92:-58]
    response.replace(r"\\n", r"\n")
    num_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    del_text = ""
    modify_text = response.split(r"\n")

    for i in range(-1, -1*len(modify_text), -1):
        if modify_text[i] == '':
            continue
        if (modify_text[i][0] in num_string) or modify_text[i][0:7] == "Summary":
            break
        else:
            del_text = modify_text[i] + del_text
    response = response.replace(del_text, "")

    fraction = response.split(r"\n")[0].replace('Overall score: ', '').split('/')
    score = float(fraction[0]) / float(fraction[1])
    return response, score

def escape_dollar_signs(text):
    return re.sub(r'\$', r'\\$', text)

# Function to split text into chunks within token limit
def split_text(text, max_length=510):
    tokens = tokenizer.tokenize(text)
    for i in range(0, len(tokens), max_length):
        yield tokenizer.convert_tokens_to_string(tokens[i:i + max_length])

# Function to filter relevant sentences based on keywords and semantic similarity
def filter_relevant_sentences(text, keyword, ref_list, threshold=0.5):
    sentences = nltk.sent_tokenize(text)
    relevant_sentences = []

    # Get embedding for the keyword for semantic similarity
    keyword_embedding = semantic_model.encode(keyword, convert_to_tensor=True)

    for sentence in sentences:
        # If keyword is in sentence, consider it relevant
        if any(keyword.lower() in sentence.lower() for keyword in ref_list):
            relevant_sentences.append(sentence)
        else:
            # Otherwise, check semantic similarity
            sentence_embedding = semantic_model.encode(sentence, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(keyword_embedding, sentence_embedding).item()
            if similarity_score > threshold:
                relevant_sentences.append(sentence)

    return " ".join(relevant_sentences)

# Function to fetch news and perform sentiment analysis
async def analyze_sentiment(crypto_name, crypto_ticker, api_key):
    NEWSAPI_URL = "https://newsapi.org/v2/everything"

    params = {
        "q": crypto_name,
        "language": "en",  # Filter for English articles
        "sortBy": "relevancy and publishedAt",
        "apiKey": api_key
    }
    async with aiohttp.ClientSession() as session:
        news_data = await fetch_url(session, NEWSAPI_URL, params=params)
    
    # Check if 'articles' key is in the response
    if 'articles' not in news_data:
        print(f"Error: 'articles' key not found in the response. Response: {news_data}")
        return pd.DataFrame(), pd.Series()  # Return empty DataFrame and Series
    
    analyzer = SentimentIntensityAnalyzer()
    articles = news_data['articles']
    sentiments = []
    dates = []
    headlines = []
    links = []
    #more_articles = news.get_yf_rss(crypto_ticker)

    def process_article(article):
        try:
            news_article = Article(article['url'])
            news_article.download()
            news_article.parse()
            text = news_article.text
        except:
            text = article.get('content') or article['title'] + " " + article['description']

        filtered_text = filter_relevant_sentences(text, crypto_name, relevant_keywords)
        chunk_scores = []
        for chunk in split_text(filtered_text, max_length=510):
            sentiment_result = sentiment_analysis(chunk)[0]
            label = sentiment_result['label']
            score = sentiment_result['score']

            if label == 'positive':
                chunk_scores.append(score)
            elif label == 'negative':
                chunk_scores.append(-score)
            else:
                score = analyzer.polarity_scores(chunk)['compound']
                chunk_scores.append(score)

        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0

        sentiments.append(avg_score)
        dates.append(datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d'))
        headlines.append(escape_dollar_signs(article['title'])) # Escape dollar signs in headlines
        links.append(article['url']) 

        return {
            'Date': datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d'),
            'Headline': escape_dollar_signs(article['title']),
            'Link': article['url'],
            'Sentiment Score': avg_score
        }
        
    articles_rdd = spark.sparkContext.parallelize(articles)
    print(articles_rdd)
    sentiment_data = articles_rdd.map(process_article).collect()
    print(sentiment_data)
    
    # Create a dataframe from the sentiments list
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Calculate daily cumulative sentiment
    sentiment_df['Positive'] = sentiment_df['Sentiment Score'].apply(lambda x: x if x >= 0 else 0)
    sentiment_pos = sentiment_df['Positive'].sum()
    sentiment_df['Negative'] = sentiment_df['Sentiment Score'].apply(lambda x: x if x < 0 else 0)
    sentiment_neg = sentiment_df['Negative'].sum()
    daily_sentiment = sentiment_df.groupby('Date').agg({'Positive': 'sum', 'Negative': 'sum'}).reset_index()

    # Plotting the bar graph
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(daily_sentiment['Date'], daily_sentiment['Positive'], color='green', label='Positive')
    ax.bar(daily_sentiment['Date'], daily_sentiment['Negative'], color='red', label='Negative')
    
    ax.set_title(f'Sentiment Analysis for {crypto_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    plt.xticks(rotation=45)
    #plt.show()
    plt.savefig(f'{crypto_name}_sentiment_graph_{datetime.today().date()}.png')

    sentiment_df = sentiment_df.drop(columns=['Positive', 'Negative'])
    sentiment_df = sentiment_df.sort_values(by='Date', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))  # Set size frame
    ax.axis('tight')
    ax.axis('off')
    tbl = table(ax, sentiment_df.tail(10), loc='center', cellLoc='center', colWidths=[0.15]*len(sentiment_df.columns))  # Create a table plot
    # Set the font size for the table
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)  # Adjust font size as needed
    # Adjust column widths and table properties for better readability
    tbl.auto_set_column_width(col=list(range(len(sentiment_df.columns))))
    plt.savefig(f'{crypto_name}_sentiment_df_{datetime.today().date()}.png', bbox_inches='tight', pad_inches=0.1, dpi = 500)

    final_score = sentiment_df['Sentiment Score'].sum()

    return sentiment_df, daily_sentiment, final_score

def evaluate_sentiment(sentiment_df, daily_sentiment, final_score):
    query = '''
You are an AI that evaluates the investment potential of cryptocurrencies. Your goal is to provide an Overall score out of 15 based on the provided sentiment data and individual score of 10. Use the following criteria to assign the score:

Factors for Evaluation
1. Sentiment Score Distribution:

Analyze the distribution of sentiment scores over time to identify trends.
Positive sentiment scores indicate positive news, while negative scores indicate negative news.
Consistent positive sentiment over time is a good indicator of long-term potential.

2. Frequency of Positive vs. Negative News:

Compare the number of positive news articles to negative ones.
A higher frequency of positive news articles suggests strong market confidence and interest.

3. Impact of Key Events:

Identify significant events mentioned in the news articles (e.g., partnerships, regulatory changes, technological advancements).
Evaluate the impact of these events on sentiment scores and their potential long-term effects.

4. Market Demand and Adoption:

Assess news articles that mention market demand and adoption rates.
Positive sentiment related to adoption and usage indicates growing interest and potential for increased value.

5. Technological Innovation:

Evaluate news articles that discuss technological advancements and innovations.
Positive sentiment regarding technology improvements suggests a strong foundation for future growth.
6. Regulatory Environment:

Analyze news articles discussing regulatory changes and compliance.
Positive sentiment around regulatory news indicates a stable and supportive environment for long-term investment.

7. Community and Ecosystem Support:

Assess the sentiment around community and ecosystem support.
Strong community backing and positive sentiment from the ecosystem indicate robust support and long-term sustainability.

For the Output produced I want you to Analyze the Sentiment Data for {} and give me a rating out of 15 based on the factors and info above.
IMPORTANT: You are free to assign a higher weightage to more important factors. MAKE SURE TO INCLUDE STATISTICS, NUMBERS, DATA NAMES WHEN APPLICABLE
You should look at the investment from a long term perspective, an year or a few years in particular.

I want a summary for each metric evaluated. 

IT SHOULD BE IN THIS ORDER STRICTLY, AGAIN I SAY IT SHOULD GIVE ME THE OUTPUT IN THE FOLLOWING ORDER BELOW:
Overall score: total score/total = /15
Metric name score:
Metric component short summary:

Example:
Overall score: 10.1/15
1. Sentiment score distribution: 7.5/10
Summary: Ethereum's...

2. Execution: 8/10
Summary: Ethereum's
'''.format(crypto_name)
    extracted_content = f'''Here are is the sentiment dataframe extracted with date, article link, headline and score: {sentiment_df}
Here are is the sentiment graph with x axis representing the date and y axis representing the sentiment score I calculated on the news headline: {daily_sentiment}
Here is the final overall score over the 100 articles I extracted over the past month {final_score}
'''
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a financial analyst and investment advisor, skilled in analyzing crpytocurrency data, formulating predicitions and providing investment advice."},
        {"role": "user", "content": query},
        {"role": "user", "content": extracted_content}
    ],
    max_tokens=4000, # Adjust based on how long you expect the answer to be
    temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
    )

    response = completion.choices[0]
    response = str(str(response))
    response = response[92:-58]
    response.replace(r"\\n", r"\n")

    num_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    del_text = ""
    modify_text = response.split(r"\n")

    for i in range(-1, -1*len(modify_text), -1):
        if modify_text[i] == '':
            continue
        if (modify_text[i][0] in num_string) or modify_text[i][0:7] == "Summary":
            break
        else:
            del_text = modify_text[i] + del_text

    response = response.replace(del_text, "")

    fraction = response.split(r"\n")[0].replace('Overall score: ', '').split('/')
    score = float(fraction[0]) / float(fraction[1])
    return response, score

# Function to fetch market metrics from CoinGecko
async def fetch_market_metrics(crypto_id, crypto_ticker):
    base_url = 'https://pro-api.coinmarketcap.com/v1'
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
    }

    params = {'symbol': crypto_id, 'convert': 'USD'}

    async with aiohttp.ClientSession() as session:
        response = await fetch_url(session, f'{base_url}/cryptocurrency/quotes/latest', params=params, headers=headers)
        market_data = response['data'][crypto_id]

    
    crypto = yfinance.Ticker(crypto_ticker)
    price_hist = yfinance.download(crypto_ticker)
    
    fully_diluted_market_cap = market_data['quote']['USD'].get('fully_diluted_market_cap')
    total_annualized_revenue = market_data['quote']['USD'].get('volume_24h') * 365
    total_locked_value = market_data.get('total_supply')
    transaction_volume_24h = market_data['quote']['USD'].get('volume_24h')
    percent_change_1h = market_data['quote']['USD'].get('percent_change_1h')
    percent_change_24h = market_data['quote']['USD'].get('percent_change_24h')
    percent_change_7d = market_data['quote']['USD'].get('percent_change_7d')
    percent_change_30d = market_data['quote']['USD'].get('percent_change_30d')
    percent_change_60d = market_data['quote']['USD'].get('percent_change_60d')
    percent_change_90d = market_data['quote']['USD'].get('percent_change_90d')
    market_cap = market_data['quote']['USD'].get('market_cap')
    annualized_transaction_volume = transaction_volume_24h * 365
    market_cap_dominance = market_data['quote']['USD'].get('market_cap_dominance')
    cmc_rank = market_data.get('cmc_rank')
    tags = market_data.get('tags')
    volume_change_24h = market_data['quote']['USD'].get('volume_change_24h')
    active_status = market_data.get('is_active') 
    inf_supply = market_data.get('infinite_supply')
    fiat_check =  market_data.get('is_fiat')
    last_updated = market_data.get('last_updated')

    price_hist = yfinance.download(crypto_ticker)
    
    ema12=ta.EMA(price_hist["Adj Close"], timeperiod=12)
    ema50=ta.EMA(price_hist["Adj Close"], timeperiod=50)
    sma20=ta.SMA(price_hist["Adj Close"], timeperiod=20)
    sma200=ta.SMA(price_hist["Adj Close"], timeperiod=200)
    ma = pd.concat([ema12, ema50, sma20, sma200],axis=1)

    rsi = ta.RSI(price_hist["Adj Close"], timeperiod=14)

    macd_val, macdsignal, macdhist = ta.MACD(price_hist['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = pd.concat([macd_val,macdsignal, macdhist],axis=1)

    return {
        'market_cap': crypto.info.get('marketCap'),
        'trading_volume': crypto.info.get('volume'),
        'circulating_supply': crypto.info.get('circulatingSupply'),
        'total_supply': total_locked_value,
        'volume_change_24h': volume_change_24h,
        'price_change_1h':percent_change_1h,
        'price_change_24h': percent_change_24h,
        'price_change_7d': percent_change_7d,  
        'price_change_30d': percent_change_30d,
        'price_change_60d': percent_change_60d,
        'price_change_90d': percent_change_90d,
        'ath': crypto.info.get('regularMarketDayHigh'),
        'atl': crypto.info.get('regularMarketDayLow'),
        'price_to_sales_ratio': fully_diluted_market_cap / total_annualized_revenue,
        'market_cap_to_tvl' : market_cap / total_locked_value,
        'nvt_ratio': market_cap / transaction_volume_24h,
        'cmc_rank': cmc_rank,
        'market_cap_dominance': market_cap_dominance,
        'annualized_transaction_volume': annualized_transaction_volume,
        'tags': tags,
        'active_status': active_status,
        'inf_supply': inf_supply,
        'fiat_check': fiat_check,
        'last_updated': last_updated,
        'price_hist': price_hist["Adj Close"],
        'ma': ma,
        'rsi': rsi,
        'macd': macd
    }

def evaluate_metrics(metrics_dict):
    query = '''
You are an AI that evaluates the investment potential of cryptocurrencies. Your goal is to provide a score out of 10 based on the provided market metrics. Use the following criteria to assign the score:

MARKET AND COIN ANALYSIS:
1. Market Cap: Indicates the overall size and value of the cryptocurrency. A higher market cap typically means more stability and lower risk.
2. Trading Volume: Reflects the liquidity of the cryptocurrency. Higher trading volume suggests easier buying and selling, reducing the risk of price manipulation.
3. Circulating Supply: The number of coins currently available in the market. It's crucial for understanding the inflation potential and scarcity of the asset.
4. Total Supply: The total number of coins that will ever exist. Helps assess the maximum potential dilution.
5. Volume Change (24h): Indicates short-term volatility. Significant changes can suggest instability or upcoming major price movements.
6. Price Changes (1h, 24h, 7d, 30d, 60d, 90d): These metrics show how the price has changed over various time frames, providing insights into the asset's volatility and price trends.
7. Price to Sales Ratio: A lower ratio generally indicates that the asset is undervalued relative to its sales.
8. Market Cap to TVL Ratio: A lower ratio suggests that the asset might be undervalued compared to the value locked in its network.
9. NVT Ratio: Network Value to Transactions ratio. A lower NVT indicates that the asset is undervalued relative to its transaction volume.
10. Market Cap Dominance: Reflects the asset's share of the total cryptocurrency market. Higher dominance indicates more confidence in the asset.
11. CMC Rank: Lower ranks indicate more established and popular cryptocurrencies.
12. Active Status: Whether the cryptocurrency is actively traded. Active coins are generally better as they indicate ongoing interest and development.
13. Infinite Supply: Cryptocurrencies with a finite supply are generally preferred as they are less prone to inflation.
14. Fiat Check: Should not be a fiat currency. Cryptocurrencies are valued for their decentralized nature.
15. Tags: Additional information about the cryptocurrency's use case, technology, and community support.

PRICE AND TECHNICAL ANALYSIS:
1. Historical Price Data: Historical price movements and trends over various time frames to identify long-term trends (uptrend, downtrend, sideways). Look for periods of consistent price increases or decreases.
2. Moving Averages: Simple moving average (SMA) and exponential moving average (EMA) for various periods to smooth out short-term fluctuations and highlight longer-term trends.
3. Relative Strength Index (RSI): An indicator of whether the cryptocurrency is overbought or oversold.
4. MACD (Moving Average Convergence Divergence): A trend-following momentum indicator.

For the Output produced I want you to Analyze the Metrics for {} and give me a rating out of 50 based on the factors and info above. Market Analysis must be rated out of 30 and Technical Analysis must be rated out of 20. Individually each metric must be rated out of 10.
IMPORTANT: You are free to assign a higher weightage to more important factors. MAKE SURE TO INCLUDE STATISTICS, NUMBERS, DATA, AND NAMES.
You should look at the investment from a long term perspective, an year or a few years in particular.

I want a long detailed summary for each metric evaluated in minimum 3 detailed lines. 
MAKE SURE TO INCLUDE STATISTICS, NUMBERS, DATA, AND NAMES,

Dont provide a concluding statement at the very end after all the metric analysis. Avoid the Overall, In conclusion line at the very end.

Make sure to explicitly mention the word Summary: before you start talking about a metric
IT SHOULD BE IN THIS ORDER STRICTLY, AGAIN I SAY IT SHOULD GIVE ME THE OUTPUT IN THE FOLLOWING ORDER BELOW:
THE OVERALL SCORE MUST BE THE FIRST THING IN THE OUTPUT, FOLLOWED BY MARKET AND COIN ANANLYSIS AND THEN PRICE AND TECHNICAL ANALYSIS ....

Overall score: 
Metric name score:
Metric component detailed brief summary:

Example:
Overall score: 40/50

Market and Coin Analysis: 27/30

1. Core functionality: 7.5/10
Summary: Ethereum's ...

2. Token utility: 8/10
Summary: Ethereum's 

.....

Price and Technical Analysis: 15/20

1. Historical Price Data: 9/10
Summary: ....
'''.format(crypto_name)
    extracted_content = f'Here are the metrics extracted: {metrics_dict}'
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a financial analyst and investment advisor, skilled in analyzing crpytocurrency data, formulating predicitions and providing investment advice."},
        {"role": "user", "content": query},
        {"role": "user", "content": extracted_content}
    ],
    max_tokens=4000, # Adjust based on how long you expect the answer to be
    temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
    )

    response = completion.choices[0]
    response = str(str(response))
    response = response[92:-58]

    response.replace(r"\\n", r"\n")
    response = response.replace(r"*", "")
    response = response.replace("Overall Score", "Overall score")
    modify_text = response.split(r"\n")
    
    num_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    del_text = ""
    modify_text = response.split(r"\n")

    for i in range(-1, -1*len(modify_text), -1):
        if modify_text[i] == '':
            continue
        if (modify_text[i][0] in num_string) or modify_text[i][0:7] == "Summary":
            break
        else:
            del_text = modify_text[i] + del_text
    response = response.replace(del_text, "")

    fraction = response.split(r"\n")[0].lower().replace('overall score: ', '').replace('*', '').split('/')
    score = float(fraction[0]) / float(fraction[1])

    return response, score

# Function to determine utility
def check_utility_sync(crypto_name):
    query = '''
You are an AI that evaluates the investment potential of cryptocurrencies. Your goal is to provide a score out of 10 based on the provided market metrics. Use the following criteria to assign the score:

1. Core Functionality and Use Case
Primary Purpose: Identify the primary purpose of the cryptocurrency. Is it designed for payments, smart contracts, decentralized applications (dApps), or another specific use case?
Problem Solving: Determine whether the cryptocurrency addresses a specific problem or enhances the functionality of the blockchain ecosystem. Examples include improving scalability, privacy, or security.
2. Ecosystem Integration
Adoption by dApps and Services: Check if the cryptocurrency is widely used by decentralized applications and services. For instance, Ethereum is integral to many dApps and DeFi platforms.
Interoperability: Assess whether the cryptocurrency can interact with other blockchain networks or systems, enhancing its utility.
3. Token Utility
Utility Tokens: Look for cryptocurrencies that serve a practical purpose within a specific platform. For example, Basic Attention Token (BAT) is used in the Brave browser ecosystem for advertising and rewards.
Governance Tokens: Determine if the token provides holders with voting rights or other forms of governance within the network.
4. Real-World Applications
Practical Use Cases: Investigate if the cryptocurrency has real-world applications, such as enabling cross-border payments, facilitating decentralized finance, or supporting digital identity systems.
Business Partnerships: Check for partnerships with established businesses or integration into existing business processes, which can enhance the cryptocurrency's utility and adoption.
5. Technological Innovation
Unique Technology: Evaluate the technological innovations introduced by the cryptocurrency. Does it offer something new or significantly improved compared to existing solutions?
Development Activity: Assess the level of ongoing development and innovation. Active development communities and regular updates indicate a commitment to improving utility.
6. Market Demand and Adoption
User Base: Determine the size and growth rate of the user base. Higher adoption rates often correlate with greater utility.
Network Effects: Consider the network effects, where the value of the cryptocurrency increases as more people use it. Bitcoin and Ethereum benefit significantly from network effects.
7. Economic Incentives
Incentive Structures: Analyze the economic incentives for various participants within the ecosystem. This includes miners, validators, developers, and users.
Tokenomics: Understand the supply dynamics, inflation rates, and mechanisms for token burning or rewards, which can affect the utility and value of the cryptocurrency.
8. Security and Reliability
Security Measures: Evaluate the security protocols in place to protect the network and its users. A secure network is more likely to maintain its utility.
Network Reliability: Consider the reliability and stability of the network. High uptime and consistent performance are critical for maintaining utility.
9. Regulatory Compliance
Compliance with Laws: Ensure the cryptocurrency complies with relevant regulations, which can impact its utility and acceptance in various jurisdictions.
Transparency and Governance: Transparent governance and clear regulatory frameworks can enhance the trust and utility of the cryptocurrency.
10. Community and Ecosystem Support
Developer Community: A strong developer community contributes to the ongoing improvement and expansion of the cryptocurrency's utility.
User Community: An active and engaged user community can drive adoption and innovation, further enhancing utility.

For the Output produced I want you to Analyze the Utility and Use Case for {} and give me a rating out of 15 based on the factors and info above.
IMPORTANT: You are free to assign a higher weightage to more important factors. MAKE SURE TO INCLUDE STATISTICS, NUMBERS, DATA NAMES WHEN APPLICABLE
You should look at the investment from a long term perspective, an year or a few years in particular.

IT SHOULD BE IN THIS ORDER STRICTLY, AGAIN I SAY IT SHOULD GIVE ME THE OUTPUT IN THE FOLLOWING ORDER BELOW:
Overall score: total score/total = /15
Metric name score:
Metric component detailed brief summary:

Example:
Overall score: 12.3/15
1. Core functionality: 7.5/10
Summary: Ethereum's ...

2. Token utility: 8/10
Summary: Ethereum's ....
'''.format(crypto_name)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a financial analyst and investment advisor, skilled in analyzing crpytocurrency data, formulating predicitions and providing investment advice."},
        {"role": "user", "content": query},
    ],
    max_tokens=2000, # Adjust based on how long you expect the answer to be
    temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
    )

    response = completion.choices[0]
    response = str(str(response))
    response = response[92:-58]
    response.replace(r"\\n", r"\n")

    num_string = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    del_text = ""
    modify_text = response.split(r"\n")

    for i in range(-1, -1*len(modify_text), -1):
        if modify_text[i] == '':
            continue
        if (modify_text[i][0] in num_string) or modify_text[i][0:7] == "Summary":
            #print(modify_text[i])
            break
        else:
            del_text = modify_text[i] + del_text
    response = response.replace(del_text, "")

    fraction = response.split(r"\n")[0].replace('Overall score: ', '').split('/')
    score = float(fraction[0]) / float(fraction[1])
    return response, score

class PDF(FPDF):
    def header(self, crypto_name):
        self.set_font("Helvetica", 'B', 12)
        self.cell(0, 10, f"{crypto_name} Investment Analysis", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

    def chapter_top(self, top):
        self.set_font("Helvetica", 'B', 16)
        self.multi_cell(0, 12, top)
        self.ln(5)
        
    def chapter_header(self, header):
        self.set_font("Helvetica", 'B', 12)
        self.multi_cell(0, 8, header)
        self.ln(3)

    def chapter_title(self, title):
        self.set_font("Helvetica", 'B', 10)
        self.multi_cell(0, 8, title)
        self.ln(1)

    def chapter_body(self, body):
        self.set_font("Helvetica", '', 10)
        self.multi_cell(0, 8, body)
        self.ln(5)

# Function to add formatted text to PDF
def add_text_to_pdf(pdf, text):
    text = text.replace(r"\\n", r"\n")
    lines = text.split(r'\n')

    for line in lines:
        if line.strip() == '':
            continue
        if line.startswith('Final Score'):
            pdf.chapter_top(line)
        elif line.startswith('Overall score:') or line.startswith("Market Metric") or line.startswith("Sentiment Analysis") or line.startswith("Utility and Use Case") or line.startswith("White Paper Review") or line.startswith("Website Analysis") or line.startswith("Market and Coin Analysis") or line.startswith("Price and Technical Analysis"):
            pdf.chapter_header(line)
        elif line[0].isdigit():
            pdf.chapter_title(line)
        elif line.startswith('Summary:'):
            pdf.chapter_body(line)
        else:
            line = line.replace('\xa0', ' ')  # Replace non-breaking spaces
            wrapped_lines = textwrap.wrap(line, width=100)  # Adjust width as needed
            for wrapped_line in wrapped_lines:
                print(f"Adding wrapped line to PDF: '{wrapped_line}'")  # Debugging print statement
                pdf.multi_cell(0, 10, wrapped_line)

def generate_pdf(final_score, market_metrics_check, sentiment_check, average_sentiment, sentiment_df, utility_check, white_paper_check, website_check, crypto_name):
    # Create PDF
    pdf = PDF()
    pdf.add_page()
    
    # Add text to PDF
    add_text_to_pdf(pdf, f"Final Score For {crypto_name}: " + str(round(final_score, 1)) + "/100")

    add_text_to_pdf(pdf, "Market Metrics")
    add_text_to_pdf(pdf, market_metrics_check)

    add_text_to_pdf(pdf, "Sentiment Analysis") 
    add_text_to_pdf(pdf, sentiment_check)

    sentiment_graph_path = f'{crypto_name}_sentiment_graph_{datetime.today().date()}.png'
    sentiment_df_path = f'{crypto_name}_sentiment_df_{datetime.today().date()}.png'
    pdf.add_page()
    pdf.image(sentiment_graph_path, x=10, y=10, w=pdf.w - 20)
    pdf.add_page()
    pdf.image(sentiment_df_path, x=10, y=10, w=pdf.w - 20)
    pdf.add_page()

    add_text_to_pdf(pdf, "Utility and Use Case")
    add_text_to_pdf(pdf, utility_check)

    add_text_to_pdf(pdf, "White Paper Review")
    add_text_to_pdf(pdf, white_paper_check)

    add_text_to_pdf(pdf, "Website Analysis")
    add_text_to_pdf(pdf, website_check)

    # Save the PDF to a file
    pdf.output(f"{crypto_name} Investment Analysis {datetime.today().date()}.pdf")

    print("PDF created successfully.")
    
# Main function to get the financial analysis and recommendation
async def evaluate_crypto(crypto_id, crypto_name, crypto_ticker):
    loop = asyncio.get_event_loop()

    website_check, site_score = await loop.run_in_executor(None, check_website_sync, crypto_name) # 5
    white_paper_check, paper_score = await loop.run_in_executor(None, analyze_white_paper_sync, crypto_name) # 15
    utility_check,util_score = await loop.run_in_executor(None, check_utility_sync, crypto_name) # 15

    market_metrics = await fetch_market_metrics(crypto_id, crypto_ticker) 
    market_metrics_check, market_score = evaluate_metrics(market_metrics) # 50

    sentiment_df, average_sentiment, sentiment_sum = await analyze_sentiment(crypto_name, crypto_ticker, NEWSAPI_API_KEY) # 15
    sentiment_check, sentiment_score = evaluate_sentiment(sentiment_df, average_sentiment, sentiment_sum)


    final_score = site_score * 5 + paper_score * 15 + market_score * 50 + sentiment_score * 15 + util_score * 15

    generate_pdf(final_score, market_metrics_check, sentiment_check, average_sentiment, sentiment_df, utility_check, white_paper_check, website_check, crypto_name)

# Run the main function
if __name__ == "__main__":
    client = OpenAI(api_key=OPENAI_API_KEY)
    crypto_ticker = 'ETH-USD'
    crypto = yfinance.Ticker(crypto_ticker)
    crypto_name = crypto.get_info()['name']
    crypto_id = 'ETH'
    created = 0
    asyncio.run(evaluate_crypto(crypto_id, crypto_name, crypto_ticker))
    print("Completed Successfully!")
    created = 1
