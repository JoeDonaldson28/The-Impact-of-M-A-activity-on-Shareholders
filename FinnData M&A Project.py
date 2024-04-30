# Personal Project
# Step 0
# In this project, my topic question will be: Whether M&A
# activity creates or destroys shareholder value in the 
# oil and gas industry between the years 2010-2020
# I will be looking at the cumulative abnormal returns
# around the companies within this industry

# Step 1 - Downloading data from WRDS and Yahoo Finance
# In this step, I will be downloading M&A data from Thomson
# Reuters on WRDS and stock price data before, during, and 
# after M&A activity for oil and gas companies with YFinance

#%% Step 1 - Connect to WRDS and download packages
import wrds
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

# Connect to WRDS (ensure you have valid credentials)
db = wrds.Connection()

#%% Step 2 - SQL query to retrieve M&A data with U.S.-based acquirers and targets
# SQL query to retrieve M&A data with U.S.-based public acquirers and targets
mergers_query = """
SELECT 
    amanames AS acquirer_name,
    tmanames AS target_name,
    dateann AS announcement_date,
    atf_mid_desc AS industry,
    deal_value,
    entval,  
    a_postmerge_own_pct,
    pct_stk,  
    apublic,
    tticker,
    attitude,
    amv,
    pm4wk
FROM 
    sdc.wrds_ma_details
WHERE 
    dateann BETWEEN '2014-01-01' AND '2024-01-31'
    AND anation = 'United States'
    AND tnation = 'United States'
    AND apublic = 'Public'
"""

# Execute the query to retrieve M&A data
m_a_data = db.raw_sql(mergers_query)

# Delete rows with industry 'REITs' from the m_a_data DataFrame
m_a_data = m_a_data[m_a_data['industry'] != 'REITs']


# Check if the data retrieval is successful
if m_a_data.empty:
    print("M&A data is empty. Check your SQL query and data source.")
else:
    print("Data retrieval successful. Displaying the first few rows:")
    print(m_a_data.head())


#%% Step 2 - Data Preprocessing
# Handle missing values in deal_value, if any, by setting to 0
m_a_data['deal_value'].fillna(0, inplace=True)

# Convert announcement dates to a datetime format for further analysis
m_a_data['announcement_date'] = pd.to_datetime(m_a_data['announcement_date'])

# List of required columns
required_columns = ['deal_value', 'entval', 'a_postmerge_own_pct', 'pct_stk']

# Drop rows where any of these columns have NaN values
m_a_data.dropna(subset=required_columns, inplace=True)

# Display the resulting dataset to ensure only valid rows are present
print("Cleaned data with no missing values:")
print(m_a_data.head())


# %% Step 3 - Extract Data
# Extract unique stock tickers for acquirers from the M&A data
acquirer_tickers = m_a_data['tticker'].unique()  # Ensure this represents acquirers' tickers

print("Acquirers' stock tickers:", acquirer_tickers)


# %% Step 4 - Stock Prices for M&A
import yfinance as yf
import pandas as pd

# Function to fetch stock prices for acquirers within a specified date range
def fetch_acquirer_stock_prices(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise Exception(f"{ticker}: No stock price data found, symbol may be delisted")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame for invalid tickers

# Define the date range for stock price data
start_date = '2013-09-01'
end_date = '2024-03-31'

# Fetch stock prices for acquirer tickers only
acquirer_stock_prices = {
    ticker: fetch_acquirer_stock_prices(ticker, start_date, end_date) for ticker in acquirer_tickers
}

# Check for empty DataFrames and identify tickers with no valid data
empty_tickers = [ticker for ticker, data in acquirer_stock_prices.items() if data.empty]
if empty_tickers:
    print("These acquirers might be delisted or invalid:", empty_tickers)

#%% 
# Identify tickers with empty DataFrames
empty_tickers = [ticker for ticker, data in acquirer_stock_prices.items() if data.empty]
if empty_tickers:
    print("These acquirers might be delisted or invalid:", empty_tickers)

#%% 
# Remove delisted or invalid tickers from the list of acquirer tickers
valid_tickers = [ticker for ticker in acquirer_tickers if ticker not in empty_tickers]

# Rebuild the stock price dataset with only valid tickers
acquirer_stock_prices = {
    ticker: acquirer_stock_prices[ticker] for ticker in valid_tickers
}

print("Valid acquirers:", valid_tickers)



# %% Step 5 - Calculate CARs
# Extract the unique stock tickers for acquirers
acquirer_tickers = m_a_data['tticker'].unique()

# Create a dictionary to store the announcement dates for each acquirer
announcement_dates = {}

# Fill the dictionary with announcement dates for each unique ticker
for _, row in m_a_data.iterrows():
    acquirer_ticker = row['tticker']  # Get the stock ticker
    announcement_date = row['announcement_date']  # Get the announcement date
    announcement_dates[acquirer_ticker] = announcement_date

# Print the dictionary to see the announcement dates for each acquirer
print("Announcement dates for each acquirer:")
for ticker, date in announcement_dates.items():
    print(f"{ticker}: {date}")

# %%
# Create a new dictionary with formatted announcement dates
formatted_announcement_dates = {}

# Loop through each entry in the original dictionary and format the date
for ticker, date in announcement_dates.items():
    formatted_announcement_dates[ticker] = date.strftime('%Y-%m-%d')  # Format to 'YYYY-MM-DD'

# Print the formatted announcement dates
print("Formatted announcement dates for each acquirer:")
for ticker, formatted_date in formatted_announcement_dates.items():
    print(f"{ticker}: {formatted_date}")

# %%
import pandas as pd
import datetime as dt

# Function to convert string dates to datetime objects
def convert_to_datetime(date_str):
    return pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")

# Convert announcement dates to datetime
formatted_announcement_dates = {k: convert_to_datetime(v) for k, v in announcement_dates.items()}

# Function to filter stock price data for 60 days after an announcement date
def filter_60_days(stock_data, announcement_date):
    if pd.isnull(announcement_date):
        return pd.DataFrame()  # Return empty DataFrame if announcement_date is null

    # Calculate the end date, 60 days from the announcement date
    end_date = announcement_date + dt.timedelta(days=60)

    # Filter the stock data for this 60-day range
    return stock_data[(stock_data.index >= announcement_date) & (stock_data.index <= end_date)]

# Dictionary to hold the filtered stock data for 60 days after announcement
filtered_acquirer_data = {}

# Iterate through each acquirer and filter their stock data
for ticker, stock_data in acquirer_stock_prices.items():
    if not stock_data.empty:
        # Get the announcement date for this ticker
        announcement_date = formatted_announcement_dates[ticker]
        # Filter the stock data for 60 days following the announcement
        filtered_acquirer_data[ticker] = filter_60_days(stock_data, announcement_date)

#%% Function to calculate returns from the filtered data using .loc to avoid SettingWithCopyWarning
def calculate_returns(stock_data):
    # Create a deep copy to work with
    stock_data_copy = stock_data.copy()
    
    # Calculate daily returns
    stock_data_copy.loc[:, "daily_returns"] = stock_data_copy["Adj Close"].pct_change()
    
    # Calculate cumulative returns
    stock_data_copy.loc[:, "cumulative_returns"] = (1 + stock_data_copy["daily_returns"]).cumprod() - 1
    
    return stock_data_copy

# Apply the returns calculation to each acquirer using .loc
filtered_acquirer_data = {}
for ticker, stock_data in acquirer_stock_prices.items():
    if not stock_data.empty:
        # Get the announcement date for this ticker
        announcement_date = formatted_announcement_dates[ticker]
        
        # Filter the stock data for 60 days following the announcement
        filtered_data = filter_60_days(stock_data, announcement_date)
        
        # Calculate returns using a deep copy to avoid SettingWithCopyWarning
        filtered_acquirer_data[ticker] = calculate_returns(filtered_data)

# Display cumulative returns for each acquirer
print("Cumulative returns for 60 days after the announcement:")
for ticker, filtered_data in filtered_acquirer_data.items():
    if not filtered_data.empty:
        print(f"{ticker}: {filtered_data['cumulative_returns'].tail(1)}")


# %% Statistical Analysis
import pandas as pd
import numpy as np
from scipy import stats

# Step 6: Collect cumulative returns into a pandas Series
cumulative_returns = {}

# Extract the final cumulative return for each acquirer in the filtered data
for ticker, data in filtered_acquirer_data.items():
    if not data.empty:
        # Get the last value of the cumulative returns
        cumulative_return = data["cumulative_returns"].iloc[-1]
        cumulative_returns[ticker] = cumulative_return

# Convert the dictionary of cumulative returns into a pandas Series
returns_series = pd.Series(cumulative_returns)

# Step 7: Calculate Descriptive Statistics
mean_return = returns_series.mean()  # Calculate the mean
median_return = returns_series.median()  # Calculate the median
std_dev_return = returns_series.std()  # Calculate the standard deviation

# Step 8: Perform a one-sample t-test to check if the mean is significantly different from zero
t_statistic, p_value = stats.ttest_1samp(returns_series, 0)

# Display the results
print("Statistical Analysis of Cumulative Returns for 60 Days After M&A Announcements:")
print(f"Mean Cumulative Return: {mean_return:.4f}")
print(f"Median Cumulative Return: {median_return:.4f}")
print(f"Standard Deviation: {std_dev_return:.4f}")
print(f"One-Sample T-Test: T-Statistic = {t_statistic:.4f}, P-Value = {p_value:.4f}")


# %%
# Function to plot a histogram of cumulative returns
def plot_histogram(returns_series):
    plt.figure(figsize=(8, 6))  # Set plot size
    plt.hist(returns_series, bins=10, alpha=0.7, color='b', edgecolor='k')  # Plot histogram
    plt.xlabel("Cumulative Returns")
    plt.ylabel("Frequency")
    plt.title("Histogram of Cumulative Returns for 60 Days After M&A Announcement")
    plt.show()

# Call the function to plot the histogram
plot_histogram(returns_series)

# %%
import seaborn as sns

# Function to create a box plot for cumulative returns
def plot_boxplot(returns_series):
    plt.figure(figsize=(8, 6))  # Set plot size
    sns.boxplot(data=returns_series, color='cyan')  # Create box plot
    plt.xlabel("Cumulative Returns")
    plt.title("Box Plot of Cumulative Returns for 60 Days After M&A Announcement")
    plt.show()

# Call the function to plot the box plot
plot_boxplot(returns_series)

#%% Machine Learning Component
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#%% Merge DataFrames
# Merge DataFrames
return_df = returns_series.reset_index()
return_df.columns = ['tticker', 'cumulative_returns']
merged_data = pd.merge(m_a_data, return_df, on='tticker', how='inner')

# Check for NaN values
if merged_data.isnull().any().any():
    print("NaN values in merged data:", merged_data.isnull().sum())
else:
    print("Merged data contains no NaN values.")

# Normalize percentages and calculate ratios
merged_data['pct_stk'] /= 100
merged_data['a_postmerge_own_pct'] /= 100
merged_data['enterprise_value_ratio'] = merged_data['deal_value'] / merged_data['entval']

# Filter out specific industries
merged_data = merged_data[merged_data['industry'] != 'REITs']

# Handle missing values (if any)
merged_data.dropna(subset=['cumulative_returns', 'enterprise_value_ratio', 'amv', 'pm4wk'], inplace=True)

# Display the DataFrame head
print(merged_data.head())

#%% 
y = merged_data['cumulative_returns']
X = merged_data[['enterprise_value_ratio', 'amv', 'deal_value', 'pct_stk', 'a_postmerge_own_pct' ]]


# %% Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Fit the Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# %% Evaluate the Model
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Analyze Regression Coefficients
coefficients = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': lin_reg.coef_
})
print(coefficients)

# %% Analyze Regression Coefficients
# Scatter plot with regression line
scatter_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.lmplot(x='Actual', y='Predicted', data=scatter_data, scatter_kws={'alpha':0.5})
plt.title("Predicted vs. Actual Cumulative Returns")
plt.show()




# %% Industry vs Return Bar Chart
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your merged dataset is named 'merged_data' and contains an 'industry' column

# Group data by industry and calculate cumulative returns
industry_returns = merged_data.groupby('industry')['cumulative_returns'].mean()

# Plot the bar chart
plt.figure(figsize=(10, 6))
industry_returns.plot(kind='bar', color='skyblue')
plt.title('Average Cumulative Returns by Industry')
plt.xlabel('Industry')
plt.ylabel('Average Cumulative Returns')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


#%% Industry Specific Review
# Assuming 'merged_data' is your DataFrame and 'industry' and 'cumulative_returns' are the column names.

# Group data by industry and calculate the mean cumulative returns
industry_returns = merged_data.groupby('industry')['cumulative_returns'].mean()

# Sort the industries based on the average cumulative returns
sorted_industries = industry_returns.sort_values()

# Select the five industries with the greatest negative returns
top_negative_industries = sorted_industries.head(5)

# Print the five industries with the greatest negative returns
print(top_negative_industries)


# %%
# Export the DataFrame to an Excel file
merged_data.to_excel("Merged_Data.xlsx", index=False, engine='openpyxl')




# %%
