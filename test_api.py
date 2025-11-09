import requests

API_KEY = "CDQ1CDC3DVL8S6HY"   # 👈 paste the exact key you put in constants.py
symbol = "AAPL"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"

response = requests.get(url)
print("Status Code:", response.status_code)
print("Response:")
print(response.text[:500])  # print first 500 characters
