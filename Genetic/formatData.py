import glob
import pandas as pd
import requests
import zipfile

headers = {
    'Authorization': 'Bearer t.',# Add own token
}

params = {
    'figi': 'TCS00A107UL4',
    'year': '2024',
}

response = requests.get('https://invest-public-api.tinkoff.ru/history-data', params=params, headers=headers)

with open('rawdata.zip', 'wb') as f:
    f.write(response.content)


with zipfile.ZipFile("rawdata.zip", 'r') as zip_ref:
    zip_ref.extractall("rawdata tcsg")

trainFiles = glob.glob('rawdata tcsg/*.{}'.format('csv'))
csvAppend = pd.DataFrame()
trainFiles[:] = [el for el in trainFiles if len(pd.read_csv(el, sep=';').index) > 600]

validation_days = 5
name = 'TCSG'

validFiles = trainFiles[-validation_days:]
trainFiles = trainFiles[:-validation_days]

for file in validFiles:
    df = pd.read_csv(file, sep=';')
    df.columns = ['toDelete', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'None']
    df = df.drop('toDelete', axis=1)
    df = df.drop('None', axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df.Date.dt.time.between(pd.to_datetime('10:00').time(), pd.to_datetime('20:45').time())]
    df['Date'] = df['Date'].dt.time
    csvAppend = csvAppend._append(df, ignore_index=True)

csvAppend.to_csv("valid"+name+".csv", index=False, sep=';')
csvAppend = pd.DataFrame()

for file in trainFiles:
    df = pd.read_csv(file, sep=';')
    df.columns = ['toDelete', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'None']
    df = df.drop('toDelete', axis=1)
    df = df.drop('None', axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df.Date.dt.time.between(pd.to_datetime('10:00').time(), pd.to_datetime('20:45').time())]
    df['Date'] = df['Date'].dt.time
    csvAppend = csvAppend._append(df, ignore_index=True)

csvAppend.to_csv("data"+name+".csv", index=False, sep=';')
csvAppend = pd.DataFrame()

for file in trainFiles:
    df = pd.read_csv(file, sep=';')
    df.columns = ['toDelete', 'date', 'open', 'close', 'high', 'low', 'volume', 'None']
    df = df.drop('toDelete', axis=1)
    df = df.drop('None', axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.date.dt.time.between(pd.to_datetime('10:00').time(), pd.to_datetime('20:45').time())]
    csvAppend = csvAppend._append(df, ignore_index=True)

csvAppend.to_csv("gymdata"+name+".csv", index=False, sep=';')