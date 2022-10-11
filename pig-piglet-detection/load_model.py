import requests

url = "http://kurtskammerater.westeurope.cloudapp.azure.com:4545/load"

data = {'path': "snapshots/model50.h5"}

res = requests.post(url, json=data)

print(res.text)
