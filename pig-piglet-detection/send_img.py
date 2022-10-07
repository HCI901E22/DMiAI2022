import requests
import base64
url = '0.0.0.0:5454/predict'

img = "you/path/here.png"

myobj = {}
with open(img, "rb") as img_file:
    b64str = base64.b64encode(img_file.read())
    myobj['img'] = b64str

x = requests.post(url, json=myobj)
