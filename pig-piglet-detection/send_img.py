import requests
import base64
from PIL import Image
from io import BytesIO
import random
url = 'http://localhost:4545/predict'

img = "pig-piglet-detection/eval-images/image25.png"

myobj = {}
with open(img, "rb") as img_file:
    b64str = base64.b64encode(img_file.read()).decode('utf8')
    #im = Image.open(BytesIO(base64.b64decode(str(b64str))))
    #path = "pig-piglet-detection/eval-images/reeee" + str(random.randint(0,10000)) + ".png"
    #im.save(path, 'PNG')
    myobj['img'] = b64str

x = requests.post(url, json=myobj)
x.raise_for_status()

print(x.json.__dict__)
