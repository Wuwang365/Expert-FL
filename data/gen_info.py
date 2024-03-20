import glob
import json

paths = glob.glob("traindata/*")
clients = [path.split('/')[-1] for path in paths]
info = {}
for client in clients:
    info[client] = {
        "delay":0
    }
    
with open('info.json','w') as f:
    f.write(json.dumps(info,indent=4))