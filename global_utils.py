import socket

hostname=socket.gethostname()
IS_SERVER =  hostname.endswith('server') # if code is running on the server


import json

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

