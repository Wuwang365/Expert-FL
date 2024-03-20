Build base on flask and torch
You need to modify ```CUDA_LIST``` and ```RAW_CUDA_LIST``` to satisfy with your cuda device
## run server
```python start_server.py```
## run client
```python start_client.py```
start_client is employed to run 120 clients parallelly.
To run one client ```python client_wapper.py```
## log file
You can find the run log at ```log_files/server_log.txt```

## data format
```
|-data // data directory
|--traindata // traindata directory
|---1 // client 1 data directory
|----0 // client 1 class 0 directory
|-----1.png
|-----2.png
|----1
|-----3.png
```