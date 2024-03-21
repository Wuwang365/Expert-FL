python start_server.py --logname beta2 --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 1 --port 8080 &
sleep 5
python start_client.py --info data/info.json --port 8080 --dataroot data/traindata

