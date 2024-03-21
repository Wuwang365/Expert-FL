python start_server.py --logname beta5 --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 1 --port 8080 --savepath /data/xy/TMC/Expert-FL/models/beta5&
sleep 3
python start_client.py --info data/info.json --port 8080 --dataroot /data/xy/TMC/data/120client/beta_0.5



python start_server.py --logname beta4 --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 2 --port 8081 --savepath /data/xy/TMC/Expert-FL/models/beta4&
sleep 3
python start_client.py --info data/info.json --port 8081 --dataroot /data/xy/TMC/data/120client/beta_0.4


python start_server.py --logname beta3 --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 3 --port 8082 --savepath /data/xy/TMC/Expert-FL/models/beta3&
sleep 3
python start_client.py --info data/info.json --port 8082 --dataroot /data/xy/TMC/data/120client/beta_0.3

python start_server.py --logname beta2 --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 4 --port 8083 --savepath /data/xy/TMC/Expert-FL/models/beta2&
sleep 3
python start_client.py --info data/info.json --port 8083 --dataroot /data/xy/TMC/data/120client/beta_0.2


python start_server.py --logname beta1 --info data/info.json --parallelnum 8 --classnum 10 --testroot data/testdata --cuda 5 --port 8084 --savepath /data/xy/TMC/Expert-FL/models/beta1&
sleep 3
python start_client.py --info data/info.json --port 8084 --dataroot /data/xy/TMC/data/120client/beta_0.1
sleep 3600
pkill python 