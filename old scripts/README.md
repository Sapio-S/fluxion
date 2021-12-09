environment:
```
pip install -r fluxion-requirements.txt
PATH=$PATH:/home/yuqingxie/.local/bin # 每次开机需要重新将部分package放入环境变量
sudo apt-get install graphviz graphviz-doc # 安装graphviz
```

mkdir:
```
mkdir log
mkdir tmp_data
```

clean data:
`python3 data_clean.py`
or upload data in `res_rm_outlier`

data preparation:
`METRIC_NUM=4 python3 data.py`

run big model:
`METRIC_NUM=1 python3 big_model.py`

run p90 only:
`METRIC_NUM=1 python3 multimodel.py`

run multipercentile:
`METRIC_NUM=4 python3 multimodel.py`

nnictl create --config /home/yuqingxie/autosys/code/fluxion/nni/config.yaml -p 8080