apt update
apt install aria2
aria2c -c -x16 -s16 --dir=./data --out=paper-slide-crawler/downloads.tar.gz https://modelscope.cn/datasets/TobyYang7/paper-slides/resolve/master/downloads.tar.gz
tar -vxzf ./paper-slide-crawler/downloads.tar.gz -C ./paper-slide-crawler