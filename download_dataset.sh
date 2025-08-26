apt update
apt install aria2 pigz -y
aria2c -c -x16 -s16 --out=paper-slide-crawler/downloads.tar.gz https://modelscope.cn/datasets/TobyYang7/paper-slides/resolve/master/downloads.tar.gz
tar -I "pigz -p8" -vxf ./paper-slide-crawler/downloads.tar.gz -C ./paper-slide-crawler