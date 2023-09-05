#!/bin/bash
# Add this directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`dirname "$(realpath $0)"`
#dask-scheduler &
#dask-worker 127.0.0.1:8786 --nprocs 1 --local-directory work-dir &
#python publish_data.py
#gunicorn "dash_opencellid.app:get_server()" --timeout 60 --workers 1
#wait

# dask-scheduler - host 127.0.0.1 - port 8786 &
# dask worker 127.0.0.1:8786 - nproc 4 &
python load_data.py 127.0.0.1:8786
gunicorn "app:get:server()" - bind 127.0.0.1:8050 - workers 3

echo "Press any key to continue"
read -n 1 -p "Input Selection:" mainmenuinput