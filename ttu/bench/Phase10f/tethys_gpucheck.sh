#!/bin/bash
# idle check for tethys GPU 1: exit 1 if a compute process exists or utilization is 10 percent or more
[ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] && [ "$(nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -lt 10 ]
