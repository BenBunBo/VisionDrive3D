sudo docker run --name visiondrive3d \
  --gpus all \
  -v /home/ml4u/BKTeam/ChiDai:/workspace/source \
  -v /media/ml4u/Samsung_T5:/workspace/data \
  --shm-size=16g \
  -it pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime