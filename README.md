# Capstone_24.15_Soccer_Tracking
Latest Model Trained (as on June 5, 2024) is in retrained_model_3

## Steps to Reproduce
Dataset:

1. Gather image dataset of soccer balls
2. Upload to Label Studio by Google
3. Follow steps to label images
4. Upload dataset in a coco format to computer

Training:

1. Make sure coco dataset folder has train/, validation/, and test/ subfolders
    - each subfolder should have a labels.json and an images/ subfolder
2. Install all requirements (requirements.txt)
3. Replace all path fields with appropriate folder paths

Tracking:

- tracking.py is to open a quick python window to view ball tracking
- tracking2.py saves video into a .mp4 file

## Drone Streaming

1. Generate RTMP server, get RTMP address
2. For each drone, connect through the DJI Fly App to a unique port for your server 
    - (example: rtmp://your-server.com:1)
3. Add a live source on OBS with the same address
4. Adjust views of all drone cameras as desired
5. Create Twitch account or use existing account:
    - username: Seattleudrone
    - password: Drones@seattleu12

