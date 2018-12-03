OBS: The downloads below should stay in the same folder as the YOIO folder

First: Download and install darknet from this link: https://pjreddie.com/darknet/yolo/

Second: Download the inception folder from this link: https://drive.google.com/drive/folders/1DfTCb41l2qU12OsIo_TD7SFOsWbpClwS?usp=sharing

Third: Do this:
    cd darknet/cfg
    nano coco.data
    <Update the paths of the variables "names" and "#valid" to be absolute paths to their files>
    <Save the file>

Fourth: Change the paths in the main.py file.