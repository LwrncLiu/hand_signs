# Hand Sign translator with mediapipe and chromadb

Detects hand signs from a webcam feed and translates the signs into the nearest letter based on the ASL alphabet. The vector database (chromadb) comes pre-loaded with 26 letters of the alphabet A-Z. It is also possible to "train" the model on new hand signs.

<p align="center">
<img src="src/static/asl_alphabet.png" alt="ASL Alphabet" style="width:50%;">
</p>

### Requirements
- anaconda
- access to webcam 

### Setup Instructions

Clone this repository:

```git clone https://github.com/LwrncLiu/hand_signs.git```

In the `hand_signs/` directory you just cloned, create a conda environment and activate it:

```
conda create -n aslenv python=3.10
conda activate aslenv
```

Install the necessary python packages using pip 

```
pip install -r requirements.txt
```

Run `app.py` to spin up the application
```
python app.py
```

### Navigate the UI
Upon startup, you see the home screen with the webcam feed in the center and three buttons along the bottom: `START TEST`, `CAPTURE PHOTO`, `RESET DB`. 

#### `START TEST`
The orientation of the detected hands (if any) is normalized, scaled and queried against the chromadb vector database. If a given orientation's vector representation's euclidean distance is within 0.10 of an existing vector in the vector database, then the label for the existing vector is returned and drawn on the webcam feed. 

Try signing one of the hand signs in the ASL Alphabet diagram above to display the letter (label) on screen!

To stop testing, simply click the `STOP TEST` button to return to the home screen.

### `CAPTURE PHOTO`
To extend this project's capabilities (e.g. adding more hand signs or increasing the "training data" of a given label), orient your hand (left or right) in view of the webcam and click `CAPTURE PHOTO`. A preview of the captured photo is shown. From here, you can add a label to the captured photo and add the hand sign to the vector database with `SUBMIT & SAVE PHOTO`. 

If the captured hand sign is not accurate or no hands were detected in the captured photo, click `RETRY` to return to the home screen.

### `RESET DB`
If you do not like the existing hand signs stored or want to start from scratch clicking `RESET DB` will delete all records in the chromadb collection so that you can start from scratch. 

This button is RED because the effects are irreversible. 