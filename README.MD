# Hand Sign translator with mediapipe and chromadb

### Introduction
Detects hand signs from a webcam feed and translates the signs into the nearest letter based on the ASL alphabet. The vector database (chromadb) comes pre-loaded with 26 letters of the alphabet A-Z. It is also possible to "train" the model on new hand signs.

### Instructions

Clone this repository:

```git clone https://github.com/LwrncLiu/hand_signs.git```

In the `hand_signs/` directory you just cloned, create a conda environment and activate it:

```
conda create -n aslenv python=3.10
conda activate aslenv
```
