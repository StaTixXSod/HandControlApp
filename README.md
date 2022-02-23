# Hand Control App
## Intro
I wrote the program that based on [MediaPipe Hands module](https://google.github.io/mediapipe/solutions/hands) and 
allows you to control your computer with gestures. It uses your webcam for recognizing
hand movements and performs system control with gestures. There are 3 gestures, that an app can handle.
- Pointing
- Scrolling
- Swiping

## Gestures
### Pointing
#### Pose
The gesture "Pointing" allows you to control the mouse.
Your hand pose should be the same as if you were pointing at something, except the thumb should be aside.

<img height="256" src="https://github.com/StaTixXSod/HandControlApp/blob/master/images/pointing_1.png?raw=true" title="pointing 1" width="256"/>
<img height="256" src="https://github.com/StaTixXSod/HandControlApp/blob/master/images/pointing_2.png?raw=true" title="pointing 2" width="256"/>

![Alt Text](https://github.com/StaTixXSod/HandControlApp/blob/master/images/ezgif_cursor.gif?raw=true)

#### Click 
To make a click you have to rich with your thumb your midpoint of middle finger, just like on gif below.

![Alt Text](https://github.com/StaTixXSod/HandControlApp/blob/master/images/ezgif_click.gif?raw=true)

### Scrolling
Idea for scrolling position I took from touchpad controlling
(actually all 3 gestures were taken from touchpad controlling, but anyway).
For perform this gesture, you have to reach up forefinger and middle finger and move it up and down.

<img height="256" src="https://github.com/StaTixXSod/HandControlApp/blob/master/images/scrolling_1.png?raw=true" title="scrolling 1" width="256"/>
<img height="256" src="https://github.com/StaTixXSod/HandControlApp/blob/master/images/scrolling_2.png?raw=true" title="scrolling 2" width="256"/>

![Alt Text](https://github.com/StaTixXSod/HandControlApp/blob/master/images/ezgif_scrolling.gif?raw=true)

### Swiping
This gesture allows you switching between desktops.
For making this, you have to open your palm and move in the chosen direction.

<img height="256" src="https://github.com/StaTixXSod/HandControlApp/blob/master/images/swiping_1.png?raw=true" title="swiping 1" width="256"/>
<img height="256" src="https://github.com/StaTixXSod/HandControlApp/blob/master/images/swiping_2.png?raw=true" title="swiping 2" width="256"/>

![Alt Text](https://github.com/StaTixXSod/HandControlApp/blob/master/images/ezgif_swiping.gif?raw=true)

## Requirements

- onnxruntime
- OpenCV
- MediaPipe
- PyAutoGUI
- Pynput

Actually the project needs 2 more libs: `numpy` and `imutils`, but the main libraries shown above.
The full list in the `requirements.txt`. And I don't know about python compatibility here. 
I have `python 3.8` installed on my machine.

## Installation
```
- git clone "https://github.com/StaTixXSod/HandControlApp.git"
- cd HandControlApp
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
```

## Settings (Arguments)

| Arguments   | Meaning                                  |
| ----------- |:----------------------------------------:|
| --mode      | Model complexity: 0 or 1                 |
| --hands     | Number of hands to be used               |
| --detConf   | Detection confidence                     |
| --trackConf | Track confidence                         |
| --window    | Smoothing window                         |
| --curV      | Cursor velocity                          |
| --scrV      | Scrolling velocity                       |
| --clickD    | Click distance                           |
| --swipeD    | Distance for swiping                     |
| --direction | Swipe direction (horizontal or vertical) |

## Run
If you want to run app with default settings, just type:
> python app.py

If you want to change something, for example change cursor velocity to 150, type:

> python app.py --curV=150

## Notice
> As you know, this app uses web camera, so it would be nice to use this app with good lighting
or good camera (or both). Furthermore, it is better to use good PC, because this app has become a bit heavy.
