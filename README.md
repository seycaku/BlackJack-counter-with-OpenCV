# BlackjackVision

BlackjackVision is a simple computer vision project built with Python and OpenCV.
The program detects playing cards from a webcam, recognizes their rank and suit, and calculates the Blackjack score in real time.

## Features

* Real-time card detection using webcam
* Card contour detection with OpenCV
* Perspective transform for rotated cards
* Rank and suit recognition using template matching
* Blackjack score calculation
* Visual UI with detected cards and score display

## Technologies

* Python
* OpenCV
* NumPy

## How It Works

1. The program captures frames from the webcam.
2. The image is preprocessed using grayscale, blur, thresholding, and morphology operations.
3. Card contours are detected and filtered.
4. Each card is transformed into a normalized top-down view.
5. The top-left corner of the card is extracted.
6. Rank and suit symbols are matched with templates.
7. Blackjack score is calculated and displayed on screen.

## Project Structure

```text
project/
│
├── templates1/
│   ├── Ace.jpg
│   ├── Two.jpg
│   ├── King.jpg
│   ├── Hearts.jpg
│   └── ...
│
├── main.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/seycaku/BlackJack-counter-with-OpenCV.git
cd BlackJack-counter-with-OpenCV
```

Install dependencies:

```bash
pip install opencv-python numpy
```

## Run

```bash
python main.py
```

Controls:

* Press `q` to quit
* Press `s` to save current frame

## Requirements

* Python 3.x
* Webcam
* Template images inside the `templates1` folder

## Example

The program detects cards such as:

```text
Ace of Spades
Ten of Hearts
Queen of Clubs
```

And displays the Blackjack score:

```text
Score: 21  BLACKJACK
```

## Limitations

* Works best with good lighting
* Sensitive to blurry images
* Template matching depends on card design similarity
* Here exists also some test cases for image recognition but now it's work only with webcam

## Future Improvements

* Better recognition under difficult lighting
* Support for more card styles
* Machine learning based classification
* Improved UI and animations
* Add select for images or real time recognition 
* Add game logic and cheating system
