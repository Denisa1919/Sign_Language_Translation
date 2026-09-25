# AI Powered Sign Language Translation

## Project Overview

AI Powered Sign language Translation is a computer-vision based application that
detects hand gestures through a webcam and translates recognized sign language gestures 
into text and speech.

The systema uses OpenCV for video capture and image processing, MediaPipe for hand detection, 
and a trained TensorFlow/Keras model for gesture classification.



## Project Enhancement using Prompt enhancement

GenAI was used during the development and enhancement of the project to assist with code generation, debugging, architecture design,
feature planning, documentation, and optimization.

The following prompt cards document the major prompts used for project enhancement.


# Prompt Cards

## Prompt card 1 - Project Enhancement

### Prompt

```text

Analyze my existing AI-Powered sign language translation project and suggest
enhancements that can improve its accuracy, usability, performance, and real-time translation capabilities.

The prompt uses Python, OpenCV, MediaPipe, TensorFlow/Keras, and a webcam for real-time hand gesture recognition.

Suggest practical improvements that can be implemented in the existing architecture.

```


### Prompt Engineering Technique
Role + Context + Specific Requirements

### Expected Output

- Identification of current limitations
- Suggested new features
- Accuracy improvements
- Performance improvements
- Better user interaction
- Future development opportunities


## Prompt card 2 - Sign Language Recognition

### Prompt

```text

Design a real-time sign language recognition system using Python, openCV, MediaPipe, and TensorFlow/Keras.
The system should capture hand gestures from a webcam, detect the hand, extract the relevant region, preprocess the image,
classify the gesture using a trained model, and display the predicted sign as text.

Explain the complete processing pipeline.

```

### Prompt Engineering Technique

System Design Prompt

### Expected Output

```text
Webcam
   ↓
Frame Capture
   ↓
Hand Detection
   ↓
Bounding Box
   ↓
Image Preprocessing
   ↓
Keras Model
   ↓
Gesture Classification
   ↓
Text Output
```
## Prompt card 3 - Improve Gesture Detection

### Prompt
```text
How can I improve hand gesture detection in my sign language translation
project using MediaPipe?

The application needs to detect one or two hands in real time and should
remain stable when the user changes hand position, distance, or orientation.

Suggest improvements to hand detection, bounding box handling, and image
preprocessing.
```

### Prompt Engineering Technique

Problem + Context + Constraints

### Expected Output

- Better hand detection
- Support for multiple hands
- Improved bounding box calculation
- Reduced detection noise
- Better processing

