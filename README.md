# Face Recognition & Name Attachment System
## Overview

This project provides a complete face recognition solution that identifies individuals in images or video streams and attaches their names to detected faces. It uses state-of-the-art face recognition algorithms with a simple command-line interface for various operations.

**Key Features:**
- Real-time face recognition from webcam feed
- Face detection and encoding using dlib's deep learning models
- Simple training interface with automatic encoding generation
- Multiple operation modes: training, validation, testing, and live camera
- Configurable recognition parameters and thresholds
- Visual bounding boxes with name annotations

## Algorithms and Libraries Used

- **Face Detection**: `face_recognition` library (using dlib's HOG or CNN models)
- **Face Encoding**: 128-dimensional embeddings using ResNet model
- **Face Matching**: Euclidean distance comparison with `face_recognition.compare_faces()`
- **Majority Voting**: `Counter` from collections for name determination
- **Image Processing**: OpenCV and Pillow libraries
- **Data Serialization**: Pickle for storing face encodings

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/inquaid/FaceRecognition_AttachName.git
cd FaceRecognition_AttachName
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** For GPU acceleration (CNN model), you'll need CUDA-compatible GPU and dlib with CUDA support.

## Directory Structure
```
FaceRecognition_AttachName/
├── training/             # Training images (subdirectories per person)
├── validation/           # Validation images
├── output/               # Output files (encodings.pkl)
├── ditector.py               # Main program
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Usage

### Training the Model
1. Create subdirectories in `training/` for each person:
```
training/
├── Person1/
│   ├── image1.jpg
│   └── image2.jpg
├── Person2/
│   ├── photo1.png
│   └── photo2.jpg
```

2. Run training command:
```bash
python main.py --train
```

3. (Optional) Use CNN model for better accuracy (requires GPU):
```bash
python main.py --train -m cnn
```

### Recognizing Faces

**1. Webcam Real-time Recognition:**
```bash
python main.py --camera
```
- Press 'Q' to quit
- Detected faces will be shown with bounding boxes and names

**2. Test on Single Image:**
```bash
python main.py --test -f path/to/image.jpg
```

**3. Validate on Validation Set:**
```bash
python main.py --validate
```

### Command Line Options
```
--train       Train model with images in 'training/' directory
--validate    Test model on validation images
--test        Test model on single image (requires -f)
--camera      Run real-time recognition from webcam
-m MODEL      Face detection model: 'hog' (CPU) or 'cnn' (GPU)
-f IMAGE      Image file path for testing
```

## Technical Details

### Face Recognition Pipeline
1. **Face Detection**:
   - Uses dlib's HOG (Histogram of Oriented Gradients) or CNN face detector
   - Locates face bounding boxes in input images

2. **Face Encoding**:
   - Generates 128-dimensional face embeddings using ResNet model
   - Encodings stored in `output/encodings.pkl`

3. **Face Matching**:
   - Compares new face encodings with stored encodings
   - Uses Euclidean distance with configurable tolerance
   - Implements majority voting for name determination

### Recognition Parameters
Key adjustable parameters in code:
```python
# Detection parameters
scale_factor = 0.5  # Image scaling for faster processing

# Matching parameters
tolerance = 0.55    # Lower = stricter matching (0.6 default)
min_votes = 2       # Minimum matches required for recognition
```

## Customization

### Adding New People
1. Create new directory in `training/` with person's name
2. Add 10-20 clear facial images (different angles/lighting)
3. Retrain the model: `python main.py --train`

### Adjusting Recognition Sensitivity
Modify these values in `main.py`:
```python
# In recognize_faces_in_camera() function:
tolerance = 0.55  # Decrease for stricter matching
min_votes = 2     # Increase to require more matches
```

## Troubleshooting

**Problem:** No faces detected in images  
**Solution:**
- Use higher resolution images (min 640x480)
- Ensure faces are clearly visible and not obscured
- Try different detection model: `-m cnn`

**Problem:** Poor recognition accuracy  
**Solution:**
- Add more training images (15-20 per person)
- Use consistent lighting in training images
- Decrease tolerance value (0.5-0.55)
- Ensure training images show different facial expressions

**Problem:** Webcam not detected  
**Solution:**
- Check camera permissions
- Verify camera index in code (default is 0)
- Try different video capture index: `cv2.VideoCapture(1)`

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Created by inquaid**  
[Report Issues](https://github.com/inquaid/FaceRecognition_AttachName/issues) | 
[View on GitHub](https://github.com/inquaid/FaceRecognition_AttachName)