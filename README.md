# Real-Time OCR System with Dynamic Bounding Box Management

A robust real-time OCR (Optical Character Recognition) system that extracts text from live video feed and places it sequentially within dynamic bounding boxes. The system automatically corrects mirror image issues and saves extracted text to files.

## Features

### 🎯 **Core Functionality**
- **Real-time text extraction** from live camera feed
- **Dynamic bounding boxes** that adapt to frame size
- **Mirror image correction** - automatically flips camera feed
- **Sequential text placement** within predefined regions
- **Confidence-based filtering** for high-quality text detection
- **Comprehensive error handling** to prevent crashes
- **Configurable settings** for easy customization

### 📦 **Dynamic Bounding Box Management**
- **Responsive layout**: Boxes automatically adjust to frame dimensions
- **Sequential placement**: Text is placed in order within defined boxes
- **Automatic overflow handling**: Moves to next box when current is full
- **Visual feedback**: Bounding boxes are drawn on screen for reference
- **Configurable ratios**: Easy to modify box positions using frame ratios

### 💾 **Text Storage System**
- **Automatic file saving**: Text is saved to file as it's detected
- **Timestamped entries**: Each text entry includes timestamp
- **Session management**: Clear file on start or append to existing
- **Storage statistics**: Track text length and file information

### 🛡️ **Error Handling**
- **Camera failure recovery**: Handles camera disconnection gracefully
- **OCR processing errors**: Continues operation even if OCR fails
- **Input validation**: Validates all inputs to prevent crashes
- **Resource cleanup**: Proper cleanup of camera and display resources

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd my_ocr_base_code
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python main.py
```

## Configuration

Edit `config.py` to customize the system:

```python
# Dynamic bounding boxes configuration (ratios of frame dimensions)
BOUNDING_BOX_RATIOS = [
    (0.1, 0.1, 0.9, 0.25),   # Box 1 - Top section (10% to 90% width, 10% to 25% height)
    (0.1, 0.3, 0.9, 0.45),   # Box 2 - Upper middle
    (0.1, 0.5, 0.9, 0.65),   # Box 3 - Lower middle
    (0.1, 0.7, 0.9, 0.85)    # Box 4 - Bottom section
]

# Text storage settings
SAVE_TEXT_TO_FILE = True      # Enable/disable text saving
TEXT_FILE_PATH = "extracted_text.txt"  # File path for saving text
CLEAR_FILE_ON_START = True    # Clear file when starting new session

# OCR settings
OCR_INTERVAL = 15         # Process OCR every N frames
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for text detection
```

## Usage

### **Running the Application**
```bash
python main.py
```

### **Controls**
- **`q`**: Quit the application
- **`c`**: Clear accumulated text and reset bounding boxes

### **Display Information**
- **Blue rectangles**: Dynamic bounding boxes that adapt to frame size
- **Green text**: Detected text placed within boxes
- **White text**: Status information at bottom of screen
- **Box counter**: Shows current box being used (e.g., "Box: 2/4")
- **Storage info**: Shows file path and text length

## Text Storage

### **Where Text is Stored**
- **File location**: `extracted_text.txt` (configurable in `config.py`)
- **Format**: Timestamped entries with detected text
- **Example**:
```
=== OCR Text Extraction Session Started: 2024-01-15 14:30:25 ===

[14:30:26] Hello World
[14:30:28] This is some text
[14:30:30] Another line of text
```

### **Storage Features**
- **Automatic saving**: Text is saved as soon as it's detected
- **Session tracking**: Each session is clearly marked
- **Timestamped entries**: Know exactly when text was detected
- **File statistics**: Track file size and modification time

## Architecture

### **File Structure**
```
my_ocr_base_code/
├── main.py                    # Main application entry point
├── camera_feed.py             # Video stream management with mirror correction
├── ocr_processor.py           # Text extraction using EasyOCR
├── text_manager.py            # Dynamic bounding box and text placement logic
├── ner_processor.py           # Named Entity Recognition (unused)
├── utils.py                   # Text visualization utilities
├── config.py                  # Configuration settings
├── extracted_text.txt         # Extracted text file (created automatically)
└── requirements.txt           # Python dependencies
```

### **Data Flow**
```
Camera Feed → Mirror Correction → OCR Processing → Text Analysis → Dynamic Bounding Box Placement → File Storage → Display
```

### **Key Components**

#### **TextManager** (`text_manager.py`)
- Manages dynamic bounding boxes based on frame dimensions
- Handles sequential text placement within responsive boxes
- Provides automatic text storage to files
- Tracks accumulated text and current position

#### **VideoStream** (`camera_feed.py`)
- Background thread for smooth frame capture
- Automatic mirror image correction
- Camera initialization and cleanup
- Error recovery for camera failures

#### **OCRProcessor** (`ocr_processor.py`)
- Uses EasyOCR for text extraction
- GPU acceleration support with CPU fallback
- Confidence-based filtering
- Comprehensive error handling

## Mirror Image Correction

The system automatically corrects the mirror effect common in webcams:
- **Automatic flipping**: Camera feed is horizontally flipped
- **Natural appearance**: Text appears in correct orientation
- **No configuration needed**: Works automatically

## Dynamic Bounding Boxes

Bounding boxes automatically adapt to different frame sizes:
- **Responsive design**: Boxes scale with frame dimensions
- **Ratio-based positioning**: Uses frame ratios instead of fixed pixels
- **Automatic recalculation**: Updates when frame size changes
- **Configurable layout**: Easy to modify box positions

## Error Handling

The system includes comprehensive error handling for:

- **Camera initialization failures**
- **Frame reading errors**
- **OCR processing failures**
- **Invalid input data**
- **Display errors**
- **File storage issues**
- **Resource cleanup issues**

All errors are logged with appropriate severity levels and the system continues operation where possible.

## Performance Optimization

- **Frame skipping**: OCR runs every 15 frames by default
- **Confidence filtering**: Only high-confidence text is displayed
- **Background processing**: Camera feed runs in separate thread
- **Async operations**: Non-blocking video processing
- **Efficient storage**: Text is saved incrementally

## Customization

### **Modifying Bounding Box Layout**
Edit `config.py`:
```python
BOUNDING_BOX_RATIOS = [
    (x1_ratio, y1_ratio, x2_ratio, y2_ratio),  # Add your custom boxes
    # ... more boxes
]
```

### **Changing Text Storage**
```python
SAVE_TEXT_TO_FILE = True      # Enable/disable saving
TEXT_FILE_PATH = "my_text.txt"  # Custom file path
CLEAR_FILE_ON_START = False   # Append to existing file
```

### **Adjusting OCR Settings**
```python
OCR_INTERVAL = 15         # Processing frequency
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence
```

## Troubleshooting

### **Camera Not Working**
- Check camera permissions
- Try different camera source in `config.py`
- Ensure camera is not in use by another application

### **Mirror Image Still Appears**
- The system automatically corrects mirror effect
- If you still see mirror effect, check if another application is interfering

### **Bounding Boxes Not Visible**
- Check frame dimensions in console output
- Verify box ratios in `config.py`
- Ensure camera is providing valid frames

### **Text Not Being Saved**
- Check `SAVE_TEXT_TO_FILE` setting in `config.py`
- Verify file path permissions
- Check the `extracted_text.txt` file directly

### **OCR Not Detecting Text**
- Ensure good lighting conditions
- Check text is clearly visible in camera view
- Adjust `CONFIDENCE_THRESHOLD` if needed

### **Performance Issues**
- Reduce `OCR_INTERVAL` for faster processing
- Increase `CONFIDENCE_THRESHOLD` to filter more text
- Check GPU availability for OCR acceleration

## Dependencies

- **OpenCV**: Video capture and image processing
- **EasyOCR**: Text recognition with GPU support
- **PyTorch**: Deep learning backend
- **Transformers**: NER capabilities (optional)
- **NumPy**: Numerical operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add error handling for new features
5. Test thoroughly
6. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in console output
3. Ensure all dependencies are installed
4. Verify camera permissions and availability
5. Check the `extracted_text.txt` file directly for stored text 