# OCR Text Extraction System with Tesseract

A real-time text extraction system using Tesseract OCR with improved stability and duplicate prevention.

## Features

- **Tesseract OCR**: More stable and consistent text detection compared to EasyOCR
- **Real-time Processing**: Live camera feed with text extraction
- **Dynamic Bounding Box**: Single expanding text box for organized display
- **Duplicate Prevention**: Advanced filtering to prevent repetitive text detection
- **Text Storage**: Automatic saving of extracted text to file
- **Performance Optimized**: Faster processing with better stability

## Installation

### 1. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

### Controls
- Press `q` to quit
- Press `c` to clear accumulated text

## Key Improvements with Tesseract

### 1. **Stable Results**
- Traditional OCR algorithms provide more consistent results
- Better handling of different text orientations and fonts
- More predictable confidence scores

### 2. **Reduced Repetitive Detection**
- Advanced preprocessing reduces false positives
- Improved duplicate filtering with position tracking
- Better text boundary detection

### 3. **Performance Benefits**
- Faster processing (especially with optimized settings)
- Lower memory usage
- More consistent across different hardware

### 4. **Enhanced Filtering**
- Better noise reduction through preprocessing
- Improved confidence thresholds
- Advanced duplicate detection with temporal consistency

## Configuration

Key settings in `config.py`:
- `OCR_INTERVAL = 15`: Process every 15 frames
- `CONFIDENCE_THRESHOLD = 0.6`: Minimum confidence for detection
- `ASYNC_SLEEP_TIME = 0.02`: Optimized for Tesseract speed

## Output

- Real-time text display in expanding bounding box
- Text saved to `extracted_text.txt`
- Console logging for debugging

## Troubleshooting

If Tesseract is not found, ensure it's properly installed and in your system PATH. 