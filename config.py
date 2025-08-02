"""
Configuration file for the OCR system with bounding box management.
"""

# Single dynamic bounding box configuration
# Format: (x1_ratio, y1_ratio, x2_ratio, y2_ratio) as ratios of frame dimensions
SINGLE_BOX_RATIOS = (0.1, 0.1, 0.9, 0.7)  # Single box: 10% to 90% width, 10% to 70% height (increased from 30%)

# Text manager configuration
TEXT_HEIGHT = 25          # Height allocated for each line of text (reduced from 30)
TEXT_MARGIN = 8           # Margin from box edges (reduced from 10)
TEXT_COLOR = (0, 255, 0) # Green color for text
BOX_COLOR = (255, 0, 0)  # Blue color for bounding boxes

# OCR configuration - Optimized for Tesseract
OCR_INTERVAL = 15         # Process OCR every N frames (reduced for Tesseract's faster processing)
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for text detection (increased for Tesseract stability)

# Camera configuration
CAMERA_SOURCE = 0         # Camera source (0 for default camera)

# Display configuration
WINDOW_TITLE = "Live Text Extraction with Tesseract OCR"
INFO_TEXT_COLOR = (255, 255, 255)  # White color for info text

# Error handling configuration
MAX_CONSECUTIVE_FAILURES = 10  # Maximum consecutive frame read failures
ASYNC_SLEEP_TIME = 0.02       # Reduced sleep time for Tesseract's faster processing

# Text storage configuration
SAVE_TEXT_TO_FILE = True      # Save accumulated text to file
TEXT_FILE_PATH = "extracted_text.txt"  # File path for saving text
CLEAR_FILE_ON_START = True    # Clear file when starting new session

# Frame comparison configuration for change detection
ENABLE_FRAME_COMPARISON = True     # Enable frame comparison to avoid processing duplicate frames
FRAME_SIMILARITY_THRESHOLD = 0.98  # Threshold for frame similarity (0.0 to 1.0) - very strict to avoid duplicates
TEXT_CHANGE_THRESHOLD = 0.8        # Threshold for detecting text changes in regions - strict to avoid duplicates
MAX_STORED_FRAMES = 3              # Number of previous frames to store for comparison 