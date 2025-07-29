"""
Configuration file for the OCR system with bounding box management.
"""

# Single dynamic bounding box configuration
# Format: (x1_ratio, y1_ratio, x2_ratio, y2_ratio) as ratios of frame dimensions
SINGLE_BOX_RATIOS = (0.1, 0.1, 0.9, 0.3)  # Single box: 10% to 90% width, 10% to 30% height

# Text manager configuration
TEXT_HEIGHT = 30          # Height allocated for each line of text
TEXT_MARGIN = 10          # Margin from box edges
TEXT_COLOR = (0, 255, 0) # Green color for text
BOX_COLOR = (255, 0, 0)  # Blue color for bounding boxes

# OCR configuration
OCR_INTERVAL = 15         # Process OCR every N frames
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for text detection

# Camera configuration
CAMERA_SOURCE = 0         # Camera source (0 for default camera)

# Display configuration
WINDOW_TITLE = "Live Text Extraction with Dynamic Bounding Box"
INFO_TEXT_COLOR = (255, 255, 255)  # White color for info text

# Error handling configuration
MAX_CONSECUTIVE_FAILURES = 10  # Maximum consecutive frame read failures
ASYNC_SLEEP_TIME = 0.03       # Sleep time between frames

# Text storage configuration
SAVE_TEXT_TO_FILE = True      # Save accumulated text to file
TEXT_FILE_PATH = "extracted_text.txt"  # File path for saving text
CLEAR_FILE_ON_START = True    # Clear file when starting new session 