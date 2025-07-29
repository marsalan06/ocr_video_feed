import easyocr
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        logger.info("Initializing OCRProcessor")
        # Initialize EasyOCR with GPU if available
        self.reader = easyocr.Reader(['en'], gpu=True)  # GPU enabled by default
        logger.info("EasyOCR reader initialized with GPU support")

    def extract_text(self, frame):
        """
        Use EasyOCR to extract text from the given frame.
        """
        logger.info("Starting text extraction from frame")
        
        # Convert frame to RGB (OpenCV captures in BGR, but OCR needs RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.info("Converted frame to RGB for OCR processing")
        
        # Run OCR on the frame
        results = self.reader.readtext(rgb)
        logger.info(f"OCR results: {results}")
        
        # Extract only the text (filter out bounding boxes and confidences)
        texts = [text for _, text, _ in results]
        logger.info(f"Extracted texts: {texts}")
        
        return texts
