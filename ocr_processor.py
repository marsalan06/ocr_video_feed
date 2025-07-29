import easyocr
import cv2
import logging
import numpy as np

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
        Returns list of tuples: (text, bbox, confidence)
        """
        logger.info("Starting text extraction from frame")
        
        # Convert frame to RGB (OpenCV captures in BGR, but OCR needs RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.info("Converted frame to RGB for OCR processing")
        
        # Run OCR on the frame
        results = self.reader.readtext(rgb)
        logger.info(f"OCR results: {results}")
        
        # Extract text, bounding box, and confidence
        # results format: [(bbox, text, confidence), ...]
        # bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        extracted_data = []
        for bbox, text, confidence in results:
            # Convert bbox to a more usable format
            # bbox is a list of 4 points, we'll use the top-left point for text placement
            top_left = bbox[0]  # [x, y] coordinates
            
            # Convert numpy types to regular Python types for better compatibility
            x = float(top_left[0]) if hasattr(top_left[0], 'item') else float(top_left[0])
            y = float(top_left[1]) if hasattr(top_left[1], 'item') else float(top_left[1])
            
            extracted_data.append((text, (x, y), confidence))
            logger.info(f"Extracted: '{text}' at position ({x}, {y}) with confidence {confidence:.2f}")
        
        logger.info(f"Extracted {len(extracted_data)} text regions")
        return extracted_data
