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
        self.reader = None
        self.initialization_successful = False
        
        try:
            # Initialize EasyOCR with GPU if available
            logger.info("Loading EasyOCR with GPU support...")
            self.reader = easyocr.Reader(['en'], gpu=True)  # GPU enabled by default
            logger.info("EasyOCR reader initialized with GPU support")
            self.initialization_successful = True
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            logger.info("Attempting to initialize EasyOCR with CPU fallback...")
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR reader initialized with CPU fallback")
                self.initialization_successful = True
            except Exception as cpu_error:
                logger.error(f"Failed to initialize EasyOCR with CPU: {cpu_error}")
                self.initialization_successful = False

    def is_initialized(self):
        """Check if OCR processor was initialized successfully"""
        return self.initialization_successful and self.reader is not None

    def extract_text(self, frame):
        """
        Use EasyOCR to extract text from the given frame.
        Returns list of tuples: (text, position, confidence)
        """
        if not self.is_initialized():
            logger.error("OCR processor not properly initialized")
            return []
        
        if frame is None:
            logger.error("Input frame is None")
            return []
        
        try:
            logger.info("Starting text extraction from frame")
            
            # Validate frame dimensions
            if len(frame.shape) != 3:
                logger.error(f"Invalid frame shape: {frame.shape}, expected 3D array")
                return []
            
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                logger.error(f"Invalid frame dimensions: {width}x{height}")
                return []
            
            logger.info(f"Processing frame with dimensions: {width}x{height}")
            
            # Convert frame to RGB (OpenCV captures in BGR, but OCR needs RGB)
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                logger.info("Converted frame to RGB for OCR processing")
            except Exception as color_error:
                logger.error(f"Failed to convert frame color space: {color_error}")
                return []
            
            # Run OCR on the frame
            try:
                results = self.reader.readtext(rgb)
                logger.info(f"OCR results: {results}")
            except Exception as ocr_error:
                logger.error(f"OCR processing failed: {ocr_error}")
                return []
            
            # Extract text, bounding box, and confidence
            # results format: [(bbox, text, confidence), ...]
            # bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            extracted_data = []
            
            for i, (bbox, text, confidence) in enumerate(results):
                try:
                    # Validate OCR result
                    if not isinstance(text, str) or not text.strip():
                        logger.warning(f"Skipping empty or invalid text at index {i}")
                        continue
                    
                    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                        logger.warning(f"Invalid confidence value {confidence} for text '{text}', skipping")
                        continue
                    
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        logger.warning(f"Invalid bbox format for text '{text}', skipping")
                        continue
                    
                    # Convert bbox to a more usable format
                    # bbox is a list of 4 points, we'll use the top-left point for text placement
                    top_left = bbox[0]  # [x, y] coordinates
                    
                    # Validate coordinates
                    if not isinstance(top_left, list) or len(top_left) != 2:
                        logger.warning(f"Invalid top_left coordinates for text '{text}', skipping")
                        continue
                    
                    # Convert numpy types to regular Python types for better compatibility
                    try:
                        x = float(top_left[0]) if hasattr(top_left[0], 'item') else float(top_left[0])
                        y = float(top_left[1]) if hasattr(top_left[1], 'item') else float(top_left[1])
                    except (ValueError, TypeError) as coord_error:
                        logger.warning(f"Failed to convert coordinates for text '{text}': {coord_error}")
                        continue
                    
                    # Validate coordinate ranges
                    if x < 0 or y < 0 or x >= width or y >= height:
                        logger.warning(f"Coordinates ({x}, {y}) out of frame bounds ({width}x{height}) for text '{text}'")
                        # Clamp coordinates to frame bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                    
                    extracted_data.append((text, (x, y), confidence))
                    logger.info(f"Extracted: '{text}' at position ({x}, {y}) with confidence {confidence:.2f}")
                    
                except Exception as item_error:
                    logger.error(f"Error processing OCR result at index {i}: {item_error}")
                    continue
            
            logger.info(f"Successfully extracted {len(extracted_data)} text regions")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Unexpected error in extract_text: {e}")
            return []
