import cv2
import logging
import numpy as np
import pytesseract
from PIL import Image
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        logger.info("Initializing OCRProcessor with Tesseract")
        self.initialization_successful = False
        
        try:
            # Check if Tesseract is available
            pytesseract.get_tesseract_version()
            logger.info("Tesseract is available")
            
            # Configure Tesseract for better performance
            self.tesseract_config = '--oem 3 --psm 6'
            
            # Initialize text tracking for duplicate prevention
            self.previous_texts = set()
            self.text_confidence_cache = {}
            self.frame_count = 0
            
            self.initialization_successful = True
            logger.info("Tesseract OCR processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            logger.error("Please install Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
            self.initialization_successful = False

    def is_initialized(self):
        """Check if OCR processor was initialized successfully"""
        return self.initialization_successful

    def preprocess_frame(self, frame):
        """
        Preprocess frame for better OCR results.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding for better text contrast
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Apply slight dilation to make text more readable
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            dilated = cv2.dilate(cleaned, kernel, iterations=1)
            
            return dilated
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame

    def extract_text(self, frame):
        """
        Use Tesseract to extract text from the given frame.
        Returns list of tuples: (text, position, confidence)
        """
        if not self.is_initialized():
            logger.error("OCR processor not properly initialized")
            return []
        
        if frame is None:
            logger.error("Input frame is None")
            return []
        
        try:
            self.frame_count += 1
            logger.info(f"Starting text extraction from frame {self.frame_count}")
            
            # Validate frame dimensions
            if len(frame.shape) != 3:
                logger.error(f"Invalid frame shape: {frame.shape}, expected 3D array")
                return []
            
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                logger.error(f"Invalid frame dimensions: {width}x{height}")
                return []
            
            logger.info(f"Processing frame with dimensions: {width}x{height}")
            
            # Preprocess frame for better OCR
            processed_frame = self.preprocess_frame(frame)
            
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(processed_frame)
            
            # Run Tesseract OCR with detailed output
            try:
                # Get detailed data including bounding boxes and confidence
                data = pytesseract.image_to_data(
                    pil_image, 
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                logger.info(f"Tesseract processed frame successfully")
            except Exception as ocr_error:
                logger.error(f"Tesseract OCR processing failed: {ocr_error}")
                return []
            
            # Extract and filter text results
            extracted_data = []
            
            for i in range(len(data['text'])):
                try:
                    text = data['text'][i].strip()
                    confidence = float(data['conf'][i])
                    
                    # Skip empty text or very low confidence
                    if not text or confidence < 30:  # Tesseract confidence is 0-100
                        continue
                    
                    # Get bounding box coordinates
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # Calculate center point for text placement
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Validate coordinates
                    if x < 0 or y < 0 or x >= width or y >= height:
                        continue
                    
                    # Filter text content
                    if self._should_skip_text(text):
                        logger.info(f"Skipping filtered text: '{text}' (confidence: {confidence:.1f})")
                        continue
                    
                    # Check for duplicates with improved logic
                    if self._is_duplicate_detection(text, (center_x, center_y)):
                        logger.info(f"Skipping duplicate text: '{text}' at ({center_x}, {center_y})")
                        continue
                    
                    # Convert confidence to 0-1 scale for consistency
                    normalized_confidence = confidence / 100.0
                    
                    extracted_data.append((text, (center_x, center_y), normalized_confidence))
                    logger.info(f"Extracted: '{text}' at pixels ({center_x}, {center_y}) with confidence {normalized_confidence:.2f}")
                    
                except Exception as item_error:
                    logger.error(f"Error processing OCR result at index {i}: {item_error}")
                    continue
            
            # Update text tracking
            self._update_text_tracking(extracted_data)
            
            logger.info(f"Successfully extracted {len(extracted_data)} text regions")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Unexpected error in extract_text: {e}")
            return []

    def _should_skip_text(self, text):
        """
        Check if text should be skipped based on content filtering.
        """
        if not text or len(text) < 2:
            return True
        
        text_lower = text.lower()
        
        # Skip common UI elements and labels
        skip_patterns = [
            'text box', 'box', 'bounding', 'frame', 'camera', 'ocr',
            'px', 'pixels', 'width', 'height', 'dimensions',
            'extracted', 'saved', 'accumulated', 'chars',
            'unique', 'confidence', 'position', 'coordinates'
        ]
        
        for pattern in skip_patterns:
            if pattern in text_lower:
                return True
        
        # Skip text with excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        # Skip very short text or mostly numeric text
        if len(text) <= 1 or (len(text) <= 3 and not any(c.isalpha() for c in text)):
            return True
        
        return False

    def _is_duplicate_detection(self, text, position):
        """
        Check if text is a duplicate based on content and position.
        """
        text_key = text.strip().lower()
        
        # Check if we've seen this text recently
        if text_key in self.previous_texts:
            # Check if position is close to previous detection
            if text_key in self.text_confidence_cache:
                prev_pos = self.text_confidence_cache[text_key]['position']
                distance = ((position[0] - prev_pos[0])**2 + (position[1] - prev_pos[1])**2)**0.5
                
                # If text is detected close to previous position, it's likely a duplicate
                if distance < 100:  # 100 pixel tolerance
                    return True
        
        return False

    def _update_text_tracking(self, extracted_data):
        """
        Update text tracking for duplicate prevention.
        """
        current_texts = set()
        
        for text, position, confidence in extracted_data:
            text_key = text.strip().lower()
            current_texts.add(text_key)
            
            # Update confidence cache
            if text_key not in self.text_confidence_cache or confidence > self.text_confidence_cache[text_key]['confidence']:
                self.text_confidence_cache[text_key] = {
                    'position': position,
                    'confidence': confidence,
                    'frame_count': self.frame_count
                }
        
        # Clean up old entries (older than 30 frames)
        cutoff_frame = self.frame_count - 30
        keys_to_remove = []
        for text_key, data in self.text_confidence_cache.items():
            if data['frame_count'] < cutoff_frame:
                keys_to_remove.append(text_key)
        
        for key in keys_to_remove:
            self.text_confidence_cache.pop(key, None)
        
        self.previous_texts = current_texts
