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
            
            # Configure Tesseract for better performance and text coherence
            # PSM 6: Uniform block of text (good for paragraphs)
            # PSM 8: Single word (alternative for individual words)
            # PSM 13: Raw line (good for line-by-line processing)
            self.tesseract_config = '--oem 3 --psm 6'
            
            # Alternative configs for different text layouts
            self.configs = {
                'paragraph': '--oem 3 --psm 6',  # Uniform block
                'line': '--oem 3 --psm 13',      # Raw line
                'word': '--oem 3 --psm 8'        # Single word
            }
            
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
        Enhanced with noise reduction and better text preservation.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply stronger Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply median blur to remove salt and pepper noise
            median_blurred = cv2.medianBlur(blurred, 3)
            
            # Apply bilateral filter to preserve edges while smoothing
            bilateral = cv2.bilateralFilter(median_blurred, 9, 75, 75)
            
            # Try multiple preprocessing approaches for better results
            
            # Method 1: Adaptive thresholding with better parameters
            thresh1 = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
            )
            
            # Method 2: Otsu's thresholding
            _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 3: Simple thresholding with higher threshold
            _, thresh3 = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            cleaned2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
            cleaned3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)
            
            # Apply erosion to remove small noise
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            eroded1 = cv2.erode(cleaned1, kernel_erode, iterations=1)
            eroded2 = cv2.erode(cleaned2, kernel_erode, iterations=1)
            eroded3 = cv2.erode(cleaned3, kernel_erode, iterations=1)
            
            # Apply slight dilation to make text more readable
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            dilated1 = cv2.dilate(eroded1, kernel_dilate, iterations=1)
            dilated2 = cv2.dilate(eroded2, kernel_dilate, iterations=1)
            dilated3 = cv2.dilate(eroded3, kernel_dilate, iterations=1)
            
            # Return the best preprocessed image (start with adaptive thresholding)
            return dilated1
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame

    def extract_text(self, frame):
        """
        Use Tesseract to extract text from the given frame with improved semantic coherence.
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
            
            # Try multiple preprocessing approaches if the first one doesn't yield good results
            if not self._test_ocr_quality(pil_image):
                logger.info("First preprocessing didn't yield good results, trying alternative approaches...")
                pil_image = self._try_alternative_preprocessing(frame)
            
            # Try multiple OCR approaches for better text coherence
            extracted_data = []
            
            # First, try to extract coherent text blocks
            coherent_text = self._extract_coherent_text(pil_image, width, height)
            if coherent_text:
                extracted_data.extend(coherent_text)
                logger.info(f"Extracted {len(coherent_text)} coherent text blocks")
            
            # If no coherent text found, fall back to word-by-word extraction
            if not extracted_data:
                logger.info("No coherent text blocks found, falling back to word extraction")
                extracted_data = self._extract_word_by_word(pil_image, width, height)
            
            # Clean up extracted data to remove garbage
            cleaned_data = self._clean_extracted_data(extracted_data)
            
            # Update text tracking
            self._update_text_tracking(cleaned_data)
            
            logger.info(f"Successfully extracted {len(cleaned_data)} text regions after cleaning")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Unexpected error in extract_text: {e}")
            return []

    def _extract_coherent_text(self, pil_image, width, height):
        """
        Extract coherent text blocks that maintain semantic meaning.
        """
        try:
            # Try multiple OCR approaches for better text extraction
            extracted_data = []
            
            # Method 1: Try simple text extraction first (best for clean text)
            try:
                simple_text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 6')
                if simple_text and simple_text.strip():
                    # Split by lines and process each line
                    lines = simple_text.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 5:  # Increased minimum length for meaningful lines
                            # Calculate approximate center position
                            center_x = width // 2
                            center_y = height // 2
                            
                            # Clean the line before processing
                            cleaned_line = self._clean_text(line)
                            if cleaned_line and len(cleaned_line) >= 5:
                                if not self._should_skip_text(cleaned_line):
                                    # Calculate confidence based on text quality
                                    confidence = self._calculate_text_confidence(cleaned_line, 0.8)
                                    if confidence > 0.5:  # Only high-quality text
                                        extracted_data.append((cleaned_line, (center_x, center_y), confidence))
                                        logger.info(f"Extracted simple text: '{cleaned_line}' (confidence: {confidence:.2f})")
                logger.info("Simple text extraction completed")
            except Exception as e:
                logger.warning(f"Simple text extraction failed: {e}")
            
            # Method 2: If simple extraction didn't work well, try detailed data extraction
            if not extracted_data:
                logger.info("Trying detailed text extraction...")
                data = pytesseract.image_to_data(
                    pil_image, 
                    config=self.configs['paragraph'],
                    output_type=pytesseract.Output.DICT
                )
                
                # Group text by lines to maintain coherence
                lines = {}
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    confidence = float(data['conf'][i])
                    
                    if not text or confidence < 30:
                        continue
                    
                    # Get line information
                    line_num = data['line_num'][i]
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # Group by line number
                    if line_num not in lines:
                        lines[line_num] = []
                    
                    lines[line_num].append({
                        'text': text,
                        'confidence': confidence,
                        'x': x, 'y': y, 'w': w, 'h': h
                    })
                
                # Process each line as a coherent text block
                for line_num, line_items in lines.items():
                    if not line_items:
                        continue
                    
                    # Sort items by x position to maintain reading order
                    line_items.sort(key=lambda item: item['x'])
                    
                    # Combine text in the line
                    line_text = ' '.join(item['text'] for item in line_items)
                    
                    # Only process meaningful text blocks
                    if len(line_text.strip()) < 5:
                        continue
                    
                    # Calculate average confidence and position
                    avg_confidence = sum(item['confidence'] for item in line_items) / len(line_items)
                    center_x = sum(item['x'] + item['w'] // 2 for item in line_items) / len(line_items)
                    center_y = sum(item['y'] + item['h'] // 2 for item in line_items) / len(line_items)
                    
                    # Filter coherent text
                    if self._should_skip_text(line_text):
                        logger.info(f"Skipping filtered coherent text: '{line_text}'")
                        continue
                    
                    # Check for duplicates
                    if self._is_duplicate_detection(line_text, (center_x, center_y)):
                        logger.info(f"Skipping duplicate coherent text: '{line_text}'")
                        continue
                    
                    normalized_confidence = avg_confidence / 100.0
                    extracted_data.append((line_text, (center_x, center_y), normalized_confidence))
                    logger.info(f"Extracted coherent text: '{line_text}' at ({center_x:.1f}, {center_y:.1f}) with confidence {normalized_confidence:.2f}")
            
            # Method 3: If still no good results, try grouping nearby words
            if not extracted_data:
                logger.info("Trying word grouping...")
                data = pytesseract.image_to_data(
                    pil_image, 
                    config=self.configs['paragraph'],
                    output_type=pytesseract.Output.DICT
                )
                extracted_data = self._group_nearby_words(data, width, height)
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting coherent text: {e}")
            return []

    def _group_nearby_words(self, data, width, height):
        """
        Group nearby words into coherent phrases when line-based grouping fails.
        """
        try:
            # Extract all valid words with their positions
            words = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                if not text or confidence < 30:
                    continue
                
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                words.append({
                    'text': text,
                    'confidence': confidence,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2
                })
            
            if not words:
                return []
            
            # Group words that are close to each other
            groups = []
            used_words = set()
            
            for i, word in enumerate(words):
                if i in used_words:
                    continue
                
                # Start a new group
                group = [word]
                used_words.add(i)
                
                # Find nearby words
                for j, other_word in enumerate(words):
                    if j in used_words:
                        continue
                    
                    # Calculate distance between word centers
                    distance = ((word['center_x'] - other_word['center_x'])**2 + 
                               (word['center_y'] - other_word['center_y'])**2)**0.5
                    
                    # Group words within 100 pixels of each other
                    if distance < 100:
                        group.append(other_word)
                        used_words.add(j)
                
                # Sort group by x position for reading order
                group.sort(key=lambda w: w['x'])
                
                # Combine words in group
                if len(group) > 1:
                    group_text = ' '.join(w['text'] for w in group)
                    avg_confidence = sum(w['confidence'] for w in group) / len(group)
                    center_x = sum(w['center_x'] for w in group) / len(group)
                    center_y = sum(w['center_y'] for w in group) / len(group)
                    
                    # Filter and add to results
                    if not self._should_skip_text(group_text):
                        normalized_confidence = avg_confidence / 100.0
                        groups.append((group_text, (center_x, center_y), normalized_confidence))
                        logger.info(f"Grouped nearby words: '{group_text}' at ({center_x:.1f}, {center_y:.1f})")
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping nearby words: {e}")
            return []

    def _extract_word_by_word(self, pil_image, width, height):
        """
        Fallback method to extract text word by word when coherent blocks aren't found.
        """
        try:
            # Use word mode for individual word extraction
            data = pytesseract.image_to_data(
                pil_image, 
                config=self.configs['word'],
                output_type=pytesseract.Output.DICT
            )
            
            extracted_data = []
            
            for i in range(len(data['text'])):
                try:
                    text = data['text'][i].strip()
                    confidence = float(data['conf'][i])
                    
                    # Skip empty text or very low confidence
                    if not text or confidence < 30:
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
                    logger.info(f"Extracted word: '{text}' at pixels ({center_x}, {center_y}) with confidence {normalized_confidence:.2f}")
                    
                except Exception as item_error:
                    logger.error(f"Error processing OCR result at index {i}: {item_error}")
                    continue
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting word by word: {e}")
            return []

    def _test_ocr_quality(self, pil_image):
        """
        Test if the preprocessed image yields good OCR results.
        """
        try:
            # Quick test with simple text extraction
            test_text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 6')
            if test_text and len(test_text.strip()) > 10:
                return True
            return False
        except Exception as e:
            logger.error(f"Error testing OCR quality: {e}")
            return False

    def _try_alternative_preprocessing(self, frame):
        """
        Try alternative preprocessing methods if the first one doesn't work well.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Otsu's thresholding
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 2: Simple thresholding
            _, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Method 3: Adaptive thresholding with different parameters
            thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
            
            # Test each method
            methods = [thresh1, thresh2, thresh3]
            best_image = None
            best_score = 0
            
            for i, method in enumerate(methods):
                try:
                    pil_method = Image.fromarray(method)
                    test_text = pytesseract.image_to_string(pil_method, config='--oem 3 --psm 6')
                    if test_text and len(test_text.strip()) > 10:
                        score = len(test_text.strip())
                        if score > best_score:
                            best_score = score
                            best_image = pil_method
                            logger.info(f"Alternative preprocessing method {i+1} yielded better results")
                except Exception as e:
                    logger.warning(f"Alternative preprocessing method {i+1} failed: {e}")
            
            if best_image:
                return best_image
            else:
                # Fallback to original preprocessing
                return Image.fromarray(self.preprocess_frame(frame))
                
        except Exception as e:
            logger.error(f"Error in alternative preprocessing: {e}")
            return Image.fromarray(self.preprocess_frame(frame))

    def _clean_extracted_data(self, extracted_data):
        """
        Clean up extracted data to remove garbage and improve text quality.
        """
        cleaned_data = []
        
        for text, position, confidence in extracted_data:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            if cleaned_text and len(cleaned_text.strip()) >= 3:
                # Recalculate confidence based on text quality
                new_confidence = self._calculate_text_confidence(cleaned_text, confidence)
                
                if new_confidence > 0.3:  # Only keep text with decent confidence
                    cleaned_data.append((cleaned_text, position, new_confidence))
                    logger.info(f"Cleaned text: '{text}' -> '{cleaned_text}' (confidence: {new_confidence:.2f})")
        
        return cleaned_data

    def _clean_text(self, text):
        """
        Clean individual text strings to remove garbage characters.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove text that's mostly special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.4:
            return ""
        
        # Remove text with too many consecutive special characters
        consecutive_special = 0
        for c in text:
            if not c.isalnum() and not c.isspace():
                consecutive_special += 1
                if consecutive_special > 2:
                    return ""
            else:
                consecutive_special = 0
        
        # Remove text that doesn't have enough alphabetic characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.4:  # At least 40% should be alphabetic
            return ""
        
        # Remove common garbage patterns
        garbage_patterns = [
            r'^[^a-zA-Z]*$',  # No alphabetic characters
            r'^[0-9\s]*$',    # Only numbers and spaces
            r'^[^\w\s]*$',     # Only special characters
            r'^[|\\/_\-\+]*$', # Only common garbage chars
        ]
        
        for pattern in garbage_patterns:
            if re.match(pattern, text):
                return ""
        
        return text.strip()

    def _calculate_text_confidence(self, text, original_confidence):
        """
        Calculate confidence based on text quality and original confidence.
        """
        if not text:
            return 0.0
        
        # Base confidence on original
        confidence = original_confidence
        
        # Boost confidence for longer, cleaner text
        if len(text) > 10:
            confidence += 0.1
        
        # Boost confidence for text with good alphabetic ratio
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio > 0.7:
            confidence += 0.1
        
        # Reduce confidence for text with too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.2:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))

    def _should_skip_text(self, text):
        """
        Check if text should be skipped based on content filtering.
        Much more aggressive filtering to remove garbage characters.
        """
        if not text or len(text) < 3:  # Increased minimum length
            return True
        
        text_lower = text.lower()
        
        # Skip common UI elements and labels
        skip_patterns = [
            'text box', 'bounding box', 'frame', 'camera', 'ocr',
            'px', 'pixels', 'width', 'height', 'dimensions',
            'extracted', 'saved', 'accumulated', 'chars',
            'unique', 'confidence', 'position', 'coordinates',
            'toxt box', 'toxt', 'text b', 'text.', 'text,'
        ]
        
        for pattern in skip_patterns:
            if pattern in text_lower:
                return True
        
        # Skip text with excessive special characters (much more aggressive)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:  # Much lower threshold - only 30% special chars allowed
            return True
        
        # Skip text that's mostly numbers and symbols
        if len(text) <= 5 and not any(c.isalpha() for c in text):
            return True
        
        # Skip text that's just punctuation or symbols
        if all(not c.isalnum() for c in text):
            return True
        
        # Skip single characters or very short fragments
        if len(text.strip()) <= 2:
            return True
        
        # Skip text that's mostly special characters
        if len(text.strip()) <= 5 and special_char_ratio > 0.3:
            return True
        
        # Skip text with too many consecutive special characters
        consecutive_special = 0
        for c in text:
            if not c.isalnum() and not c.isspace():
                consecutive_special += 1
                if consecutive_special > 2:  # More than 2 consecutive special chars
                    return True
            else:
                consecutive_special = 0
        
        # Skip text that doesn't contain enough alphabetic characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.3:  # At least 30% should be alphabetic
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
