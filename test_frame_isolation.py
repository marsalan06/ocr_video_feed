import cv2
import pytesseract
import numpy as np
import os
import threading
import logging
from time import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameIsolationTester:
    def __init__(self, video_source=0, output_file="test_extracted_text.txt"):
        """
        Initialize the frame isolation tester with improved text extraction.
        
        Args:
            video_source: Camera source (0 for default camera)
            output_file: Path to save extracted text
        """
        self.video_source = video_source
        self.output_file = output_file
        self.previous_text = set()
        self.frame_buffer = []
        self.buffer_size = 5
        self.stable_frame_threshold = 0.95
        self.last_stable_frame = None
        self.frames_since_last_stable = 0
        self.frame_count = 0
        self.frame_skip = 3  # Process every 3rd frame for efficiency
        
        # Initialize Tesseract
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise
        
        # Setup output file
        self.setup_output_file()
    
    def setup_output_file(self):
        """Setup the output file with session header."""
        try:
            with open(self.output_file, 'w') as f:
                f.write(f"=== Frame Isolation Test Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            logger.info(f"Output file setup: {self.output_file}")
        except Exception as e:
            logger.error(f"Error setting up output file: {e}")
    
    def preprocess_frame_gentle(self, frame):
        """
        Gentle preprocessing for stable frames that preserves text details.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply light Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply bilateral filter to preserve edges while smoothing
            bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
            
            # Try multiple thresholding approaches
            # Method 1: Adaptive thresholding with gentle parameters
            thresh1 = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Method 2: Otsu's thresholding
            _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 3: Simple thresholding with moderate threshold
            _, thresh3 = cv2.threshold(bilateral, 127, 255, cv2.THRESH_BINARY)
            
            # Test which method works best
            methods = [thresh1, thresh2, thresh3]
            best_image = None
            best_score = 0
            
            for i, method in enumerate(methods):
                try:
                    test_text = pytesseract.image_to_string(method, config='--oem 3 --psm 6')
                    if test_text and len(test_text.strip()) > 5:
                        score = len(test_text.strip())
                        if score > best_score:
                            best_score = score
                            best_image = method
                            logger.debug(f"Preprocessing method {i+1} yielded best results")
                except Exception as e:
                    logger.warning(f"Preprocessing method {i+1} failed: {e}")
            
            if best_image is not None:
                return best_image
            else:
                # Fallback to original preprocessing
                return self.preprocess_frame_aggressive(frame)
                
        except Exception as e:
            logger.error(f"Error in gentle preprocessing: {e}")
            return frame
    
    def preprocess_frame_aggressive(self, frame):
        """
        Aggressive preprocessing for comparison (original method).
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
            
            # Adaptive thresholding with better parameters
            thresh = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
            )
            
            # Apply morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Apply erosion to remove small noise
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            eroded = cv2.erode(cleaned, kernel_erode, iterations=1)
            
            # Apply slight dilation to make text more readable
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)
            
            return dilated
            
        except Exception as e:
            logger.error(f"Error in aggressive preprocessing: {e}")
            return frame
    
    def is_frame_stable(self, frame_buffer, threshold=0.95):
        """
        Check if frames in the buffer are stable (similar to each other).
        Returns True if frames are stable enough for text extraction.
        """
        if len(frame_buffer) < 2:
            return False
        
        try:
            # Convert all frames to grayscale for comparison
            gray_frames = []
            for frame in frame_buffer:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to smaller size for faster comparison
                resized = cv2.resize(gray, (128, 128))
                gray_frames.append(resized)
            
            # Calculate similarity between consecutive frames
            similarities = []
            for i in range(len(gray_frames) - 1):
                similarity = self._calculate_frame_similarity(gray_frames[i], gray_frames[i + 1])
                similarities.append(similarity)
            
            # Check if all similarities are above threshold
            avg_similarity = sum(similarities) / len(similarities)
            is_stable = avg_similarity >= threshold
            
            logger.info(f"Frame stability: avg similarity {avg_similarity:.3f} (threshold: {threshold}) - {'STABLE' if is_stable else 'UNSTABLE'}")
            
            return is_stable
            
        except Exception as e:
            logger.error(f"Error checking frame stability: {e}")
            return False
    
    def _calculate_frame_similarity(self, frame1, frame2):
        """
        Calculate simple similarity between two grayscale frames.
        Returns a value between 0.0 and 1.0.
        """
        try:
            # Calculate mean squared error
            mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
            max_mse = 255 ** 2  # Maximum possible MSE
            similarity = 1.0 - (mse / max_mse)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Error calculating frame similarity: {e}")
            return 0.0
    
    def frames_are_different(self, frame1, frame2, threshold=0.8):
        """
        Check if two frames are significantly different.
        Returns True if frames are different enough to warrant processing.
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Resize for faster comparison
            gray1_resized = cv2.resize(gray1, (128, 128))
            gray2_resized = cv2.resize(gray2, (128, 128))
            
            # Calculate similarity
            similarity = self._calculate_frame_similarity(gray1_resized, gray2_resized)
            
            # Return True if frames are different (similarity below threshold)
            return similarity < threshold
            
        except Exception as e:
            logger.error(f"Error comparing frames: {e}")
            return True  # Default to processing if comparison fails
    
    def clean_text_stable(self, text):
        """
        Clean individual text strings for stable frames with less aggressive filtering.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove text that's mostly special characters (less aggressive)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:  # More lenient threshold
            return ""
        
        # Remove text with too many consecutive special characters
        consecutive_special = 0
        for c in text:
            if not c.isalnum() and not c.isspace():
                consecutive_special += 1
                if consecutive_special > 3:  # More lenient threshold
                    return ""
            else:
                consecutive_special = 0
        
        # Remove text that doesn't have enough alphabetic characters (less aggressive)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.2:  # Lower threshold for stable frames
            return ""
        
        # Remove common garbage patterns
        garbage_patterns = [
            r'^[^a-zA-Z]*$',  # No alphabetic characters
            r'^[0-9\s]*$',    # Only numbers and spaces
            r'^[^\w\s]*$',     # Only special characters
        ]
        
        import re
        for pattern in garbage_patterns:
            if re.match(pattern, text):
                return ""
        
        return text.strip()
    
    def should_skip_text_stable(self, text):
        """
        Check if text should be skipped for stable frames with less aggressive filtering.
        """
        if not text or len(text) < 2:  # Lower minimum length
            return True
        
        text_lower = text.lower()
        
        # Skip common UI elements and labels (less aggressive)
        skip_patterns = [
            'text box', 'bounding box', 'frame', 'camera', 'ocr',
            'px', 'pixels', 'width', 'height', 'dimensions',
            'extracted', 'saved', 'accumulated', 'chars',
            'unique', 'confidence', 'position', 'coordinates'
        ]
        
        for pattern in skip_patterns:
            if pattern in text_lower:
                return True
        
        # Skip text with excessive special characters (less aggressive)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.4:  # More lenient threshold
            return True
        
        # Skip text that's just punctuation or symbols
        if all(not c.isalnum() for c in text):
            return True
        
        # Skip single characters
        if len(text.strip()) <= 1:
            return True
        
        # Skip text that doesn't contain enough alphabetic characters (less aggressive)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.2:  # Lower threshold for stable frames
            return True
        
        return False
    
    def extract_text_from_stable_frame(self, frame):
        """
        Extract text from a stable frame with improved preprocessing.
        """
        try:
            # Use gentle preprocessing for stable frames
            processed_frame = self.preprocess_frame_gentle(frame)
            
            # Extract text with multiple approaches
            extracted_texts = []
            
            # Method 1: Simple text extraction
            try:
                simple_text = pytesseract.image_to_string(processed_frame, config='--oem 3 --psm 6')
                if simple_text and simple_text.strip():
                    lines = simple_text.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 2:
                            cleaned_line = self.clean_text_stable(line)
                            if cleaned_line and not self.should_skip_text_stable(cleaned_line):
                                extracted_texts.append(cleaned_line)
                                logger.info(f"Extracted stable text: '{cleaned_line}'")
            except Exception as e:
                logger.warning(f"Simple text extraction failed: {e}")
            
            # Method 2: Detailed data extraction if simple extraction didn't work well
            if not extracted_texts:
                try:
                    data = pytesseract.image_to_data(processed_frame, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
                    
                    # Group text by lines
                    lines = {}
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        confidence = float(data['conf'][i])
                        
                        if not text or confidence < 20:  # Lower threshold for stable frames
                            continue
                        
                        line_num = data['line_num'][i]
                        if line_num not in lines:
                            lines[line_num] = []
                        
                        lines[line_num].append(text)
                    
                    # Process each line
                    for line_num, line_items in lines.items():
                        if not line_items:
                            continue
                        
                        line_text = ' '.join(line_items)
                        if len(line_text.strip()) >= 2:
                            cleaned_text = self.clean_text_stable(line_text)
                            if cleaned_text and not self.should_skip_text_stable(cleaned_text):
                                extracted_texts.append(cleaned_text)
                                logger.info(f"Extracted stable line: '{cleaned_text}'")
                except Exception as e:
                    logger.warning(f"Detailed text extraction failed: {e}")
            
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Error extracting text from stable frame: {e}")
            return []
    
    def save_text_to_file(self, text, method="stable"):
        """Save unique text to the output file with method indicator."""
        if text and text not in self.previous_text:
            try:
                timestamp = datetime.now().strftime('%H:%M:%S')
                with open(self.output_file, "a", encoding='utf-8') as f:
                    f.write(f"[{timestamp}] [{method.upper()}] {text}\n")
                self.previous_text.add(text)
                logger.info(f"Saved new text [{method}]: '{text}'")
                return True
            except Exception as e:
                logger.error(f"Error saving text to file: {e}")
        return False
    
    def process_frame_with_isolation(self, frame):
        """
        Process frame with frame isolation logic.
        """
        self.frame_count += 1
        
        # Only process every few frames for efficiency
        if self.frame_count % self.frame_skip != 0:
            return
        
        # Add frame to buffer for stabilization
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Check if we have enough frames for stabilization
        if len(self.frame_buffer) >= self.buffer_size:
            # Calculate frame stability
            is_stable = self.is_frame_stable(self.frame_buffer, self.stable_frame_threshold)
            
            if is_stable:
                # Use the middle frame from the buffer as the stable frame
                stable_frame = self.frame_buffer[self.buffer_size // 2]
                self.frames_since_last_stable = 0
                
                # Only process if this is a new stable frame
                if self.last_stable_frame is None or self.frames_are_different(stable_frame, self.last_stable_frame):
                    self.last_stable_frame = stable_frame.copy()
                    logger.info(f"Processing stable frame {self.frame_count}")
                    
                    # Extract text from stable frame
                    extracted_texts = self.extract_text_from_stable_frame(stable_frame)
                    
                    # Save each extracted text
                    for text in extracted_texts:
                        self.save_text_to_file(text, "stable")
                else:
                    logger.info("Frame is stable but similar to last processed frame - skipping")
            else:
                self.frames_since_last_stable += 1
                logger.info(f"Frame not stable yet ({self.frames_since_last_stable} frames since last stable)")
        else:
            # Still building frame buffer
            logger.info(f"Building frame buffer ({len(self.frame_buffer)}/{self.buffer_size})")
    
    def process_frame_original(self, frame):
        """
        Process frame with original aggressive preprocessing for comparison.
        """
        self.frame_count += 1
        
        # Only process every few frames for efficiency
        if self.frame_count % self.frame_skip != 0:
            return
        
        try:
            # Use aggressive preprocessing
            processed_frame = self.preprocess_frame_aggressive(frame)
            
            # Extract text
            text = pytesseract.image_to_string(processed_frame, config='--oem 3 --psm 6')
            
            if text and text.strip():
                # Simple cleaning for comparison
                cleaned_text = ' '.join(text.strip().split())
                if len(cleaned_text) > 2:
                    self.save_text_to_file(cleaned_text, "original")
                    
        except Exception as e:
            logger.error(f"Error in original frame processing: {e}")
    
    def video_stream_thread(self, use_isolation=True):
        """
        Capture video in a separate thread for efficient processing.
        """
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            logger.error("Could not open video stream.")
            return
        
        logger.info(f"Starting video stream processing with {'frame isolation' if use_isolation else 'original'} method")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process the frame
            if use_isolation:
                self.process_frame_with_isolation(frame)
            else:
                self.process_frame_original(frame)
            
            # Display the frame with info
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Method: {'Isolation' if use_isolation else 'Original'}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Frame Isolation Test', display_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video stream processing completed")

def main():
    """Main function to run the frame isolation test."""
    print("Frame Isolation Test for OCR Text Extraction")
    print("=" * 50)
    print("This test compares frame isolation vs original processing")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Create tester instance
    tester = FrameIsolationTester(video_source=0, output_file="test_extracted_text.txt")
    
    # Start video processing with frame isolation
    try:
        tester.video_stream_thread(use_isolation=True)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")

if __name__ == "__main__":
    main() 