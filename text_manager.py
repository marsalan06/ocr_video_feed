import logging
import cv2
import os
from datetime import datetime
from config import TEXT_HEIGHT, TEXT_MARGIN, BOX_COLOR, SINGLE_BOX_RATIOS, SAVE_TEXT_TO_FILE, TEXT_FILE_PATH, CLEAR_FILE_ON_START

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextManager:
    def __init__(self, frame_width=None, frame_height=None):
        """
        Initializes the TextManager with a single dynamic bounding box.
        frame_width, frame_height: Frame dimensions for calculating box positions.
        """
        self.accumulated_text = ""  # Initialize accumulated text
        self.bounding_box = None  # Will be calculated based on frame size
        self.current_text_y = 0  # Keeps track of vertical position for text within the box
        self.text_height = TEXT_HEIGHT  # Height allocated for each line of text
        self.margin = TEXT_MARGIN  # Margin from box edges
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_box_height = 0  # Track maximum height needed
        
        # Improved duplicate detection
        self.detected_texts = set()  # Set to track unique texts
        self.text_positions = {}  # Dictionary to track text positions
        self.position_tolerance = 150  # Increased tolerance for position matching (from 100)
        self.text_timestamps = {}  # Track when text was last detected
        self.min_time_between_detections = 3.0  # Reduced minimum seconds between same text detection (from 5.0)
        
        # Initialize text storage
        self.setup_text_storage()
        
        # Calculate bounding box if frame dimensions are provided
        if frame_width and frame_height:
            self.calculate_bounding_box(frame_width, frame_height)
        
        logger.info(f"TextManager initialized with single dynamic bounding box")
        logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
        logger.info(f"Text storage enabled: {SAVE_TEXT_TO_FILE}")
    
    def setup_text_storage(self):
        """Setup text storage functionality"""
        if SAVE_TEXT_TO_FILE:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(TEXT_FILE_PATH) if os.path.dirname(TEXT_FILE_PATH) else '.', exist_ok=True)
                
                # Clear file on start if configured
                if CLEAR_FILE_ON_START:
                    with open(TEXT_FILE_PATH, 'w') as f:
                        f.write(f"=== OCR Text Extraction Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                        logger.info(f"Text file cleared: {TEXT_FILE_PATH}")
                else:
                    # Append to existing file
                    with open(TEXT_FILE_PATH, 'a') as f:
                        f.write(f"\n=== New Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                        logger.info(f"Text file appended: {TEXT_FILE_PATH}")
                        
            except Exception as e:
                logger.error(f"Error setting up text storage: {e}")
    
    def calculate_bounding_box(self, frame_width, frame_height):
        """
        Calculate single bounding box based on frame dimensions and ratios.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = SINGLE_BOX_RATIOS
        x1 = int(x1_ratio * frame_width)
        y1 = int(y1_ratio * frame_height)
        x2 = int(x2_ratio * frame_width)
        y2 = int(y2_ratio * frame_height)
        
        self.bounding_box = (x1, y1, x2, y2)
        self.max_box_height = y2 - y1  # Initial height
        
        logger.info(f"Initial bounding box: ({x1}, {y1}, {x2}, {y2}) - {x2-x1}x{y2-y1}")
    
    def update_frame_dimensions(self, frame_width, frame_height):
        """
        Update frame dimensions and recalculate bounding box.
        """
        if frame_width != self.frame_width or frame_height != self.frame_height:
            logger.info(f"Updating frame dimensions: {frame_width}x{frame_height}")
            self.calculate_bounding_box(frame_width, frame_height)
            # Reset text placement when frame size changes
            self.current_text_y = 0
    
    def expand_box_if_needed(self, required_height):
        """
        Expand the bounding box if more height is needed, with a maximum limit.
        """
        if self.bounding_box is None:
            return
        
        x1, y1, x2, y2 = self.bounding_box
        current_height = y2 - y1
        needed_height = required_height + (2 * self.margin)
        
        # Set maximum box height to 70% of frame height (increased from 50%)
        max_allowed_height = int(self.frame_height * 0.7)
        
        if needed_height > current_height:
            # Expand the box downward, but respect maximum height
            new_y2 = min(y1 + needed_height, y1 + max_allowed_height)
            
            # Don't expand beyond frame bounds
            new_y2 = min(new_y2, self.frame_height - self.margin)
            
            if new_y2 > y2:  # Only expand if we actually need more space
                self.bounding_box = (x1, y1, x2, new_y2)
                self.max_box_height = new_y2 - y1
                logger.info(f"Expanded bounding box height to {self.max_box_height}px (max allowed: {max_allowed_height}px)")
            else:
                logger.info(f"Box expansion not needed or limited by max height")
        else:
            logger.info(f"Box height sufficient: {current_height}px >= {needed_height}px needed")
    
    def is_duplicate_text(self, text, position):
        """
        Check if text is a duplicate based on content and position with improved logic.
        """
        text_key = text.strip().lower()
        current_time = datetime.now().timestamp()
        
        # Check if we've seen this text before
        if text_key in self.detected_texts:
            # Check if position is close to previous detection
            if text_key in self.text_positions:
                prev_pos = self.text_positions[text_key]
                distance = ((position[0] - prev_pos[0])**2 + (position[1] - prev_pos[1])**2)**0.5
                
                # Check time since last detection
                last_detection_time = self.text_timestamps.get(text_key, 0)
                time_since_last = current_time - last_detection_time
                
                # Less strict duplicate detection
                if distance < self.position_tolerance and time_since_last < self.min_time_between_detections:
                    logger.info(f"Duplicate text detected: '{text}' at pixels ({position[0]:.1f}, {position[1]:.1f}) - distance: {distance:.1f}px, time: {time_since_last:.1f}s")
                    return True
        
        return False
    
    def is_valid_text(self, text):
        """
        Check if text is valid content (not UI elements or labels) with improved filtering.
        """
        if not text or not text.strip():
            return False
        
        text_lower = text.strip().lower()
        
        # Filter out UI elements and labels (less aggressive)
        invalid_patterns = [
            'text box', 'box', 'bounding', 'frame', 'camera', 'ocr',
            'px', 'pixels', 'width', 'height', 'dimensions',
            'extracted', 'saved', 'accumulated', 'chars',
            'unique', 'confidence', 'position', 'coordinates', 'toxt', 'bov'
        ]
        
        for pattern in invalid_patterns:
            if pattern in text_lower:
                logger.info(f"Filtered out UI/text: '{text}' (matched pattern: {pattern})")
                return False
        
        # Filter out very short text (likely noise)
        if len(text.strip()) < 2:
            logger.info(f"Filtered out short text: '{text}' (length: {len(text.strip())})")
            return False
        
        # Filter out text that's mostly numbers and symbols
        if len(text.strip()) <= 3 and not any(c.isalpha() for c in text):
            logger.info(f"Filtered out numeric/symbol text: '{text}'")
            return False
        
        # Filter out common OCR artifacts (less aggressive)
        if any(char in text for char in ['_', '|', '\\', '/']):
            logger.info(f"Filtered out text with artifacts: '{text}'")
            return False
        
        return True
    
    def add_text(self, text, original_position, frame):
        """
        Adds text to the single bounding box, expanding it if needed.
        Returns (position, text) tuple for display, or None if duplicate/no space available.
        """
        if not text or not text.strip():
            logger.warning("Attempted to add empty text")
            return None
        
        # Validate text content
        if not self.is_valid_text(text):
            return None
        
        # Check for duplicates with improved logic
        if self.is_duplicate_text(text, original_position):
            return None
        
        # Update frame dimensions if needed
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
            self.update_frame_dimensions(frame_width, frame_height)
        
        if not self.bounding_box:
            logger.error("No bounding box defined")
            return None
        
        try:
            x1, y1, x2, y2 = self.bounding_box
            
            # Calculate required height for new text
            required_height = y1 + self.margin + self.current_text_y + self.text_height
            
            # Expand box if needed
            self.expand_box_if_needed(required_height - y1)
            
            # Recalculate position after potential expansion
            x1, y1, x2, y2 = self.bounding_box
            
            # Check if we're still within frame bounds
            if required_height > self.frame_height - self.margin:
                logger.warning("Text exceeds frame bounds, cannot add more text")
                return None
            
            # Calculate text position within the bounding box
            text_x = x1 + self.margin
            text_y = y1 + self.margin + self.current_text_y
            
            # Validate position is within frame bounds
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
                if text_x < 0 or text_y < 0 or text_x >= frame_width or text_y >= frame_height:
                    logger.warning(f"Text position ({text_x}, {text_y}) out of frame bounds ({frame_width}x{frame_height})")
                    # Clamp to frame bounds
                    text_x = max(0, min(text_x, frame_width - 1))
                    text_y = max(0, min(text_y, frame_height - 1))
            
            # Check if accumulated text is getting too long (limit to 500 characters)
            max_accumulated_length = 500
            if len(self.accumulated_text) > max_accumulated_length:
                logger.info(f"Accumulated text too long ({len(self.accumulated_text)} chars), clearing old text")
                self.accumulated_text = ""
                self.current_text_y = 0
            
            # Add text to accumulated text
            if self.accumulated_text:
                self.accumulated_text += " " + text.strip()
            else:
                self.accumulated_text = text.strip()
            
            # Track this text as detected with timestamp
            text_key = text.strip().lower()
            self.detected_texts.add(text_key)
            self.text_positions[text_key] = original_position
            self.text_timestamps[text_key] = datetime.now().timestamp()
            
            # Save text to file if enabled
            self.save_text_to_file(text.strip())
            
            # Update position for next text
            self.current_text_y += self.text_height
            
            logger.info(f"Added new text: '{text}' at original pixels ({original_position[0]:.1f}, {original_position[1]:.1f}) -> display position ({text_x:.1f}, {text_y:.1f}) in expanding box")
            return (text_x, text_y), text.strip()
            
        except Exception as e:
            logger.error(f"Error adding text '{text}': {e}")
            return None

    def save_text_to_file(self, text):
        """Save text to file if enabled"""
        if SAVE_TEXT_TO_FILE and text:
            try:
                with open(TEXT_FILE_PATH, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    f.write(f"[{timestamp}] {text}\n")
            except Exception as e:
                logger.error(f"Error saving text to file: {e}")

    def get_accumulated_text(self):
        """
        Returns the currently accumulated text in sequence.
        """
        return self.accumulated_text

    def clear_text(self):
        """
        Clears the accumulated text and resets tracking structures.
        """
        logger.info("Clearing accumulated text")
        self.accumulated_text = ""
        self.current_text_y = 0
        self.detected_texts.clear()
        self.text_positions.clear()
        self.text_timestamps.clear()  # Clear timestamps too
        
        # Reset box to initial size
        if self.frame_width and self.frame_height:
            self.calculate_bounding_box(self.frame_width, self.frame_height)
        
        # Clear file if configured
        if SAVE_TEXT_TO_FILE and CLEAR_FILE_ON_START:
            try:
                with open(TEXT_FILE_PATH, 'w') as f:
                    f.write(f"=== OCR Text Extraction Session Cleared: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                logger.info("Text file cleared")
            except Exception as e:
                logger.error(f"Error clearing text file: {e}")

    def auto_cleanup_old_texts(self):
        """
        Automatically cleanup old text detections to prevent memory buildup.
        """
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - 30.0  # Remove texts older than 30 seconds
        
        texts_to_remove = []
        for text_key, timestamp in self.text_timestamps.items():
            if timestamp < cutoff_time:
                texts_to_remove.append(text_key)
        
        for text_key in texts_to_remove:
            self.detected_texts.discard(text_key)
            self.text_positions.pop(text_key, None)
            self.text_timestamps.pop(text_key, None)
        
        if texts_to_remove:
            logger.info(f"Cleaned up {len(texts_to_remove)} old text detections")

    def draw_bounding_box(self, frame):
        """
        Draws the single bounding box on the frame for visualization.
        """
        if frame is None or self.bounding_box is None:
            return frame
        
        # Update frame dimensions if needed
        frame_height, frame_width = frame.shape[:2]
        self.update_frame_dimensions(frame_width, frame_height)
        
        try:
            x1, y1, x2, y2 = self.bounding_box
            
            # Validate coordinates
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            
            # Add box label (but don't save this text)
            label = f"Text Box ({x2-x1}x{y2-y1})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
            
            logger.info(f"Drew single dynamic bounding box on frame")
            
        except Exception as e:
            logger.error(f"Error drawing bounding box: {e}")
        
        return frame

    def get_box_info(self):
        """
        Returns information about the current bounding box.
        """
        if self.bounding_box:
            x1, y1, x2, y2 = self.bounding_box
            return {
                'box_coordinates': self.bounding_box,
                'current_y_offset': self.current_text_y,
                'box_width': x2 - x1,
                'box_height': y2 - y1,
                'max_height': self.max_box_height
            }
        else:
            return {
                'box_coordinates': None,
                'current_y_offset': 0,
                'box_width': 0,
                'box_height': 0,
                'max_height': 0
            }
    
    def get_text_storage_info(self):
        """
        Returns information about text storage.
        """
        return {
            'save_to_file': SAVE_TEXT_TO_FILE,
            'file_path': TEXT_FILE_PATH,
            'accumulated_text_length': len(self.accumulated_text),
            'text_file_exists': os.path.exists(TEXT_FILE_PATH) if SAVE_TEXT_TO_FILE else False,
            'unique_texts_detected': len(self.detected_texts)
        } 