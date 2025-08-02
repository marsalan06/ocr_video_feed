import logging
import cv2
import os
import numpy as np
from datetime import datetime
from config import (TEXT_HEIGHT, TEXT_MARGIN, BOX_COLOR, SINGLE_BOX_RATIOS, 
                   SAVE_TEXT_TO_FILE, TEXT_FILE_PATH, CLEAR_FILE_ON_START,
                   ENABLE_FRAME_COMPARISON, FRAME_SIMILARITY_THRESHOLD, 
                   TEXT_CHANGE_THRESHOLD, MAX_STORED_FRAMES)

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
        
        # Frame comparison for change detection
        self.previous_frames = []  # Store previous frames for comparison
        self.max_stored_frames = MAX_STORED_FRAMES  # Number of previous frames to store
        self.frame_similarity_threshold = FRAME_SIMILARITY_THRESHOLD  # Threshold for frame similarity (0.0 to 1.0)
        self.text_change_threshold = TEXT_CHANGE_THRESHOLD  # Threshold for detecting text changes
        self.last_frame_hash = None  # Hash of last processed frame
        self.frame_comparison_enabled = ENABLE_FRAME_COMPARISON  # Enable/disable frame comparison
        
        # Check if scikit-image is available for advanced similarity calculations
        try:
            from skimage.metrics import structural_similarity
            self.skimage_available = True
            logger.info("Frame comparison: scikit-image available for advanced similarity calculations")
        except ImportError:
            self.skimage_available = False
            logger.warning("Frame comparison: scikit-image not available, using fallback calculations")
        
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
                distance = ((position[0] - prev_pos[0]) ** 2 + (position[1] - prev_pos[1]) ** 2) ** 0.5

                # If text is detected close to previous position, it's likely a duplicate
                if distance < self.position_tolerance:
                    logger.info(f"Duplicate text detected: '{text}' at pixels ({position[0]:.1f}, {position[1]:.1f}) - distance: {distance:.1f}px")
                    return True
                
                # Also check time-based duplicate detection
                if text_key in self.text_timestamps:
                    time_since_last = current_time - self.text_timestamps[text_key]
                    if time_since_last < self.min_time_between_detections:
                        logger.info(f"Duplicate text detected (time-based): '{text}' - {time_since_last:.1f}s since last detection")
                        return True

        return False

    
    def is_valid_text(self, text):
        """
        Check if text is valid content (not UI elements or labels) with improved filtering.
        Less aggressive to preserve semantic content.
        """
        if not text or not text.strip():
            return False
        
        text_lower = text.strip().lower()
        
        # Filter out UI elements and labels (less aggressive)
        invalid_patterns = [
            'text box', 'bounding box', 'frame', 'camera', 'ocr',
            'px', 'pixels', 'width', 'height', 'dimensions',
            'extracted', 'saved', 'accumulated', 'chars',
            'unique', 'confidence', 'position', 'coordinates'
        ]
        
        for pattern in invalid_patterns:
            if pattern in text_lower:
                logger.info(f"Filtered out UI/text: '{text}' (matched pattern: {pattern})")
                return False
        
        # Filter out very short text (less aggressive)
        if len(text.strip()) < 2:
            logger.info(f"Filtered out short text: '{text}' (length: {len(text.strip())})")
            return False
        
        # Filter out text that's mostly numbers and symbols (less aggressive)
        if len(text.strip()) <= 2 and not any(c.isalpha() for c in text):
            logger.info(f"Filtered out numeric/symbol text: '{text}'")
            return False
        
        # Filter out text that's just punctuation or symbols
        if all(not c.isalnum() for c in text):
            logger.info(f"Filtered out symbol-only text: '{text}'")
            return False
        
        # Filter out common OCR artifacts (less aggressive)
        if len(text.strip()) <= 3 and any(char in text for char in ['_', '|', '\\', '/']):
            logger.info(f"Filtered out short text with artifacts: '{text}'")
            return False
        
        return True
    
    def add_text(self, text, original_position, frame):
        """
        Adds text to the single bounding box, expanding it if needed.
        Improved to handle coherent text blocks better.
        Returns (position, text) tuple for display, or None if duplicate/no space available.
        """
        if not text or not text.strip():
            logger.warning("Attempted to add empty text")
            return None

        # Validate text content
        logger.info(f"Validating text: '{text}'")
        if not self.is_valid_text(text):
            logger.info(f"Text validation failed for: '{text}'")
            return None

        # Check for duplicates based on both content and position
        logger.info(f"Checking for duplicates: '{text}' at position {original_position}")
        if self.is_duplicate_text(text, original_position):
            logger.info(f"Duplicate text detected: '{text}'")
            return None
        logger.info(f"No duplicate detected for: '{text}'")

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

            # Check if accumulated text is getting too long (increased limit for coherent text)
            max_accumulated_length = 1000  # Increased from 500 to accommodate longer coherent text
            if len(self.accumulated_text) > max_accumulated_length:
                logger.info(f"Accumulated text too long ({len(self.accumulated_text)} chars), clearing old text")
                self.accumulated_text = ""
                self.current_text_y = 0

            # Add text to accumulated text with better formatting for coherent blocks
            if self.accumulated_text:
                # Check if the new text looks like it should be on a new line
                if len(text.strip()) > 20 or text.strip().endswith('.') or text.strip().endswith('!'):
                    # Add newline for longer text or sentences
                    self.accumulated_text += "\n" + text.strip()
                else:
                    # Add space for shorter text
                    self.accumulated_text += " " + text.strip()
            else:
                self.accumulated_text = text.strip()

            # Track this text as detected with timestamp
            text_key = text.strip().lower()
            self.detected_texts.add(text_key)
            self.text_positions[text_key] = original_position
            self.text_timestamps[text_key] = datetime.now().timestamp()

            # Save text to file if enabled
            logger.info(f"About to save text: '{text.strip()}' to file")
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
        logger.info(f"Attempting to save text to file: '{text}' (SAVE_TEXT_TO_FILE={SAVE_TEXT_TO_FILE})")
        if SAVE_TEXT_TO_FILE and text:
            try:
                with open(TEXT_FILE_PATH, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    f.write(f"[{timestamp}] {text}\n")
                logger.info(f"Successfully saved text to file: '{text}'")
            except Exception as e:
                logger.error(f"Error saving text to file: {e}")
        else:
            logger.warning(f"Not saving text: SAVE_TEXT_TO_FILE={SAVE_TEXT_TO_FILE}, text='{text}'")

    def get_accumulated_text(self):
        """
        Returns the currently accumulated text in sequence.
        """
        return self.accumulated_text

    def get_formatted_text_for_display(self):
        """
        Returns formatted text for display with proper line breaks.
        """
        if not self.accumulated_text:
            return ""
        
        # Split by newlines and format for display
        lines = self.accumulated_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():
                # Wrap long lines for better display
                if len(line) > 60:
                    # Simple word wrapping
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= 60:
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                formatted_lines.append(current_line)
                            current_line = word
                    if current_line:
                        formatted_lines.append(current_line)
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

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
    
    def calculate_frame_hash(self, frame):
        """
        Calculate a hash of the frame for quick comparison.
        """
        if frame is None:
            return None
        
        # Convert to grayscale and resize for faster hashing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))  # Resize to 64x64 for faster processing
        
        # Calculate hash using dhash (difference hash)
        diff = resized[1:, :] > resized[:-1, :]
        hash_value = sum([2**i for (i, v) in enumerate(diff.flatten()) if v])
        
        return hash_value
    
    def calculate_frame_similarity(self, frame1, frame2):
        """
        Calculate similarity between two frames using structural similarity index.
        Returns a value between 0.0 (completely different) and 1.0 (identical).
        """
        if frame1 is None or frame2 is None:
            return 0.0
        
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Resize frames to same size for comparison
            height, width = gray1.shape
            gray2_resized = cv2.resize(gray2, (width, height))
            
            # Calculate similarity based on available libraries
            if self.skimage_available:
                # Use structural similarity index if scikit-image is available
                from skimage.metrics import structural_similarity as ssim
                similarity = ssim(gray1, gray2_resized)
            else:
                # Fallback to simpler similarity calculation
                # Simple mean squared error based similarity
                mse = np.mean((gray1.astype(float) - gray2_resized.astype(float)) ** 2)
                max_mse = 255 ** 2  # Maximum possible MSE
                similarity = 1.0 - (mse / max_mse)
            
            return max(0.0, min(1.0, similarity))  # Clamp between 0.0 and 1.0
            
        except Exception as e:
            logger.error(f"Error calculating frame similarity: {e}")
            return 0.0
    
    def detect_text_changes(self, current_frame, text_regions):
        """
        Detect if text content has changed by comparing current frame with previous frames.
        Returns True if significant changes are detected.
        """
        if current_frame is None:
            return False
        
        # Calculate current frame hash
        current_hash = self.calculate_frame_hash(current_frame)
        
        # If this is the first frame, store it and return True
        if self.last_frame_hash is None:
            self.last_frame_hash = current_hash
            self.store_frame(current_frame)
            return True
        
        # Check if frame hash has changed significantly
        if current_hash != self.last_frame_hash:
            self.last_frame_hash = current_hash
            self.store_frame(current_frame)
            
            # Compare with previous frames
            if len(self.previous_frames) > 0:
                # Calculate similarity with the most recent previous frame
                similarity = self.calculate_frame_similarity(current_frame, self.previous_frames[-1])
                
                if similarity < self.frame_similarity_threshold:
                    logger.info(f"Frame change detected - similarity: {similarity:.3f} (threshold: {self.frame_similarity_threshold})")
                    return True
                else:
                    logger.info(f"Frame similar - similarity: {similarity:.3f} (threshold: {self.frame_similarity_threshold})")
                    return False
            else:
                return True
        
        return False
    
    def store_frame(self, frame):
        """
        Store frame in the frame history, maintaining max_stored_frames limit.
        """
        if frame is None:
            return
        
        # Add current frame to history
        self.previous_frames.append(frame.copy())
        
        # Remove oldest frame if we exceed the limit
        if len(self.previous_frames) > self.max_stored_frames:
            self.previous_frames.pop(0)
        
        logger.debug(f"Stored frame, history size: {len(self.previous_frames)}")
    
    def analyze_text_regions(self, frame, text_regions):
        """
        Analyze text regions to detect changes in specific areas.
        Returns True if text content has changed significantly.
        """
        if not text_regions or frame is None:
            return False
        
        try:
            # Extract regions around detected text
            text_areas = []
            for text, position, confidence in text_regions:
                x, y = position
                
                # Define region around text (adjust size as needed)
                region_size = 50
                x1 = max(0, int(x - region_size))
                y1 = max(0, int(y - region_size))
                x2 = min(frame.shape[1], int(x + region_size))
                y2 = min(frame.shape[0], int(y + region_size))
                
                # Extract region
                region = frame[y1:y2, x1:x2]
                if region.size > 0:
                    text_areas.append(region)
            
            # Compare with previous text areas if available
            if hasattr(self, 'previous_text_areas') and self.previous_text_areas:
                changes_detected = 0
                total_regions = min(len(text_areas), len(self.previous_text_areas))
                
                for i in range(total_regions):
                    similarity = self.calculate_frame_similarity(
                        text_areas[i], self.previous_text_areas[i]
                    )
                    if similarity < self.text_change_threshold:
                        changes_detected += 1
                
                change_ratio = changes_detected / total_regions if total_regions > 0 else 0
                logger.info(f"Text region change ratio: {change_ratio:.3f} ({changes_detected}/{total_regions})")
                
                # Update previous text areas
                self.previous_text_areas = text_areas
                
                return change_ratio > 0.5  # Return True if more than 50% of regions changed
            else:
                # First time, store current text areas
                self.previous_text_areas = text_areas
                return True
                
        except Exception as e:
            logger.error(f"Error analyzing text regions: {e}")
            return False
    
    def should_process_frame(self, frame, text_regions):
        """
        Determine if the frame should be processed based on change detection.
        Returns True if frame should be processed, False if it can be skipped.
        """
        if frame is None:
            return False
        
        # If frame comparison is disabled, always process
        if not self.frame_comparison_enabled:
            return True
        
        # Always process if we have text regions (this ensures new text gets captured)
        if text_regions:
            # Check if any of the text regions are new
            new_texts = 0
            for text, position, confidence in text_regions:
                text_key = text.strip().lower()
                if text_key not in self.detected_texts:
                    new_texts += 1
            
            if new_texts > 0:
                logger.info(f"Found {new_texts} new text regions - processing frame")
                return True
        
        # Check for frame-level changes (for cases where text might be in similar positions)
        frame_changed = self.detect_text_changes(frame, text_regions)
        
        # Check for text region changes
        text_changed = self.analyze_text_regions(frame, text_regions)
        
        # Process if either frame or text content has changed
        should_process = frame_changed or text_changed
        
        if should_process:
            logger.info("Frame change detected - processing text")
        else:
            logger.info("No significant changes detected - skipping text processing")
        
        return should_process
    
    def get_frame_comparison_info(self):
        """
        Returns information about frame comparison settings.
        """
        return {
            'stored_frames': len(self.previous_frames),
            'max_stored_frames': self.max_stored_frames,
            'frame_similarity_threshold': self.frame_similarity_threshold,
            'text_change_threshold': self.text_change_threshold,
            'last_frame_hash': self.last_frame_hash is not None,
            'skimage_available': self.skimage_available,
            'enabled': self.frame_comparison_enabled
        } 