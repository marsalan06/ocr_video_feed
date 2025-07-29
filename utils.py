import cv2
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_text(frame, text_data):
    """
    Draws the extracted text on the video frame at the exact detection points.
    text_data: list of tuples (text, position, confidence)
    Only displays high confidence text (>0.5)
    """
    if frame is None:
        logger.error("Cannot draw text on None frame")
        return frame
    
    if not isinstance(text_data, list):
        logger.error(f"Invalid text_data type: {type(text_data)}, expected list")
        return frame
    
    logger.info(f"Drawing {len(text_data)} texts on frame at their detection points")
    
    confidence_threshold = 0.5  # Only display high confidence text
    displayed_count = 0
    
    try:
        # Validate frame dimensions
        if len(frame.shape) != 3:
            logger.error(f"Invalid frame shape: {frame.shape}, expected 3D array")
            return frame
        
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            logger.error(f"Invalid frame dimensions: {width}x{height}")
            return frame
        
        for i, item in enumerate(text_data):
            try:
                # Validate text_data item format
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    logger.warning(f"Invalid text_data item at index {i}: {item}, skipping")
                    continue
                
                text, position, confidence = item
                
                # Validate text
                if not isinstance(text, str) or not text.strip():
                    logger.warning(f"Empty or invalid text at index {i}: '{text}', skipping")
                    continue
                
                # Validate position
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    logger.warning(f"Invalid position format at index {i}: {position}, skipping")
                    continue
                
                # Validate confidence
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    logger.warning(f"Invalid confidence value at index {i}: {confidence}, skipping")
                    continue
                
                # Filter out low confidence detections
                if confidence < confidence_threshold:
                    logger.info(f"Filtering out low confidence text: '{text}' (confidence: {confidence:.2f})")
                    continue
                
                # Convert numpy types to regular Python integers for OpenCV compatibility
                try:
                    x = int(position[0])
                    y = int(position[1])
                except (ValueError, TypeError) as coord_error:
                    logger.warning(f"Failed to convert coordinates for text '{text}': {coord_error}")
                    continue
                
                # Validate coordinate ranges
                if x < 0 or y < 0 or x >= width or y >= height:
                    logger.warning(f"Coordinates ({x}, {y}) out of frame bounds ({width}x{height}) for text '{text}'")
                    # Clamp coordinates to frame bounds
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                
                logger.info(f"Drawing text: '{text}' at position ({x}, {y}) with confidence {confidence:.2f}")
                
                # Draw a rectangle around the text transparently
                try:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    rect_x1 = max(0, x)
                    rect_y1 = max(0, y - text_size[1] - 10)
                    rect_x2 = min(width - 1, x + text_size[0] + 10)
                    rect_y2 = min(height - 1, y + 5)
                    
                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), 2)
                except Exception as rect_error:
                    logger.warning(f"Failed to draw rectangle for text '{text}': {rect_error}")
                
                # Draw the text with confidence-based color
                # Green for high confidence (>0.8), Yellow for medium (0.5-0.8), Red for low (<0.5)
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Display text at the detected position
                try:
                    text_x = max(0, min(x + 5, width - 1))
                    text_y = max(0, min(y - 5, height - 1))
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception as text_error:
                    logger.warning(f"Failed to draw text '{text}': {text_error}")
                    continue
                
                logger.info(f"Text drawn at position ({x}, {y}) with color {color}")
                displayed_count += 1
                
            except Exception as item_error:
                logger.error(f"Error processing text_data item at index {i}: {item_error}")
                continue
        
        logger.info(f"Displayed {displayed_count} high-confidence texts on frame (threshold: {confidence_threshold})")
        
    except Exception as e:
        logger.error(f"Unexpected error in draw_text: {e}")
    
    return frame