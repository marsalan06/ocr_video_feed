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
    """
    logger.info(f"Drawing {len(text_data)} texts on frame at their detection points")
    
    for text, position, confidence in text_data:
        # Convert numpy types to regular Python integers for OpenCV compatibility
        x = int(position[0])
        y = int(position[1])
        
        logger.info(f"Drawing text: '{text}' at position ({x}, {y}) with confidence {confidence:.2f}")
        
        # Draw a background rectangle for better text visibility
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y + 5), (0, 0, 0), 2)
        
        # Draw the text with confidence-based color
        # Green for high confidence (>0.8), Yellow for medium (0.5-0.8), Red for low (<0.5)
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Display text at the detected position
        cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw bounding box around the detected text area (optional)
        # cv2.rectangle(frame, (x, y), (x + text_size[0] + 10, y + text_size[1] + 10), color, 2)
        
        logger.info(f"Text drawn at position ({x}, {y}) with color {color}")
    
    logger.info("All texts drawn on frame at their detection points")
    return frame