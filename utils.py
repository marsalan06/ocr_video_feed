import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_text(frame, texts):
    """
    Draws the extracted text on the video frame.
    """
    logger.info(f"Drawing {len(texts)} texts on frame")
    
    # Set starting vertical position for the text
    y_offset = 50  # Increased starting position
    
    for text in texts:
        logger.info(f"Drawing text: {text}")
        # Display each text on the frame at the given position with larger font
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        y_offset += 40  # Increased spacing between lines
        logger.info(f"Text drawn at position (10, {y_offset-40})")
    
    logger.info("All texts drawn on frame")
    return frame