import cv2
import asyncio
import logging
from camera_feed import VideoStream
from ocr_processor import OCRProcessor
from utils import draw_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_feed():
    logger.info("Starting process_feed function")

    # Initialize video stream and OCR processor
    stream = VideoStream(0)  # Start camera feed
    logger.info("VideoStream initialized")
    
    ocr = OCRProcessor()  # Initialize OCR processor
    logger.info("OCRProcessor initialized")

    frame_count = 0
    ocr_interval = 15  # Process OCR every 15 frames (about 2 times per second at 30fps)
    last_text_data = []  # Store last detected text data for stable display

    while True:
        # Capture a frame from the video feed
        ret, frame = stream.read()
        if not ret:
            logger.info("Failed to read frame, breaking loop")
            break

        frame_count += 1

        # Only run OCR every few frames to slow down text updates
        if frame_count % ocr_interval == 0:
            logger.info(f"Running OCR on frame {frame_count}")
            # Extract text from the frame using OCR
            text_data = ocr.extract_text(frame)
            if text_data:
                last_text_data = text_data
                logger.info(f"New text data detected: {len(text_data)} regions")
            else:
                logger.info("No new text data detected")
        else:
            logger.info(f"Using cached text data from previous OCR run")

        # Draw the extracted text on the frame at exact detection points
        draw_text(frame, last_text_data)

        # Display the live video feed with text overlay
        cv2.imshow("Live Text Extraction", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("'q' key pressed, stopping stream")
            stream.stop()
            logger.info("Stream stopped")
            break
        
        # Async sleep to ensure video feed is not blocked
        await asyncio.sleep(0.03)  # Small delay to slow down processing

if __name__ == "__main__":
    logger.info("Starting main application")
    logger.info("Following flow: Camera Feed → OCR → Text Analysis → Overlay Results")
    logger.info("Text updates slowed down: OCR runs every 15 frames for stability")
    logger.info("Text will be overlaid at exact detection points with confidence-based colors")
    asyncio.run(process_feed())
    logger.info("Application finished")