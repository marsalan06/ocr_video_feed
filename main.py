import cv2
import asyncio
import logging
import sys
from camera_feed import VideoStream
from ocr_processor import OCRProcessor
from utils import draw_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_feed():
    logger.info("Starting process_feed function")
    
    stream = None
    ocr = None
    
    try:
        # Initialize video stream and OCR processor
        logger.info("Initializing VideoStream...")
        stream = VideoStream(0)  # Start camera feed
        if not stream.is_initialized():
            logger.error("Failed to initialize video stream")
            return
        logger.info("VideoStream initialized successfully")
        
        logger.info("Initializing OCRProcessor...")
        ocr = OCRProcessor()  # Initialize OCR processor
        logger.info("OCRProcessor initialized successfully")

        frame_count = 0
        ocr_interval = 15  # Process OCR every 15 frames (about 2 times per second at 30fps)
        last_text_data = []  # Store last detected text data for stable display
        consecutive_failures = 0
        max_consecutive_failures = 10

        while True:
            try:
                # Capture a frame from the video feed
                ret, frame = stream.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame read failures, stopping")
                        break
                    await asyncio.sleep(0.1)  # Brief pause before retry
                    continue
                
                # Reset failure counter on successful frame read
                consecutive_failures = 0
                frame_count += 1

                # Only run OCR every few frames to slow down text updates
                if frame_count % ocr_interval == 0:
                    logger.info(f"Running OCR on frame {frame_count}")
                    try:
                        # Extract text from the frame using OCR
                        text_data = ocr.extract_text(frame)
                        if text_data:
                            last_text_data = text_data
                            logger.info(f"New text data detected: {len(text_data)} regions")
                        else:
                            logger.info("No new text data detected")
                    except Exception as ocr_error:
                        logger.error(f"OCR processing failed: {ocr_error}")
                        # Continue with last known good data
                        logger.info("Continuing with cached text data")
                else:
                    logger.info(f"Using cached text data from previous OCR run")

                # Draw the extracted text on the frame at exact detection points
                try:
                    draw_text(frame, last_text_data)
                except Exception as draw_error:
                    logger.error(f"Failed to draw text on frame: {draw_error}")
                    # Continue without text overlay

                # Display the live video feed with text overlay
                try:
                    cv2.imshow("Live Text Extraction", frame)
                except Exception as display_error:
                    logger.error(f"Failed to display frame: {display_error}")
                    break

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("'q' key pressed, stopping stream")
                    break
                
                # Async sleep to ensure video feed is not blocked
                await asyncio.sleep(0.03)  # Small delay to slow down processing
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, stopping gracefully")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before continuing
                continue
                
    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        return
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if stream:
            try:
                stream.stop()
                logger.info("Stream stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        
        try:
            cv2.destroyAllWindows()
            logger.info("All windows closed")
        except Exception as e:
            logger.error(f"Error closing windows: {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting main application")
        logger.info("Following flow: Camera Feed → OCR → Text Analysis → Overlay Results")
        logger.info("Text updates slowed down: OCR runs every 15 frames for stability")
        logger.info("Text will be overlaid at exact detection points with confidence-based colors")
        asyncio.run(process_feed())
        logger.info("Application finished successfully")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application crashed with error: {e}")
        sys.exit(1)