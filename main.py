import cv2
import asyncio
import logging
import sys
from camera_feed import VideoStream
from ocr_processor import OCRProcessor
from utils import draw_text
from text_manager import TextManager  # Import the new module
from config import *  # Import configuration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_feed():
    logger.info("Starting process_feed function")
    
    stream = None
    ocr = None
    text_manager = None
    
    try:
        # Initialize video stream and OCR processor
        logger.info("Initializing VideoStream...")
        stream = VideoStream(CAMERA_SOURCE)  # Start camera feed
        if not stream.is_initialized():
            logger.error("Failed to initialize video stream")
            return
        logger.info("VideoStream initialized successfully")
        
        logger.info("Initializing OCRProcessor...")
        ocr = OCRProcessor()  # Initialize OCR processor
        if not ocr.is_initialized():
            logger.error("Failed to initialize OCR processor")
            return
        logger.info("OCRProcessor initialized successfully")

        # Get initial frame to determine dimensions
        ret, initial_frame = stream.read()
        if not ret or initial_frame is None:
            logger.error("Failed to get initial frame for dimension calculation")
            return
        
        frame_height, frame_width = initial_frame.shape[:2]
        logger.info(f"Initial frame dimensions: {frame_width}x{frame_height}")

        # Initialize text manager with single dynamic bounding box
        logger.info("Initializing TextManager with single dynamic bounding box...")
        text_manager = TextManager(frame_width, frame_height)
        logger.info("TextManager initialized successfully")

        frame_count = 0
        last_text_data = []  # Store last detected text data for stable display
        consecutive_failures = 0

        while True:
            try:
                # Capture a frame from the video feed
                ret, frame = stream.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error("Too many consecutive frame read failures, stopping")
                        break
                    await asyncio.sleep(0.1)  # Brief pause before retry
                    continue
                
                # Reset failure counter on successful frame read
                consecutive_failures = 0
                frame_count += 1

                # Draw bounding box on frame for visualization
                try:
                    text_manager.draw_bounding_box(frame)
                except Exception as draw_error:
                    logger.error(f"Failed to draw bounding box: {draw_error}")

                # Only run OCR every few frames to slow down text updates
                if frame_count % OCR_INTERVAL == 0:
                    logger.info(f"Running OCR on frame {frame_count}")
                    try:
                        # Extract text from the frame using OCR
                        text_data = ocr.extract_text(frame)
                        if text_data:
                            last_text_data = text_data
                            logger.info(f"New text data detected: {len(text_data)} regions")
                            
                            # Process each detected text through TextManager
                            for text, position, confidence in text_data:
                                try:
                                    # Filter by confidence threshold
                                    if confidence < CONFIDENCE_THRESHOLD:
                                        logger.info(f"Filtering low confidence text: '{text}' (confidence: {confidence:.2f})")
                                        continue
                                    
                                    # Add text to the single expanding bounding box with duplicate detection
                                    result = text_manager.add_text(text, position, frame)
                                    if result:
                                        display_position, placed_text = result
                                        # Display text in the bounding box
                                        cv2.putText(frame, placed_text, display_position, 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
                                        logger.info(f"Placed new text '{placed_text}' in expanding box")
                                    else:
                                        logger.info(f"Text '{text}' was duplicate or could not be placed")
                                except Exception as text_error:
                                    logger.error(f"Error processing text '{text}': {text_error}")
                        else:
                            logger.info("No new text data detected")
                    except Exception as ocr_error:
                        logger.error(f"OCR processing failed: {ocr_error}")
                        # Continue with last known good data
                        logger.info("Continuing with cached text data")
                else:
                    logger.info(f"Using cached text data from previous OCR run")

                # Display accumulated text information
                try:
                    accumulated_text = text_manager.get_accumulated_text()
                    if accumulated_text:
                        # Display accumulated text at the bottom of the frame
                        info_text = f"Accumulated: {accumulated_text[:50]}..."
                        cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_TEXT_COLOR, 2)
                        
                        # Display box info
                        box_info = text_manager.get_box_info()
                        box_text = f"Box: {box_info['box_width']}x{box_info['box_height']}px"
                        cv2.putText(frame, box_text, (10, frame.shape[0] - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_TEXT_COLOR, 2)
                        
                        # Display text storage info
                        storage_info = text_manager.get_text_storage_info()
                        if storage_info['save_to_file']:
                            storage_text = f"Text saved to: {storage_info['file_path']}"
                            cv2.putText(frame, storage_text, (10, frame.shape[0] - 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_TEXT_COLOR, 1)
                            
                            # Show text length and unique texts
                            length_text = f"Text: {storage_info['accumulated_text_length']} chars, {storage_info['unique_texts_detected']} unique"
                            cv2.putText(frame, length_text, (10, frame.shape[0] - 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_TEXT_COLOR, 1)
                except Exception as info_error:
                    logger.error(f"Failed to display text information: {info_error}")

                # Display the live video feed with text overlay
                try:
                    cv2.imshow(WINDOW_TITLE, frame)
                except Exception as display_error:
                    logger.error(f"Failed to display frame: {display_error}")
                    break

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("'q' key pressed, stopping stream")
                    break
                
                # Clear text if 'c' is pressed
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    logger.info("'c' key pressed, clearing accumulated text")
                    text_manager.clear_text()
                
                # Async sleep to ensure video feed is not blocked
                await asyncio.sleep(ASYNC_SLEEP_TIME)  # Small delay to slow down processing
                
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
        logger.info("Following flow: Camera Feed → OCR → Text Analysis → Single Dynamic Bounding Box")
        logger.info(f"Text updates slowed down: OCR runs every {OCR_INTERVAL} frames for stability")
        logger.info("Text will be placed in a single expanding bounding box with duplicate detection")
        logger.info("Press 'q' to quit, 'c' to clear accumulated text")
        logger.info(f"Text storage enabled: {SAVE_TEXT_TO_FILE}")
        if SAVE_TEXT_TO_FILE:
            logger.info(f"Text will be saved to: {TEXT_FILE_PATH}")
        asyncio.run(process_feed())
        logger.info("Application finished successfully")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application crashed with error: {e}")
        sys.exit(1)