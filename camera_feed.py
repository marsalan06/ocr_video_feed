import cv2
import logging
import time
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, src=0):
        logger.info(f"Initializing VideoStream with source: {src}")
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.thread = None
        self.initialization_successful = False
        
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source {src}")
                return
            
            # Test reading first frame
            self.ret, self.frame = self.cap.read()
            if not self.ret or self.frame is None:
                logger.error("Failed to read initial frame from camera")
                return
            
            logger.info("VideoCapture object created successfully")
            logger.info(f"Camera resolution: {self.frame.shape[1]}x{self.frame.shape[0]}")
            
            # Start background thread
            logger.info("Starting background thread for frame updates")
            self.thread = Thread(target=self.update, args=(), daemon=True)
            self.thread.start()
            
            # Wait a moment for thread to start
            time.sleep(0.1)
            if self.thread.is_alive():
                self.initialization_successful = True
                logger.info("VideoStream initialization completed successfully")
            else:
                logger.error("Background thread failed to start")
                
        except Exception as e:
            logger.error(f"Error during VideoStream initialization: {e}")
            self.cleanup()
    
    def is_initialized(self):
        """Check if the video stream was initialized successfully"""
        return self.initialization_successful and self.cap is not None and self.cap.isOpened()

    def update(self):
        """Method to read frames from camera in background thread"""
        logger.info("Background update thread started")
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 30  # Allow more failures in background thread
        
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.error("Camera not available in background thread")
                    break
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.ret = ret
                    self.frame = frame
                    consecutive_failures = 0
                    frame_count += 1
                    
                    # Only log every 30 frames (about once per second at 30fps)
                    if frame_count % 30 == 0:
                        logger.info(f"Background thread: Read {frame_count} frames successfully")
                else:
                    consecutive_failures += 1
                    logger.warning(f"Background thread: Failed to read frame (attempt {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures in background thread, stopping")
                        break
                    
                    time.sleep(0.01)  # Brief pause before retry
                    
            except Exception as e:
                logger.error(f"Error in background update thread: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many errors in background thread, stopping")
                    break
                time.sleep(0.01)
        
        logger.info("Background update thread stopped")
    
    def read(self):
        """Return the latest frame"""
        try:
            if not self.is_initialized():
                logger.error("VideoStream not properly initialized")
                return False, None
            
            if self.frame is None:
                logger.warning("No frame available")
                return False, None
            
            return self.ret, self.frame.copy()  # Return a copy to avoid threading issues
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def stop(self):
        """Stop the video stream and release camera"""
        logger.info("Stopping video stream")
        try:
            self.stopped = True
            
            # Wait for background thread to finish
            if self.thread and self.thread.is_alive():
                logger.info("Waiting for background thread to finish...")
                self.thread.join(timeout=2.0)  # Wait up to 2 seconds
                if self.thread.is_alive():
                    logger.warning("Background thread did not finish gracefully")
            
            # Release camera
            if self.cap is not None:
                self.cap.release()
                logger.info("Camera released successfully")
                
        except Exception as e:
            logger.error(f"Error stopping video stream: {e}")

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        logger.info("VideoStream destructor called")
        self.stop()

