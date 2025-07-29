from transformers import pipeline
import torch
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERProcessor:
    def __init__(self):
        logger.info("Initializing NERProcessor")
        self.nlp = None
        self.initialization_successful = False
        
        try:
            # Detect GPU or CPU
            device = 0 if torch.backends.mps.is_available() else -1
            logger.info(f"Detected device for NER: {device}")

            # Initialize Hugging Face NER pipeline
            logger.info("Loading Hugging Face NER pipeline...")
            logger.info("This may take a few minutes on first run as the model downloads (433MB)")
            
            try:
                self.nlp = pipeline("ner", model="dslim/bert-base-NER", device=device)
                logger.info("NER pipeline initialized successfully")
                self.initialization_successful = True
                logger.info("Checkpoint: NER model fully loaded and ready")
            except Exception as pipeline_error:
                logger.error(f"Failed to initialize NER pipeline: {pipeline_error}")
                logger.info("Attempting to initialize with CPU fallback...")
                try:
                    self.nlp = pipeline("ner", model="dslim/bert-base-NER", device=-1)
                    logger.info("NER pipeline initialized with CPU fallback")
                    self.initialization_successful = True
                except Exception as cpu_error:
                    logger.error(f"Failed to initialize NER pipeline with CPU: {cpu_error}")
                    self.initialization_successful = False
                    
        except Exception as e:
            logger.error(f"Error during NERProcessor initialization: {e}")
            self.initialization_successful = False

    def is_initialized(self):
        """Check if NER processor was initialized successfully"""
        return self.initialization_successful and self.nlp is not None

    def is_meaningful_text(self, text):
        """Check if text contains meaningful content for NER analysis"""
        if not isinstance(text, str):
            return False
            
        # Skip pure numbers, symbols, or very short text
        if len(text) < 3:
            return False
        # Skip if text is mostly numbers/symbols
        if re.match(r'^[\d\s\.\,\-\+\*\/]+$', text):
            return False
        # Skip if text is mostly special characters
        if re.match(r'^[^\w\s]+$', text):
            return False
        return True

    def analyze_entities(self, texts):
        """
        Analyze entities in a text using a Hugging Face NER pipeline
        """
        if not self.is_initialized():
            logger.error("NER processor not properly initialized")
            return []
        
        if not isinstance(texts, (list, tuple)):
            logger.error(f"Invalid texts type: {type(texts)}, expected list or tuple")
            return []
        
        logger.info(f"Starting entity analysis for {len(texts)} texts")
        all_entities = []
        logger.info("Initialized all_entities list")

        for i, text in enumerate(texts):
            try:
                # Validate text input
                if not isinstance(text, str):
                    logger.warning(f"Invalid text type at index {i}: {type(text)}, skipping")
                    continue
                
                logger.info(f"Analyzing text: {text}")
                
                # Skip meaningless text to improve performance
                if not self.is_meaningful_text(text):
                    logger.info(f"Skipping text '{text}' - not meaningful for NER")
                    continue
                
                # Process text with NER pipeline
                try:
                    # Hugging Face returns [{"word":...,"entity":...,"score":...,"index":...}]
                    entities = self.nlp(text)
                    logger.info(f"Entities found: {entities}")
                    
                    # Validate entities format
                    if isinstance(entities, list):
                        for entity in entities:
                            if isinstance(entity, dict) and 'word' in entity and 'entity' in entity:
                                all_entities.append(entity)
                            else:
                                logger.warning(f"Invalid entity format: {entity}")
                    else:
                        logger.warning(f"Unexpected entities format: {type(entities)}")
                        
                except Exception as ner_error:
                    logger.error(f"NER processing failed for text '{text}': {ner_error}")
                    continue
                
                logger.info("Entities added to all_entities list")

            except Exception as text_error:
                logger.error(f"Error processing text at index {i}: {text_error}")
                continue

        logger.info(f"Returning {len(all_entities)} total entities")
        return all_entities