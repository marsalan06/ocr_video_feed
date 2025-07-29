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
        #detect GPU or CPU
        device = 0 if torch.backends.mps.is_available() else -1
        logger.info(f"Detected device for NER: {device}")

        #initialize Hugging Face NER pipeline
        logger.info("Loading Hugging Face NER pipeline...")
        logger.info("This may take a few minutes on first run as the model downloads (433MB)")
        self.nlp = pipeline("ner", model="dslim/bert-base-NER", device=device)
        logger.info("NER pipeline initialized successfully")
        logger.info("Checkpoint: NER model fully loaded and ready")

    def is_meaningful_text(self, text):
        """Check if text contains meaningful content for NER analysis"""
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
        logger.info(f"Starting entity analysis for {len(texts)} texts")
        """
        Analyze entities in a text using a Hugging Face NER pipeline
        """
        all_entities = []
        logger.info("Initialized all_entities list")

        for text in texts:
            logger.info(f"Analyzing text: {text}")
            
            # Skip meaningless text to improve performance
            if not self.is_meaningful_text(text):
                logger.info(f"Skipping text '{text}' - not meaningful for NER")
                continue
                
            #Hugging Face returns [{"word":...,"entity":...,"score":...,"index":...}]
            entities = self.nlp(text)
            logger.info(f"Entities found: {entities}")
            all_entities.extend(entities)
            logger.info("Entities added to all_entities list")

        logger.info(f"Returning {len(all_entities)} total entities")
        return all_entities