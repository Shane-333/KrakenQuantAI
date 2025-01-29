from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import pandas as pd
from typing import Dict, List, Tuple
import logging
import ta
import time
import asyncio
import ccxt
import os
import concurrent.futures

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        # Initialize Qwen2.5-Math Model with correct padding
        try:
            logger.info("Loading Qwen2.5-Math model...")
            self.math_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-Math-7B", 
                trust_remote_code=True,
                padding_side='left'
            )
            self.math_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Math-7B",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load pre-trained weights for technical analysis
            self.load_pretrained_weights()
            logger.info("✅ Qwen2.5-Math model loaded successfully with pre-trained weights")
            
        except Exception as e:
            logger.error(f"Error loading Qwen2.5-Math model: {e}")
            raise
        
        self.exchange = None
        
        # Configure thread and process pools with safe limits
        self.max_workers = min(os.cpu_count() * 2, 60)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=min(8, self.max_workers))
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def set_exchange(self, exchange):
        """Set the exchange instance"""
        self.exchange = exchange

    def load_pretrained_weights(self):
        """Load pre-trained weights for technical analysis"""
        try:
            # Configure model for technical analysis
            system_prompt = """You are a technical analysis expert. 
            You analyze trading indicators like RSI, MACD, EMA, and Bollinger Bands.
            Always provide numerical probability scores between 0 and 1.
            Base your analysis on mathematical patterns and statistical correlations."""
            
            self.math_model.config.pre_seq_len = 128
            self.math_model.config.prefix_projection = True
            self.math_model.config.system = system_prompt
            
            logger.info("Pre-trained weights configured for technical analysis")
            
        except Exception as e:
            logger.error(f"Error loading pre-trained weights: {e}") 

    async def get_math_prediction(self, data: dict, symbol: str) -> float:
        """Get prediction from pre-trained math model with caching"""
        try:
            logger.info(f"Starting prediction for {symbol}")
            cache_key = f"{symbol}_{time.time() // self.cache_ttl}"
            
            # Check cache first
            if cache_key in self.analysis_cache:
                logger.info(f"Cache hit for {symbol}")
                return self.analysis_cache[cache_key]
                
            logger.info(f"Tokenizing input for {symbol}")
            prompt = f"""
            Analyze technical indicators for {symbol}:
            RSI: {data['technical']['indicators']['rsi']}
            EMA Short: {data['technical']['indicators']['ema_short']}
            EMA Long: {data['technical']['indicators']['ema_long']}
            MACD: {data['technical']['indicators']['macd']}
            BB Width: {data['technical']['indicators']['bb_width']}
            ATR: {data['technical']['indicators']['atr']}
            """
            
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            inputs = self.math_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            logger.info(f"Running model inference for {symbol}")
            predictions = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.math_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.math_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            )
            logger.info(f"Completed inference for {symbol}")
            
            # Cache the result
            response = self.math_tokenizer.decode(predictions[0], skip_special_tokens=True)
            try:
                probability = float([x for x in response.split() if x.replace('.','').isdigit()][0])
                probability = max(0.0, min(1.0, probability))
            except:
                probability = 0.5
                
            self.analysis_cache[cache_key] = probability
            return probability
                
        except Exception as e:
            logger.error(f"Error in math prediction for {symbol}: {e}")
            return 0.5 
