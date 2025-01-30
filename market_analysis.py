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
        # Initialize FinBERT Model
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'ProsusAI/finbert',
                torch_dtype=torch.float16
            )
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            logger.info(f"✅ FinBERT model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
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

    async def get_market_sentiment(self, data: dict, symbol: str) -> float:
        """Get sentiment prediction from FinBERT model with caching"""
        try:
            logger.info(f"Starting sentiment analysis for {symbol}")
            cache_key = f"{symbol}_{time.time() // self.cache_ttl}"
            
            # Check cache first
            if cache_key in self.analysis_cache:
                logger.info(f"Cache hit for {symbol}")
                return self.analysis_cache[cache_key]
            
            # Generate comprehensive analysis text
            analysis_text = f"""
            Technical analysis for {symbol}:
            RSI is at {data['technical']['indicators']['rsi']:.2f}, indicating {'oversold' if data['technical']['indicators']['rsi'] < 30 else 'overbought' if data['technical']['indicators']['rsi'] > 70 else 'neutral'} conditions.
            
            Moving Averages:
            - EMA Short: {data['technical']['indicators']['ema_short']:.2f}
            - EMA Long: {data['technical']['indicators']['ema_long']:.2f}
            - Trend is {'bullish' if data['technical']['indicators']['ema_short'] > data['technical']['indicators']['ema_long'] else 'bearish'}
            
            MACD Analysis:
            - MACD: {data['technical']['indicators']['macd']:.4f}
            - Showing {'bullish' if data['technical']['indicators']['macd'] > 0 else 'bearish'} momentum
            
            Volatility Metrics:
            - Bollinger Band Width: {data['technical']['indicators']['bb_width']:.4f}
            - ATR: {data['technical']['indicators']['atr']:.4f}
            - Volatility is {'high' if data['technical']['indicators']['bb_width'] > 0.5 else 'normal'}
            """
            
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            inputs = self.tokenizer(analysis_text, return_tensors="pt", padding=True, truncation=True)
            
            logger.info(f"Running sentiment analysis for {symbol}")
            outputs = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.model(**inputs)
            )
            
            # Get sentiment probabilities
            sentiment = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment_scores = sentiment[0].tolist()
            
            # Cache and return positive sentiment score
            positive_score = sentiment_scores[0]  # FinBERT order: positive, negative, neutral
            self.analysis_cache[cache_key] = positive_score
            
            return positive_score
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return 0.5 
