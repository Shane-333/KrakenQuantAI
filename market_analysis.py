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
import json
from datetime import datetime

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
        self.model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes

        self.warmup_done = False

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
        """Optimized sentiment analysis with batching support"""
        # Validate input data structure first
        if not isinstance(data, dict) or 'timeframe' not in data or 'price_data' not in data:
            logger.warning(f"Invalid data type for {symbol}: {type(data)}")
            return 0.5
            
        try:
            price_data = data['price_data']  # Direct access instead of get()
            if not isinstance(price_data, dict):
                logger.warning(f"Invalid price_data type for {symbol}: {type(price_data)}")
                return 0.5
            
            required_keys = ['open', 'high', 'low', 'close']
            if not all(key in price_data for key in required_keys):
                logger.warning(f"Missing required keys for {symbol}: {price_data.keys()}")
                return 0.5
                
            if not all(isinstance(price_data[key], list) for key in required_keys):
                logger.warning(f"Price data values must be lists for {symbol}")
                return 0.5

            # Add timeout and validation
            analysis_text = self._generate_analysis_text(
                price_data,
                data['timeframe']
            )
            
            if not analysis_text:
                return 0.5

            # Add input validation
            inputs = self.tokenizer(
                analysis_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding='max_length'  # More stable than 'longest'
            )

            # Use dedicated executor for model inference
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                self.model_executor,  # Create dedicated ThreadPoolExecutor
                lambda: self.model(**inputs)
            )

            # Safer score calculation
            if outputs.logits.dim() != 2:
                return 0.5
                
            return torch.nn.functional.softmax(outputs.logits, dim=1)[0][0].item()
        
        except Exception as e:
            logger.error(f"Sentiment error: {str(e)[:200]}")
            return 0.5

    def _generate_analysis_text(self, price_data, timeframe):
        """Generate enriched technical analysis text"""
        try:
            # Get last 3 periods for trend context
            closes = price_data.get('close', [])[-3:]
            lows = price_data.get('low', [])
            highs = price_data.get('high', [])
            
            # Calculate technical context
            current_close = closes[-1] if closes else None
            prev_close = closes[-2] if len(closes) >=2 else current_close
            price_change_pct = ((current_close - prev_close)/prev_close * 100) if prev_close else 0
            
            # Build analysis string
            analysis = [
                f"{timeframe} market analysis:",
                f"Latest Close: {current_close:.2f} ({price_change_pct:+.1f}% vs previous)",
                f"Recent Range: {min(lows[-3:]):.2f}-{max(highs[-3:]):.2f}" if 'low' in price_data and 'high' in price_data else "",
                f"Volatility (ATR): {self._calculate_atr(price_data):.2f}" if len(closes) >=14 else ""
            ]
            
            return " | ".join([s for s in analysis if s])
        
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Analysis text error: {e}")
            return f"Basic {timeframe} conditions: Close {price_data['close'][-1]:.2f}"

    def _calculate_atr(self, price_data, period=14):
        """Calculate Average True Range from price data"""
        try:
            highs = price_data.get('high', [])
            lows = price_data.get('low', [])
            closes = price_data.get('close', [])
            
            if len(highs) < period or len(lows) < period or len(closes) < period:
                return 0.0
            
            true_ranges = [
                max(h - l, abs(h - pc), abs(l - pc))
                for h, l, pc in zip(highs[-period:], lows[-period:], closes[-period-1:-1])
            ]
            
            return sum(true_ranges) / period
        
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0

    async def warmup_model(self):
        """Warm up the model with empty input"""
        if not self.warmup_done:
            dummy_input = self.tokenizer("", return_tensors="pt")
            _ = self.model(**dummy_input)
            self.warmup_done = True

    def calculate_signals(self, df: pd.DataFrame) -> dict:
        """Calculate technical signal strength from dataframe"""
        try:
            # Correct RSI calculation
            rsi_indicator = ta.momentum.RSIIndicator(df['close'], 14)
            rsi = rsi_indicator.rsi().iloc[-1]
            
            # Correct MACD initialization
            macd = ta.trend.MACD(df['close'])
            macd_hist = macd.macd_diff().iloc[-1]  # Histogram values
            
            # Add these calculations after RSI and MACD
            rsi_strength = max(0, min(1, (rsi - 30) / 40))  # 30-70 range → 0-1
            macd_strength = abs(macd_hist) / 0.05  # Normalize to 0-1 scale
            
            # Volume spike
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            volume_strength = min(1, df['volume'].iloc[-1] / (vol_ma * 2))
            
            return {
                'signal_strength': (rsi_strength + macd_strength + volume_strength) / 3,
                'rsi': rsi,
                'macd_hist': macd_hist,
                'volume_ratio': volume_strength
            }
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return {'signal_strength': 0.0}
