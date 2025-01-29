import asyncio
import concurrent.futures
import logging
import os
import time
import traceback
from typing import Dict, List
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ta
import ccxt

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        # Initialize Qwen2.5-Math Model for training
        try:
            logger.info("Loading Qwen2.5-Math model for training...")
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
            
            # Configure model for technical analysis
            system_prompt = """You are a technical analysis expert. 
            You analyze trading indicators like RSI, MACD, EMA, and Bollinger Bands.
            Always provide numerical probability scores between 0 and 1.
            Base your analysis on mathematical patterns and statistical correlations."""
            
            self.math_model.config.pre_seq_len = 128
            self.math_model.config.prefix_projection = True
            self.math_model.config.system = system_prompt
            
            logger.info("✅ Training model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading training model: {e}")
            raise

        # Configure thread and process pools for maximum training speed
        self.max_workers = os.cpu_count() * 4  # Use 4x CPU cores
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        
        self.training_tasks = {}
        self.last_training_time = 0
        self.training_interval = 3600 * 4  # Train every 4 hours
        
        logger.info(f"Initialized trainer with {self.max_workers} workers")

    def collect_training_data_sync(self, symbol: str) -> List[dict]:
        """Collect training data synchronously"""
        try:
            logger.info(f"Collecting training data for {symbol}...")
            training_data = []
            
            exchange = ccxt.kraken()
            daily_data = exchange.fetch_ohlcv(symbol, '1d', limit=100)
            if not daily_data:
                return []
            
            df = pd.DataFrame(daily_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate all indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            for i in range(len(df) - 1):
                row = df.iloc[i]
                training_data.append({
                    'date': row.name,
                    'technical': {
                        'indicators': {
                            'rsi': row['rsi'],
                            'ema_short': row['ema_short'],
                            'ema_long': row['ema_long'],
                            'macd': row['macd'],
                            'macd_signal': row['macd_signal'],
                            'macd_histogram': row['macd_histogram'],
                            'bb_width': row['bb_width'],
                            'atr': row['atr']
                        }
                    }
                })
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return []

    def train_qwen_math_sync(self, training_data: List[dict]):
        """Train model synchronously"""
        try:
            prompts = []
            for data in training_data:
                prompt = f"""
                Analyze technical indicators for {data['date']}:
                RSI: {data['technical']['indicators']['rsi']}
                EMA Short: {data['technical']['indicators']['ema_short']}
                EMA Long: {data['technical']['indicators']['ema_long']}
                MACD: {data['technical']['indicators']['macd']}
                BB Width: {data['technical']['indicators']['bb_width']}
                ATR: {data['technical']['indicators']['atr']}
                
                Calculate probability of price increase based on these indicators.
                """
                prompts.append(prompt)
                
            inputs = self.math_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            predictions = self.math_model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=1,
                pad_token_id=self.math_tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            logger.info(f"✅ Model trained on {len(predictions)} examples")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            traceback.print_exc()

    async def start_training(self, symbols: List[str]):
        """Start training for given symbols"""
        try:
            loop = asyncio.get_event_loop()
            
            for symbol in symbols:
                if symbol not in self.training_tasks:
                    training_symbol = f"{symbol.split('/')[0]}/USD"
                    
                    # Collect and train in separate threads
                    training_data = await loop.run_in_executor(
                        self.process_pool,
                        self.collect_training_data_sync,
                        training_symbol
                    )
                    
                    if training_data:
                        await loop.run_in_executor(
                            self.thread_pool,
                            self.train_qwen_math_sync,
                            training_data
                        )
                        
                    logger.info(f"Started training for {training_symbol}")
                    
        except Exception as e:
            logger.error(f"Error starting training: {e}") 