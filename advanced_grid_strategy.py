import asyncio
import logging
import os
import platform
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import talib
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from datetime import datetime
import time
import traceback
from market_analysis import MarketAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KrakenAdvancedGridStrategy:
    def __init__(self, demo_mode=False):  # Set default to False for live
        self.demo_mode = demo_mode
        
        # Load environment variables
        load_dotenv()
        
        # Select appropriate API credentials based on mode
        if self.demo_mode:
            self.api_key = os.getenv('KRAKEN_PAPER_API_KEY')
            self.secret_key = os.getenv('KRAKEN_PAPER_SECRET_KEY')
            self.base_url = os.getenv('KRAKEN_FUTURES_BASE_URL')
            self.ws_url = os.getenv('KRAKEN_FUTURES_WS_URL')
            logger.info("🔄 Starting in DEMO mode")
        else:
            self.api_key = os.getenv('KRAKEN_API_KEY')
            self.secret_key = os.getenv('KRAKEN_SECRET_KEY')
            self.base_url = os.getenv('KRAKEN_LIVE_BASE_URL')  # Updated
            self.ws_url = os.getenv('KRAKEN_LIVE_WS_URL')      # Updated
            logger.info("🚨 Starting in LIVE mode ")
            
        # Initialize exchange
        self.exchange = ccxt.kraken({  # Note: kraken not krakenfutures
            'apiKey': self.api_key,  # Already in base64
            'secret': self.secret_key,  # Already in base64
            'enableRateLimit': True,
            'urls': {
                'api': {
                    'public': 'https://api.kraken.com',
                    'private': 'https://api.kraken.com',
                    'ws': 'wss://ws.kraken.com',
                    'wsauth': 'wss://ws-auth.kraken.com'
                }
            },
            'options': {
                'adjustForTimeDifference': True,
                'computePnL': True,
                'fetchMinOrderAmounts': True
            }
        })
        
        # Set sandbox mode for paper trading
        if self.demo_mode:
            self.exchange.set_sandbox_mode(True)
            logger.info("Running in demo mode")
        else:
            logger.info("Running in live mode")
                
        # Initialize empty tiers (will be populated by get_tradeable_instruments)
        self.tier1_symbols = []
        self.tier2_symbols = []
        self.tier3_symbols = []
        self.tier4_symbols = []
        
        # Strategy parameters
        self.grid_levels = 5
        self.base_grid_spacing = 0.005  # Keep this
        self.grid_spacing = self.base_grid_spacing
        self.min_grid_spacing = 0.003   # Keep this
        self.max_grid_spacing = 0.05    # Keep this
        self.min_distance = 0.002       # Increased from 0.001
        self.risk_reward_ratio = 2.0
        self.max_risk_per_trade = 0.10
        self.last_balance = 0.0
        self.leverage = 50  # Default leverage
        
        # Technical indicators parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.ema_short_period = 9
        self.ema_long_period = 21
        self.atr_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Add Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std = 2
        self.min_bbw = 0.01   # Decreased from 0.015
        self.max_bbw = 0.08   # Increased from 0.05
        
        # Add risk management parameters
        self.max_open_positions = 10
        self.max_daily_trades = 10
        self.daily_trade_count = 0
        self.last_trade_reset = datetime.now().date()
        
        # Data storage
        self.grid_orders = {}
        self.active_positions = {}
        self.historical_data = {}
        
        # Add trading state parameters
        self.min_trade_interval = 3600  # 1 hour in seconds
        
        # Add position tracking
        self.positions = {}
        self.order_history = []
        
        # Add performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        
        # Add fee and slippage parameters
        self.maker_fee = 0.0002  # 0.02% maker fee
        self.taker_fee = 0.0005  # 0.05% taker fee
        self.slippage_buffer = 0.001  # 0.1% slippage buffer
        self.volatility_buffer = 0.002  # 0.2% volatility buffer
        
        # Add stale order timeout parameter
        self.stale_order_timeout_hours = 24  # Default timeout for stale orders (24 hours)
        
        # Initialize exchange and price tracking
        self.initialize_exchange()

        self.price_update_times = {}
        self.price_fetch_retries = 3
        
        # Fetch and categorize tradeable instruments
        loop = asyncio.get_event_loop()
        categorized_symbols = loop.run_until_complete(self.get_tradeable_instruments())
        
        # Update tier symbols
        self.tier1_symbols = categorized_symbols.get('tier1', [])
        self.tier2_symbols = categorized_symbols.get('tier2', [])
        self.tier3_symbols = categorized_symbols.get('tier3', [])
        self.tier4_symbols = categorized_symbols.get('tier4', [])
        
        # Initialize active_symbols with tier 1 if no symbols provided
        self.symbols = self.tier1_symbols
        self.active_symbols = self.tier1_symbols.copy()
        
        # Initialize current_prices with actual symbols
        self.current_prices = {symbol: None for symbol in self.symbols}
        
        # Initialize last_symbol_update
        self.last_symbol_update = 0
        
        # Initialize last_trade_time with all possible symbols
        all_possible_symbols = (
            self.tier1_symbols + 
            self.tier2_symbols + 
            self.tier3_symbols + 
            self.tier4_symbols
        )
        self.last_trade_time = {symbol: None for symbol in all_possible_symbols}
        
        # Add to existing init
        self.market_analyzer = MarketAnalyzer()
        self.market_analyzer.set_exchange(self.exchange)
        
        # Remove training task creation from here
        self.training_tasks = []  # Just initialize empty list

    def initialize_exchange(self) -> None:
        """Initialize CCXT exchange connection"""
        try:
            load_dotenv()
            
            # Select appropriate API credentials based on mode
            if self.demo_mode:
                self.api_key = os.getenv('KRAKEN_PAPER_API_KEY')
                self.secret_key = os.getenv('KRAKEN_PAPER_SECRET_KEY')
                logger.info("🔄 Initializing in DEMO mode")
            else:
                self.api_key = os.getenv('KRAKEN_API_KEY')
                self.secret_key = os.getenv('KRAKEN_SECRET_KEY')
                logger.info("🚨 Initializing in LIVE mode")
            
            if not self.api_key or not self.secret_key:
                raise ValueError("API credentials not found in .env file")
            
            self.exchange = ccxt.kraken({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'urls': {
                    'api': {
                        'public': 'https://api.kraken.com',
                        'private': 'https://api.kraken.com',
                        'ws': 'wss://ws.kraken.com',
                        'wsauth': 'wss://ws-auth.kraken.com'
                    }
                },
                'options': {
                    'adjustForTimeDifference': True,
                    'computePnL': True,
                    'fetchMinOrderAmounts': True
                }
            })
            
            # Set sandbox mode for paper trading
            if self.demo_mode:
                self.exchange.set_sandbox_mode(True)
                logger.info("Running in demo mode")
            else:
                logger.info("Running in live mode")
                
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    async def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance['total']['USD'])
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0

    async def get_open_positions(self) -> List[Dict]:
        """Get open positions including both spot and margin/futures with FIFO tracking"""
        try:
            active_positions = []
            
            # Only check actual spot balances
            balance = await self.exchange.fetch_balance()
            for currency, amount in balance.get('free', {}).items():
                if currency in ['USD', 'USDT']:
                    continue
                    
                total_amount = float(amount)
                if total_amount > 0.00001:  # Minimum threshold
                    try:
                        symbol = f"{currency}/USD"
                        
                        # Get the entry price from CLOSED order history (not open orders)
                        closed_orders = await self.exchange.fetch_closed_orders(symbol, limit=10)
                        
                        # Filter for buy orders that were filled
                        filled_buys = [o for o in closed_orders if o['side'] == 'buy' and o['status'] == 'closed']
                        
                        # Sort by timestamp for FIFO tracking (oldest first)
                        filled_buys.sort(key=lambda x: x['timestamp'])
                        
                        if filled_buys:
                            # Calculate weighted average entry price for FIFO tracking
                            total_cost = 0
                            total_quantity = 0
                            
                            # Log all entry points for FIFO tracking
                            logger.info(f"\nFIFO Entry Points for {symbol}:")
                            for i, order in enumerate(filled_buys):
                                order_price = float(order['price'])
                                order_amount = float(order['filled'])
                                order_cost = order_price * order_amount
                                order_time = datetime.fromtimestamp(order['timestamp']/1000)
                                
                                total_cost += order_cost
                                total_quantity += order_amount
                                
                                logger.info(f"Entry #{i+1}: {order_amount} @ ${order_price} on {order_time}")
                            
                            # Calculate weighted average entry
                            avg_entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                            
                            logger.info(f"Found actual spot position for {symbol}:")
                            logger.info(f"Balance: {total_amount} {currency}")
                            logger.info(f"FIFO Entries: {len(filled_buys)} orders")
                            logger.info(f"Avg Entry: ${avg_entry_price:.4f}")
                            
                            spot_position = {
                                'symbol': symbol,
                                'info': {
                                    'symbol': symbol,
                                    'size': total_amount,
                                    'price': avg_entry_price,
                                    'side': 'long',
                                    'is_spot': True,
                                    'fifo_entries': filled_buys  # Store FIFO entries for later use
                                }
                            }
                            
                            active_positions.append(spot_position)
                            logger.info(f"Active spot position: {symbol}: long {total_amount} @ {avg_entry_price:.4f}")
                        else:
                            logger.warning(f"No entry orders found for {symbol} despite having balance of {total_amount}")
                            
                            # Fallback to current price if no entry orders found
                            current_price = await self.get_current_price(symbol)
                            if current_price:
                                spot_position = {
                                    'symbol': symbol,
                                    'info': {
                                        'symbol': symbol,
                                        'size': total_amount,
                                        'price': current_price,
                                        'side': 'long',
                                        'is_spot': True,
                                        'fifo_entries': []  # Empty FIFO entries
                                    }
                                }
                                active_positions.append(spot_position)
                                logger.info(f"Using current price as entry: {symbol}: long {total_amount} @ {current_price}")
                    except Exception as e:
                        logger.error(f"Error processing {currency} balance: {e}")
                        continue
            
            # Log summary of actual positions
            if active_positions:
                logger.info("\nActive Positions Summary:")
                for pos in active_positions:
                    symbol = pos['info']['symbol']
                    try:
                        current_price = await self.get_current_price(symbol)
                        if current_price:  # Only call manage_stop_loss if we have a valid price
                            await self.manage_stop_loss(symbol, pos, current_price)
                            logger.info(f"{symbol}: {pos['info']['side']} {pos['info']['size']} @ {pos['info']['price']}")
                        else:
                            logger.error(f"Could not get current price for {symbol}")
                    except Exception as e:
                        logger.error(f"Error processing position for {symbol}: {e}")
            else:
                logger.info("No active positions found")
            
            return active_positions
                
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    async def get_historical_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get historical OHLCV data and calculate indicators"""
        try:
            # Normalize the symbol first
            symbol = self.normalize_symbol(symbol)
            
            # Convert timeframe to minutes for Kraken API
            timeframe_minutes = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
                '1w': 10080
            }
            
            interval = timeframe_minutes.get(timeframe, 60)  # Default to 1h
            
            # Get OHLCV data using Kraken's specific endpoint
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                timeframe,
                limit=limit,
                params={"interval": interval}
            )
            
            if not ohlcv:
                logger.error(f"No OHLCV data returned for {symbol}")
                return None
                
            logger.info(f"Received {len(ohlcv)} candles for {symbol}")
            
            # Create DataFrame and calculate indicators
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate existing indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], self.rsi_period).rsi()
            df['ema_short'] = ta.trend.EMAIndicator(df['close'], self.ema_short_period).ema_indicator()
            df['ema_long'] = ta.trend.EMAIndicator(df['close'], self.ema_long_period).ema_indicator()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period).average_true_range()
            
            # Calculate MACD
            macd = ta.trend.MACD(
                df['close'], 
                window_slow=self.macd_slow, 
                window_fast=self.macd_fast, 
                window_sign=self.macd_signal
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                df['close'],
                window=self.bb_period,
                window_dev=self.bb_std
            )
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Calculate Bollinger Band Width
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            logger.info(f"Calculated all indicators for {symbol}")
            logger.info(f"Latest RSI: {df['rsi'].iloc[-1]:.2f}")
            logger.info(f"Latest MACD: {df['macd'].iloc[-1]:.4f}")
            logger.info(f"Latest BB Width: {df['bb_width'].iloc[-1]:.4f}")
            
            return df
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def analyze_market_conditions(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float]:
        """Analyze market conditions using technical indicators and FinBERT sentiment"""
        try:
            # Check if we have insufficient funds and there's no active position
            if not self.has_sufficient_funds and symbol not in self.active_positions:
                return False, 0.0  # Skip analysis for new positions
                
            # Quick check for active positions first (always check regardless of funds)
            if symbol in self.active_positions:
                current_price = df.iloc[-1]['close']
                position = self.active_positions[symbol]
                await self.manage_stop_loss(symbol, position, current_price)
            
            # Only continue analysis if we have funds or an active position
            if self.has_sufficient_funds or symbol in self.active_positions:
                # Rest of existing analysis code...
                async with asyncio.timeout(15):  # Increased from 5 to 15 seconds
                    tasks = []
                    last_row = df.iloc[-1]
                    technical_data = {
                        'timeframe': '5m',  # Since df is the 5-minute dataframe
                        'price_data': {
                            'open': df['open'].tolist(),
                            'high': df['high'].tolist(),
                            'low': df['low'].tolist(),
                            'close': df['close'].tolist()
                        },
                        'technical': {
                            'indicators': {
                                'rsi': last_row['rsi'],
                                'ema_short': last_row['ema_short'],
                                'ema_long': last_row['ema_long'],
                                'macd': last_row['macd'],
                                'bb_width': last_row['bb_width'],
                                'atr': last_row['atr']
                            }
                        }
                    }
                    
                    # Add timeout for sentiment analysis
                    ai_task = asyncio.create_task(
                        asyncio.wait_for(
                            self.market_analyzer.get_market_sentiment(technical_data, symbol),
                            timeout=15
                        )
                    )
                    tasks.append(ai_task)
                    
                    # Task 2: Calculate technical signals concurrently
                    def calculate_signals():
                        rsi = last_row['rsi']
                        rsi_signal = 30 <= rsi <= 70
                        
                        ema_trend = last_row['ema_short'] > last_row['ema_long']
                        
                        macd_cross = last_row['macd'] > last_row['macd_signal']
                        
                        current_bbw = last_row['bb_width']
                        bbw_valid = current_bbw < 0.5
                        
                        return rsi_signal, ema_trend, macd_cross, current_bbw, bbw_valid
                    
                    # Run technical analysis in thread pool
                    loop = asyncio.get_event_loop()
                    tech_task = loop.run_in_executor(self.market_analyzer.thread_pool, calculate_signals)
                    tasks.append(tech_task)
                    
                    # Wait for all tasks to complete
                    sentiment_score, tech_signals = await asyncio.gather(*tasks)
                    
                    # Unpack technical signals
                    rsi_signal, ema_trend, macd_cross, current_bbw, bbw_valid = tech_signals
                    
                    # Calculate final signals (using sentiment score directly)
                    signal_strength = (
                        (1 if rsi_signal else 0) +
                        (1 if ema_trend else 0) +
                        (1 if macd_cross else 0) +
                        (1 if bbw_valid else 0) +
                        (sentiment_score)  # Add sentiment as part of signal strength
                    ) / 5.0
                    
                    should_trade = signal_strength >= 0.6  # Adjusted threshold
                    
                    logger.info(f"Analysis complete - Sentiment: {sentiment_score:.2f}, Signal: {signal_strength:.2f}, Trade: {'✅' if should_trade else '❌'}")
                    
                    return should_trade, current_bbw
                    
        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout for {symbol}")
            return False, 0.0
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return False, 0.0

    def adjust_grid_parameters(self, volatility_ratio: float, current_price: float) -> None:
        """Adjust grid parameters based on market volatility and price"""
        try:
            # Price-based grid adjustments
            if current_price >= 20000:  # BTC range
                self.base_grid_spacing = 0.04    # 4% spacing
                self.min_profit_threshold = 3.5   # Higher profit needed
                position_multiplier = 1           # Standard size
            elif current_price >= 1000:  # ETH range
                self.base_grid_spacing = 0.035   # 3.5% spacing
                self.min_profit_threshold = 3.0
                position_multiplier = 1.5
            elif current_price >= 100:   # High-value alts
                self.base_grid_spacing = 0.03    # 3% spacing
                self.min_profit_threshold = 2.5
                position_multiplier = 2
            elif current_price >= 10:    # Mid-value alts
                self.base_grid_spacing = 0.025   # 2.5% spacing
                self.min_profit_threshold = 2.0
                position_multiplier = 3
            elif current_price >= 1:     # Low-value alts
                self.base_grid_spacing = 0.02    # 2% spacing
                self.min_profit_threshold = 1.5
                position_multiplier = 4
            else:                        # Micro-price alts
                self.base_grid_spacing = 0.015   # 1.5% spacing
                self.min_profit_threshold = 1.0
                position_multiplier = 5

            # Adjust grid spacing with volatility
            new_spacing = self.base_grid_spacing * volatility_ratio
            self.grid_spacing = max(self.min_grid_spacing, min(self.max_grid_spacing, new_spacing))
            
            # Adjust grid levels based on price range and volatility
            if current_price < 1:
                base_levels = max(4, min(8, self.grid_levels))  # More levels for cheaper coins
            else:
                base_levels = self.grid_levels
                
            self.grid_levels = max(3, min(8, int(base_levels * volatility_ratio)))
            
            logger.info(f"Adjusted parameters for price ${current_price:.4f}:")
            logger.info(f"Grid Spacing: {self.grid_spacing:.4f}")
            logger.info(f"Grid Levels: {self.grid_levels}")
            logger.info(f"Position Multiplier: {position_multiplier}x")
            logger.info(f"Min Profit Threshold: ${self.min_profit_threshold}")
            
        except Exception as e:
            logger.error(f"Error adjusting grid parameters: {e}")

    async def update_performance_metrics(self, trade_result: float) -> None:
        """Update strategy performance metrics with enhanced risk monitoring"""
        try:
            self.total_trades += 1
            if trade_result > 0:
                self.winning_trades += 1
            self.total_profit += trade_result
            
            # Calculate win rate
            win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
            
            # Update max drawdown
            if trade_result < 0:
                current_drawdown = abs(trade_result)
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Calculate risk metrics
            risk_ratio = self.max_drawdown / self.total_profit if self.total_profit > 0 else float('inf')
            
            logger.info(f"Performance metrics - Win rate: {win_rate:.2f}%, "
                       f"Total profit: {self.total_profit:.2f}, "
                       f"Max drawdown: {self.max_drawdown:.2f}, "
                       f"Risk ratio: {risk_ratio:.2f}")
            
            # Check risk thresholds
            if risk_ratio > 0.5:  # Risk-reward ratio threshold
                logger.warning("Risk-reward ratio exceeded threshold")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def check_trade_conditions(self, symbol: str, df: pd.DataFrame) -> bool:
        """Check additional trading conditions"""
        try:
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return False
            
            last_row = df.iloc[-1]
            
            # Check time-based conditions
            current_time = datetime.now().timestamp()
            if (self.last_trade_time[symbol] and 
                current_time - self.last_trade_time[symbol] < self.min_trade_interval):
                logger.info(f"Trade interval not met for {symbol}")
                return False
            
            # Calculate BB position
            bb_upper = last_row['bb_upper']
            bb_lower = last_row['bb_lower']
            price = last_row['close']
            bb_range = bb_upper - bb_lower
            bb_position = (price - bb_lower) / bb_range if bb_range > 0 else 0
            
            # Calculate volume condition
            volume_sma = df['volume'].rolling(window=3).mean()
            current_volume = last_row['volume']
            volume_ratio = current_volume / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 0
            
            # Adjust conditions based on asset
            if 'ETH/USD' in symbol or 'PI_ETHUSD' in symbol:
                # Super lenient conditions for ETH
                volume_condition = volume_ratio > 0.01  # Only require 1% of SMA volume for ETH
                bb_position_valid = True  # Always valid for ETH
                
                # Detailed ETH logging
                logger.info(f"ETH Detailed Conditions:")
                logger.info(f"- Current Volume: {current_volume}")
                logger.info(f"- Volume SMA: {volume_sma.iloc[-1]}")
                logger.info(f"- Volume Ratio: {volume_ratio:.3f}")
                logger.info(f"- BB Position: {bb_position:.3f}")
            else:
                # Normal conditions for other assets
                volume_condition = volume_ratio > 0.3
                bb_position_valid = 0.0 <= bb_position <= 1.2
            
            conditions_met = volume_condition and bb_position_valid
            
            logger.info(f"Trade conditions for {symbol} - BB Position: {bb_position:.2f}, "
                       f"Volume condition: {volume_condition} (ratio: {volume_ratio:.3f}), "
                       f"BB valid: {bb_position_valid}, "
                       f"Asset type: {'ETH' if 'ETH' in symbol else 'Other'}")
            
            return conditions_met
            
        except Exception as e:
            logger.error(f"Error checking trade conditions for {symbol}: {e}")
            return False

    async def check_trade_interval(self, symbol: str) -> bool:
        """Check if enough time has passed since last trade"""
        try:
            current_time = datetime.now().timestamp()
            if (self.last_trade_time[symbol] and 
                current_time - self.last_trade_time[symbol] < self.min_trade_interval):
                logger.info(f"Trade interval not met for {symbol}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking trade interval: {e}")
            return False

    async def update_trade_history(self, symbol: str, order: dict) -> None:
        """Update trade history and last trade time"""
        try:
            self.order_history.append({
                'symbol': symbol,
                'order': order,
                'timestamp': datetime.now().timestamp()
            })
            self.last_trade_time[symbol] = datetime.now().timestamp()
            logger.info(f"Updated trade history for {symbol}")
        except Exception as e:
            logger.error(f"Error updating trade history: {e}")

    async def check_risk_limits(self, symbol: str) -> bool:
        """Check if risk limits allow new trades with enhanced validation"""
        try:
            # Reset daily trade count if it's a new day
            current_date = datetime.now().date()
            if current_date > self.last_trade_reset:
                self.daily_trade_count = 0
                self.last_trade_reset = current_date
            
            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                logger.info("Daily trade limit reached")
                return False
            
            # Check maximum open positions
            open_positions_count = len(self.active_positions)
            if open_positions_count >= self.max_open_positions:
                logger.info("Maximum open positions limit reached")
                return False
            
            # Check account balance vs risk
            current_balance = await self.get_account_balance()
            if current_balance < self.last_balance * 0.9:  # 10% drawdown limit
                logger.info("Account drawdown limit reached")
                return False
            
            # Additional position-specific risk check
            if symbol in self.positions:
                position = self.positions[symbol]
                entry_price = position.get('entry_price')
                current_price = await self.get_current_price(symbol)
                if entry_price and current_price:
                    loss_pct = ((current_price - entry_price) / entry_price) * 100
                    if abs(loss_pct) > 5:  # 5% max loss per position
                        logger.warning(f"Position risk too high for {symbol}: {loss_pct}%")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    async def check_existing_orders(self, symbol: str) -> bool:
        """Check if there are already grid orders placed for this symbol"""
        try:
            # Fetch only open orders for the symbol
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            if open_orders:
                # Remove duplicates using set and ensure proper float conversion
                order_prices = sorted(set(float(order['price']) for order in open_orders))
                
                # Get current price for comparison
                current_price = await self.get_current_price(symbol)
                if current_price:
                    # Count orders on each side of current price
                    buy_orders = len([p for p in order_prices if p < current_price])
                    sell_orders = len([p for p in order_prices if p > current_price])
                    
                    logger.info(f"\nExisting Grid Orders for {symbol}:")
                    logger.info(f"Current Price: ${current_price:.4f}")
                    logger.info(f"Buy Orders: {buy_orders}")
                    logger.info(f"Sell Orders: {sell_orders}")
                    logger.info(f"Grid Prices: {[f'${p:.4f}' for p in order_prices]}")
                    
                    return True
            
            # This code was unreachable due to incorrect indentation
            logger.info(f"No open grid orders found for {symbol}")
            return False
            
        except Exception as e:
            logger.error(f"Error checking existing orders for {symbol}: {e}")
            return False

    async def validate_existing_orders(self, symbol: str, valid_grid_prices: List[float]) -> None:
        """Validate and cancel orders that don't align with current strategy"""
        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
            current_price = await self.get_current_price(symbol)
            if not current_price:
                return

            # Allow small deviation from grid prices (0.5%)
            tolerance = current_price * 0.005

            for order in open_orders:
                order_price = float(order['price'])
                
                # Check if order price is close to any valid grid price
                is_valid_price = any(abs(order_price - grid_price) <= tolerance 
                                   for grid_price in valid_grid_prices)
                
                # Cancel if price isn't valid
                if not is_valid_price:
                    logger.info(f"Cancelling order at {order_price} - not aligned with current grid levels")
                    await self.exchange.cancel_order(order['id'], symbol)
                    if order['id'] in self.grid_orders:
                        del self.grid_orders[order['id']]
                    await asyncio.sleep(0.1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error validating existing orders for {symbol}: {e}")

    async def calculate_support_resistance(self, df: pd.DataFrame, symbol: str) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels using daily, 1h and 15m timeframes"""
        try:
            # Get multi-timeframe data with 1h instead of 5m
            daily_df = await self.get_historical_data(symbol, timeframe='1d', limit=100)  # ~3 months
            hourly_df = await self.get_historical_data(symbol, timeframe='1h', limit=168)  # ~7 days (1 week)
            fifteen_min_df = await self.get_historical_data(symbol, timeframe='15m', limit=100)  # ~24h
            
            if daily_df is None or hourly_df is None or fifteen_min_df is None:
                logger.warning("No data available for S/R calculation")
                return [], []

            # Get current price for filtering
            current_price = df['close'].iloc[-1]
            
            # Calculate daily Fibonacci levels (structural)
            daily_high = daily_df['high'].max()
            daily_low = daily_df['low'].min()
            daily_diff = daily_high - daily_low
            
            # Updated daily Fibonacci levels with both retracements and extensions
            daily_fib_levels = {
                # Support levels (retracements)
                -1: daily_low - daily_diff,      # Extension below
                0: daily_low,                    # Low point
                0.236: daily_low + (daily_diff * 0.236),
                0.382: daily_low + (daily_diff * 0.382),
                0.5: daily_low + (daily_diff * 0.5),
                0.618: daily_low + (daily_diff * 0.618),
                0.786: daily_low + (daily_diff * 0.786),
                1: daily_high,                   # High point
                # Resistance levels (extensions)
                1.414: daily_high + (daily_diff * 0.414),
                1.618: daily_high + (daily_diff * 0.618),
                2.618: daily_high + (daily_diff * 1.618),
                3.618: daily_high + (daily_diff * 2.618),
                4.236: daily_high + (daily_diff * 3.236)
            }

            # Calculate hourly S/R (replacing 5min with 1h)
            hourly_high = hourly_df['high'].rolling(window=24).max()  # 24 hours of data
            hourly_low = hourly_df['low'].rolling(window=24).min()
            
            # Calculate 15min S/R
            fifteen_high = fifteen_min_df['high'].rolling(window=20).max()
            fifteen_low = fifteen_min_df['low'].rolling(window=20).min()
            
            # Calculate hourly Fibonacci levels (medium-term) - replacing 5min
            recent_high = hourly_df['high'].max()
            recent_low = hourly_df['low'].min()
            diff = recent_high - recent_low
            
            hourly_fib_levels = {
                0: recent_low,                   # Start level
                0.236: recent_low + (diff * 0.236),
                0.382: recent_low + (diff * 0.382),
                0.5: recent_low + (diff * 0.5),
                0.618: recent_low + (diff * 0.618),
                0.786: recent_low + (diff * 0.786),
                1: recent_high,                  # End level
                1.414: recent_high + (diff * 0.414),
                1.618: recent_high + (diff * 0.618),
                2.618: recent_high + (diff * 1.618),
                3.618: recent_high + (diff * 2.618),
                4.236: recent_high + (diff * 3.236)
            }
            
            # Combine all timeframes
            resistance_levels = []
            support_levels = []

            # Define reasonable range for filtering (30% above and below current price)
            upper_limit = current_price * 1.30
            lower_limit = current_price * 0.70
            
            # Add daily levels with proper categorization and filtering
            for level, value in daily_fib_levels.items():
                if level >= 1 and value <= upper_limit:  # 1 and above are resistance
                    resistance_levels.append(value)
                elif level < 1 and value >= lower_limit:  # Below 1 are support
                    support_levels.append(value)

            # Add hourly levels (medium priority) with filtering
            for level in hourly_high.dropna().tolist():
                if level <= upper_limit:
                    resistance_levels.append(level)
                    
            for level in hourly_low.dropna().tolist():
                if level >= lower_limit:
                    support_levels.append(level)
            
            # Add hourly Fibonacci levels with filtering
            for level in [1, 1.414, 1.618, 2.618, 3.618, 4.236]:
                value = hourly_fib_levels[level]
                if value <= upper_limit:
                    resistance_levels.append(value)
                    
            for level in [0, 0.236, 0.382, 0.5, 0.618, 0.786]:
                value = hourly_fib_levels[level]
                if value >= lower_limit:
                    support_levels.append(value)
            
            # Add 15min levels (lowest priority) with filtering
            for level in fifteen_high.dropna().tolist():
                if level <= upper_limit:
                    resistance_levels.append(level)
                    
            for level in fifteen_low.dropna().tolist():
                if level >= lower_limit:
                    support_levels.append(level)
            
            # Add recent high/low if within range
            if recent_high <= upper_limit:
                resistance_levels.append(recent_high)
            if recent_low >= lower_limit:
                support_levels.append(recent_low)

            # Log all levels
            logger.info(f"\nS/R Levels for {symbol} (Current Price: ${current_price:.4f}):")
            logger.info(f"Filtering range: ${lower_limit:.4f} to ${upper_limit:.4f}")
            
            logger.info("Daily Fibonacci Levels:")
            logger.info("Support Levels:")
            for k in [-1, 0, 0.236, 0.382, 0.5, 0.618, 0.786]:
                value = daily_fib_levels[k]
                included = "✓" if lower_limit <= value <= upper_limit else "✗"
                logger.info(f"- {k}: ${value:.4f} {included}")
            
            logger.info("Resistance Levels:")
            for k in [1, 1.414, 1.618, 2.618, 3.618, 4.236]:
                value = daily_fib_levels[k]
                included = "✓" if lower_limit <= value <= upper_limit else "✗"
                logger.info(f"- {k}: ${value:.4f} {included}")
                    
            logger.info(f"\nHourly Fib Extensions: {[round(hourly_fib_levels[level], 4) for level in [1.414, 1.618, 2.618, 3.618, 4.236]]}")
            logger.info(f"Hourly Fib Retracements: {[round(hourly_fib_levels[level], 4) for level in [0, 0.236, 0.382, 0.5, 0.618, 0.786]]}")
            logger.info(f"Recent High/Low: {round(recent_high, 4)}/{round(recent_low, 4)}")
            
            # Improved deduplication with tolerance
            def deduplicate_levels(levels: List[float], tolerance: float = 0.001) -> List[float]:
                if not levels:
                    return []
                result = [levels[0]]
                for level in levels[1:]:
                    if all(abs(level - x) / x > tolerance for x in result):
                        result.append(level)
                return sorted(result)
            
            # Deduplicate before returning
            support_levels = deduplicate_levels(support_levels)
            resistance_levels = deduplicate_levels(resistance_levels)
            
            # Final filtering - only include levels within reasonable range of current price
            support_levels = [level for level in support_levels if level >= lower_limit]
            resistance_levels = [level for level in resistance_levels if level <= upper_limit]
            
            logger.info(f"\nFinal Support Levels: {[round(level, 4) for level in support_levels]}")
            logger.info(f"Final Resistance Levels: {[round(level, 4) for level in resistance_levels]}")
            
            return support_levels, resistance_levels
                
        except Exception as e:
            logger.error(f"Error calculating S/R levels: {e}")
            return [], []

    async def get_funding_rate(self, symbol: str) -> Tuple[float, float]:
        """Get current and estimated next funding rate from exchange"""
        try:
            # Fetch funding rate info from Kraken Futures
            funding_info = await self.exchange.fetch_funding_rate(symbol)
            
            if funding_info:
                current_rate = float(funding_info.get('fundingRate', 0.0001))
                next_rate = float(funding_info.get('estimatedRate', current_rate))
                
                logger.info(f"Funding rates for {symbol}:")
                logger.info(f"Current Rate: {current_rate:.6%}")
                logger.info(f"Est. Next Rate: {next_rate:.6%}")
                
                return current_rate, next_rate
                
            return 0.0001, 0.0001  # Default fallback rates
            
        except Exception as e:
            logger.warning(f"Error fetching funding rate for {symbol}: {e}, using defaults")
            return 0.0001, 0.0001  # Default fallback rates

    async def get_exchange_fees(self, symbol: str) -> Tuple[float, float]:
        """Get current maker/taker fees from Kraken Spot"""
        try:
            # Use CCXT's built-in fee fetching for spot
            fees = await self.exchange.fetch_trading_fees()
            
            if fees and 'maker' in fees and 'taker' in fees:
                maker_fee = fees['maker']
                taker_fee = fees['taker']
                
                logger.info(f"\nSpot Trading Fees:")
                logger.info(f"Maker Fee: {maker_fee:.4%}")
                logger.info(f"Taker Fee: {taker_fee:.4%}")
                
                return maker_fee, taker_fee
                
            logger.warning("Could not fetch spot trading fees")
            
        except Exception as e:
            logger.error(f"Error fetching Kraken Spot fees: {e}")
        
        # Default conservative spot fees
        return 0.0026, 0.0040  # 0.26% default Kraken spot fees

    async def calculate_total_fees(self, price: float, size: float, symbol: str) -> float:
        """Calculate total round-trip fees including entry, exit, and funding"""
        try:
            # Get current exchange fees - ENSURE THIS IS AWAITED
            maker_fee, taker_fee = await self.get_exchange_fees(symbol)  # Changed to await
            
            # Get current funding rate - ENSURE THIS IS AWAITED
            funding_rate = await self.get_funding_rate(symbol)  # Changed to await
            
            # Calculate fees
            entry_fee = price * size * maker_fee
            exit_fee = price * size * taker_fee
            funding_fee = price * size * abs(funding_rate)
            
            total_fees = entry_fee + exit_fee + funding_fee
            
            # Log fee breakdown with actual rates
            logger.info(f"\nFee Breakdown for {symbol}:")
            logger.info(f"Entry Fee: ${entry_fee:.4f} (Maker Rate: {maker_fee:.4%})")
            logger.info(f"Exit Fee: ${exit_fee:.4f} (Taker Rate: {taker_fee:.4%})")
            logger.info(f"Funding Fee: ${funding_fee:.4f} (Rate: {funding_rate:.6f})")
            logger.info(f"Total Fees: ${total_fees:.4f}")
            
            return total_fees
            
        except Exception as e:
            logger.error(f"Error calculating fees: {e}")
            return 0.0

    def validate_trade_profitability(self, price: float, size: float, symbol: str) -> bool:
        """Validate if trade is profitable after fees with dynamic thresholds"""
        try:
            # Calculate total fees
            total_fees = self.calculate_total_fees(price, size, symbol)
            
            # Get price-appropriate minimum profit threshold
            if price >= 20000:
                min_profit_multiplier = 2.0  # Higher threshold for BTC
            elif price >= 1000:
                min_profit_multiplier = 1.8  # ETH range
            elif price >= 100:
                min_profit_multiplier = 1.6  # High-value alts
            elif price >= 10:
                min_profit_multiplier = 1.4  # Mid-value alts
            elif price >= 1:
                min_profit_multiplier = 1.2  # Low-value alts
            else:
                min_profit_multiplier = 1.1  # Micro-price alts
                
            # Calculate minimum profit needed with dynamic threshold
            min_profit_needed = total_fees * min_profit_multiplier
            
            # Calculate grid profit
            grid_profit = price * size * self.grid_spacing
            
            # Check if grid profit covers fees with appropriate buffer
            is_profitable = grid_profit > min_profit_needed
            
            logger.info(f"\nProfitability Check for {symbol}:")
            logger.info(f"Price Range: ${price:.4f}")
            logger.info(f"Grid Profit: ${grid_profit:.4f}")
            logger.info(f"Min Profit Needed: ${min_profit_needed:.4f}")
            logger.info(f"Profit Multiplier: {min_profit_multiplier}x")
            logger.info(f"Profitable: {'✅' if is_profitable else '❌'}")
            
            return is_profitable
            
        except Exception as e:
            logger.error(f"Error validating profitability: {e}")
            return False

    async def execute_grid_orders(self, symbol: str, force_create: bool = False) -> None:
        """Execute grid orders with minimum volume validation"""
        try:
            # Get current price and calculate position size
            current_price = await self.get_current_price(symbol)
            position_size = await self.calculate_position_size(current_price, symbol)
            
            # Exchange minimum volume requirements
            min_volumes = {
                'CRV/USD': 5.0,
                'XLM/USD': 30.0,
                'DOGE/USD': 50.0,
                'MATIC/USD': 10.0,
                'SOL/USD': 0.1,
                'ETH/USD': 0.02,
                'BTC/USD': 0.0001,
                'AVAX/USD': 0.1,
                'DOT/USD': 1.0,
                'ADA/USD': 5.0,  # Updated ADA minimum
                'LINK/USD': 1.0,
                'ATOM/USD': 0.1,
                'FIL/USD': 0.1,
                'UNI/USD': 1.0,
                'AAVE/USD': 0.01,
                'LTC/USD': 0.1,
                'OP/USD': 1.0,
                'APE/USD': 1.0,
                'NEAR/USD': 1.0,
                'FTM/USD': 10.0
            }
            
            # Get minimum volume for this symbol
            min_volume = min_volumes.get(symbol, 5.0)  # Default to 5.0 if not specified
            
            # Check if position size meets minimum
            if position_size < min_volume:
                logger.warning(f"❌ Calculated position size {position_size} is below minimum {min_volume} for {symbol}")
                logger.warning(f"Skipping grid creation to prevent locked positions")
                return
            
            # Calculate grid levels
            grid_levels = await self.calculate_grid_levels(current_price, symbol)
            
            # Validate each grid level meets minimum
            for level in grid_levels:
                level_size = position_size / len(grid_levels)  # Size per grid level
                if level_size < min_volume:
                    logger.warning(f"❌ Grid level size {level_size} is below minimum {min_volume} for {symbol}")
                    logger.warning(f"Skipping grid creation to prevent locked positions")
                    return
            
            # If we get here, all sizes meet minimum requirements
            logger.info(f"✅ All order sizes meet minimum volume requirements for {symbol}")
            
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 STARTING GRID EXECUTION FOR {symbol}")
            
            # Get historical data first
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                logger.warning(f"❌ No historical data for {symbol}")
                return

            # First verify if this is a spot position
            try:
                balance = await self.exchange.fetch_balance()
                asset = symbol.split('/')[0]
                spot_balance = float(balance.get(asset, {}).get('free', 0))
                is_spot = spot_balance > 0
                logger.info(f"Asset: {asset}, Spot Balance: {spot_balance}, Is Spot: {is_spot}")
            except Exception as e:
                logger.error(f"Error checking spot balance: {e}")
                is_spot = False

             # Get multi-timeframe trend analysis
            trend_data = await self.analyze_multi_timeframe_trend(symbol)
            
            # Log trend analysis
            logger.info(f"\nMulti-Timeframe Analysis for {symbol}:")
            for timeframe, data in trend_data.items():
                logger.info(f"{timeframe}: {data['trend'].upper()} (Strength: {data['strength']:.2f}%)")
                logger.info(f"RSI: {data['rsi']:.2f}")
                logger.info(f"Near Support: {'✅' if data['near_support'] else '❌'}")
                logger.info(f"Near Resistance: {'✅' if data['near_resistance'] else '❌'}")

            # Adjust grid levels based on technical analysis
            if is_spot:
                # Add more buy grids if oversold in higher timeframes
                if trend_data['1h']['is_oversold'] and trend_data['1h']['near_support']:
                    self.grid_levels += 1
                    logger.info("✅ Adding extra buy grid - Oversold conditions")
                
                # Reduce grids if overbought
                if trend_data['1h']['is_overbought'] and trend_data['1h']['near_resistance']:
                    self.grid_levels = max(1, self.grid_levels - 1)
                    logger.info("⚠️ Reducing grid levels - Overbought conditions")

            # Market Analysis - Do this BEFORE checking existing orders
            logger.info(f"\n{'='*30}")
            logger.info(f"🔍 ANALYZING MARKET CONDITIONS FOR {symbol}")
            logger.info(f"{'='*30}")
 
            should_trade, volatility = await self.analyze_market_conditions(df, symbol)
            logger.info(f"Market conditions check - Should trade: {'✅' if should_trade else '❌'}, Volatility: {volatility:.2f}%")
            
            if not should_trade:
                logger.info(f"❌ Market conditions not favorable for {symbol}, skipping")
                return
                
            # Adjust grid parameters based on volatility
            self.adjust_grid_parameters(volatility, float(df['close'].iloc[-1]))
            logger.info(f"Grid parameters adjusted for volatility: {volatility:.2f}%")

            logger.info(f"🔍 Checking existing orders for {symbol}")
            # Now check existing orders
            if not force_create:
                has_existing_orders = await self.check_existing_orders(symbol)
                logger.info(f"Existing orders check: {'✅' if has_existing_orders else '❌'}")
                if has_existing_orders:
                    logger.info(f"Valid existing orders for {symbol}, skipping")
                    return

            # Get current price and validate
            current_price = await self.get_current_price(symbol)
            if not current_price:
                logger.warning(f"❌ Could not get current price for {symbol}")
                return
            logger.info(f"Current price for {symbol}: ${current_price:.4f}")

            # Calculate S/R levels
            logger.info(f"📊 Calculating support/resistance levels")
            support_levels, resistance_levels = await self.calculate_support_resistance(df, symbol)

            # Calculate grid prices with confidence scores
            logger.info(f"📈 Calculating grid prices")
            grid_prices = await self.calculate_hybrid_grids(symbol, df, support_levels, resistance_levels)
            if not grid_prices:
                logger.warning(f"❌ No valid grid levels for {symbol}")
                return

            logger.info(f"✅ Generated {len(grid_prices)} grid prices")
            
            # Convert numpy float64 to regular float and ensure proper format
            normalized_prices = []
            for price in grid_prices:
                if isinstance(price, (tuple, list)):
                    price_val, confidence = price
                    normalized_prices.append((float(price_val), float(confidence)))
                elif isinstance(price, (float, np.float64)):
                    # Handle case where price is a single value
                    normalized_prices.append((float(price), 1.0))
                else:
                    logger.warning(f"Skipping invalid price format: {price}")
                    continue

            # Normalize and validate grid prices in one pass
            max_confidence = max((conf for _, conf in normalized_prices), default=0)

            # Initialize valid grids list and confidence map
            valid_grids = []
            grid_confidence = {}  # Store confidence values separately
            previous_price = None

            # Calculate price change and volume confirmation
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            recent_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_confirmed = current_volume > recent_volume * 0.1  # Reduce to 10%

            # Handle single float values in grid_prices
            for price_item in grid_prices:
                # Convert price to float and assign default confidence
                if isinstance(price_item, (np.float64, float)):
                    price_obj = float(price_item)
                    raw_confidence = 1.0
                else:
                    price_obj, raw_confidence = price_item
                    price_obj = float(price_obj)
                
                # Normalize confidence
                if max_confidence > 0:
                    normalized_confidence = (raw_confidence / max_confidence) * 100  # Scale to 0-100 instead of 0-10
                    if normalized_confidence < 70:  # Require 70% of max confidence
                        logger.info(f"Skipping price {price_obj} - confidence too low: {normalized_confidence:.2f}")
                        continue
                else:
                    normalized_confidence = 0
                
                # Define minimum distance between grids
                min_distance = current_price * 0.02

                # Check minimum distance from previous grid
                if previous_price and abs(price_obj - previous_price) < min_distance:
                    logger.info(f"Skipping price {price_obj} - too close to previous grid at {previous_price}")
                    continue
                
                # Check S/R alignment
                is_near_support = any(abs(price_obj - s) / s <= 0.01 for s in support_levels)
                is_near_resistance = any(abs(price_obj - r) / r <= 0.01 for r in resistance_levels)
                
                # Calculate price change and momentum
                would_be_side = "buy" if price_obj < current_price else "sell"
                momentum_aligned = (price_change > 0 and price_obj > current_price) or \
                                 (price_change < 0 and price_obj < current_price)

                # Adjust validation requirements
                valid_for_trade = (
                    (momentum_aligned or volume_confirmed) or  # Need either momentum OR volume
                    (
                        (would_be_side == "buy" and is_near_support) or
                        (would_be_side == "sell" and is_near_resistance) or
                        normalized_confidence > 8  
                    )
                )
                
                if valid_for_trade:
                    logger.info(f"✅ Validated grid at {price_obj}")
                    grid_confidence[price_obj] = normalized_confidence
                    valid_grids.append(price_obj)
                    previous_price = price_obj
                    logger.info(f"✅ Added {would_be_side} level at {price_obj}")
                else:
                    logger.info(f"\n❌ Rejected {price_obj} due to:")
                    logger.info(f"Volume confirmed: {volume_confirmed}")
                    logger.info(f"Momentum aligned: {momentum_aligned}")
                    logger.info(f"Side: {would_be_side}")
                    logger.info(f"Near support: {is_near_support}")
                    logger.info(f"Near resistance: {is_near_resistance}")

            # NEW ORDER PLACEMENT SECTION
            logger.info(f"\n{'='*30}")
            logger.info(f"💰 PLACING ORDERS FOR {symbol}")
            logger.info(f"{'='*30}")

            orders_placed = 0
            min_buffer = current_price * 0.005

            for price in valid_grids:
                side = "buy" if price < current_price else "sell"
                
                # Skip sell orders for spot positions
                if is_spot and side == "sell":
                    logger.info(f"❌ Skipping sell order at ${price:.4f} - Spot position")
                    continue
                
                logger.info(f"\nValidating {side.upper()} order at ${price:.4f}")
                
                if abs(price - current_price) < min_buffer:
                    logger.info(f"❌ Skipping - too close to current price (buffer: ${min_buffer:.4f})")
                    continue
                    
                # Get initial size
                size = await self.calculate_position_size(price, symbol)
                logger.info(f"Initial calculated position size: {size}")
                
                # Add size validation before fee check
                size = await self.validate_grid_order_size(symbol, size)
                logger.info(f"Validated position size: {size}")
                
                # Add fee validation
                if not self.validate_trade_profitability(price, size, symbol):
                    logger.info(f"❌ Skipping - not profitable after fees")
                    continue
                
                try:
                    # Modify the order parameters to use Kraken's specific flags
                    order_params = {
                        'trading_agreement': 'agree',
                        'oflags': 'post'  # This is Kraken's way of setting post-only orders
                    }
                    
                    # Log the exact order parameters
                    logger.info(f"Attempting to place order with params:")
                    logger.info(f"Symbol: {symbol}")
                    logger.info(f"Type: limit")
                    logger.info(f"Side: {side}")
                    logger.info(f"Amount: {size}")
                    logger.info(f"Price: {price}")
                    logger.info(f"Params: {order_params}")
                    
                    order = await self.exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side=side,
                        amount=size,
                        price=price,
                        params=order_params
                    )
                    
                    if order and 'id' in order:
                        self.grid_orders[order['id']] = {
                            'symbol': symbol,
                            'side': side,
                            'size': float(size),
                            'price': float(price),
                            'confidence': grid_confidence.get(float(price), 'default'),
                            'sr_aligned': True
                        }
                        orders_placed += 1
                        logger.info(f"✅ Successfully placed {side} order at {price} for {size} {symbol}")
                        logger.info(f"Order ID: {order['id']}")
                    
                    await asyncio.sleep(0.1)  # Rate limiting between orders
                    
                except Exception as e:
                    logger.error(f"Order placement failed for {symbol} at {price}: {str(e)}")
                    continue
            
            if orders_placed > 0:
                logger.info(f"✅ Successfully placed {orders_placed} orders for {symbol}")
            else:
                logger.warning(f"⚠️ No orders were placed for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error executing grid orders: {e}")

    async def get_current_price(self, symbol: str) -> float:
        """Get current price with enhanced error handling and validation"""
        try:
            # Check cache first with strict timeout
            cached_price = self.current_prices.get(symbol)
            cache_time = self.price_update_times.get(symbol, 0)
            if cached_price and time.time() - cache_time < 15:  # 15-second cache
                return cached_price

            # Fetch new price with retry and longer delay
            for attempt in range(3):
                try:
                    # Add delay between retries
                    if attempt > 0:
                        await asyncio.sleep(1)
                        
                    ticker = await self.exchange.fetch_ticker(symbol)
                    if not ticker:
                        logger.warning(f"No ticker data received for {symbol}")
                        continue
                    
                    # Try last price first
                    price = ticker.get('last')
                    if price is not None:
                        price = float(price)
                        if price > 0:
                            self.current_prices[symbol] = price
                            self.price_update_times[symbol] = time.time()
                            return price
                    
                    # Fall back to bid/ask average
                    bid = ticker.get('bid')
                    ask = ticker.get('ask')
                    if bid is not None and ask is not None:
                        price = (float(bid) + float(ask)) / 2
                        if price > 0:
                            self.current_prices[symbol] = price
                            self.price_update_times[symbol] = time.time()
                            return price
                        
                    logger.warning(f"No valid price found in ticker for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
            # If we get here, we failed to get a valid price
            logger.error(f"Failed to get valid price for {symbol} after all attempts")
            return None
                
        except Exception as e:
            logger.error(f"Error in get_current_price for {symbol}: {e}")
            return None

    async def calculate_grid_levels(self, current_price: float, symbol: str = None) -> List[float]:
        """Calculate grid price levels with dynamic precision and position sizing"""
        try:
            levels = []
            
            # First verify if this is a spot position
            try:
                balance = await self.exchange.fetch_balance()
                asset = symbol.split('/')[0] if symbol else None
                spot_balance = float(balance.get(asset, {}).get('free', 0)) if asset else 0
                is_spot = spot_balance > 0
                logger.info(f"Asset: {asset}, Spot Balance: {spot_balance}, Is Spot: {is_spot}")
            except Exception as e:
                logger.error(f"Error checking spot balance: {e}")
                is_spot = False
            
            # Dynamic precision and parameters based on price range
            if current_price >= 20000:  # BTC range
                self.grid_spacing = max(0.04, self.base_grid_spacing)   
                self.min_distance = 0.025                               
                decimal_places = 1                                      
                self.grid_levels = 3 if not is_spot else 2  # Fewer grids for spot                                   
                min_size = 0.0001 * 3                                 
            elif current_price >= 1000:  # ETH range
                self.grid_spacing = max(0.05, self.base_grid_spacing)   
                self.min_distance = 0.03                               
                decimal_places = 0                                     
                self.grid_levels = 3                                   
                min_size = 0.01 * 3                                   
            elif current_price >= 100:   # High-value alts
                self.grid_spacing = max(0.055, self.base_grid_spacing)  
                self.min_distance = 0.032                              
                decimal_places = 2                                     
                self.grid_levels = 2                                   
                min_size = 0.1 * 3                                    
            else:  # Lower-value alts
                self.grid_spacing = max(0.06, self.base_grid_spacing)   
                self.min_distance = 0.035                              
                decimal_places = 4                                     
                self.grid_levels = 2                                   
                min_size = 1.0 * 3                                    
            
            account_value = self.last_balance
            allocation_per_grid = account_value * 0.10 / self.grid_levels
            
            if is_spot:
                # For spot, ONLY create buy grids below current price
                for i in range(1, self.grid_levels + 1):
                    grid_price = round(current_price * (1 - (i * self.grid_spacing)), decimal_places)
                    grid_size = max(allocation_per_grid / grid_price, min_size)
                    
                    # Add buy grid
                    levels.append({
                        'price': grid_price,
                        'size': round(grid_size, 8),
                        'side': 'buy'  # Only buy grids for spot
                    })
            else:
                # For futures, keep center grid and both directions
                center_price = round(current_price, decimal_places)
                center_size = max(allocation_per_grid / center_price, min_size)
                levels.append({
                    'price': center_price,
                    'size': round(center_size, 8),
                    'confidence': 1.0,
                    'side': None
                })
                
                for i in range(1, self.grid_levels + 1):
                    # Upper grid (sell)
                    upper_price = round(current_price * (1 + (i * self.grid_spacing)), decimal_places)
                    upper_size = max(allocation_per_grid / upper_price, min_size)
                    levels.append({
                        'price': upper_price,
                        'size': round(upper_size, 8),
                        'side': 'sell'
                    })
                    
                    # Lower grid (buy)
                    lower_price = round(current_price * (1 - (i * self.grid_spacing)), decimal_places)
                    lower_size = max(allocation_per_grid / lower_price, min_size)
                    levels.append({
                        'price': lower_price,
                        'size': round(lower_size, 8),
                        'side': 'buy'
                    })
            
            logger.info(f"\nGrid Configuration for {symbol}:")
            logger.info(f"Type: {'Spot Buy-Only' if is_spot else 'Futures Buy/Sell'}")
            logger.info(f"Current Price: ${current_price}")
            logger.info(f"Grid Count: {len(levels)}")
            for level in levels:
                logger.info(f"Price: ${level['price']}, Size: {level['size']}, Side: {level['side']}")
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating grid levels: {e}")
            return []

    async def calculate_position_size(self, price: float, symbol: str = None) -> float:
        """Calculate position size with dynamic account scaling and volatility"""
        try:
            account_value = self.last_balance
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 POSITION SIZING FOR {symbol}")
            logger.info(f"Account Value: ${account_value:.2f}")
            logger.info(f"Current Price: ${price:.2f}")

            # Initialize volatility
            volatility = 1.0  # Default value
            try:
                volatility = await self.get_asset_volatility(symbol)
            except Exception as e:
                logger.warning(f"Using default volatility: {e}")

            # Dynamic allocation based on account size AND number of grid levels
            grid_levels = 4  # Typical number of grid levels
            allocation_scale = {
                (0, 50): 0.25/grid_levels,    # 25% total for micro accounts (<$50)
                (50, 100): 0.20/grid_levels,  # 20% total for very small accounts
                (100, 500): 0.15/grid_levels, # 15% total for small accounts
                (500, 1000): 0.12/grid_levels,# 12% total for medium accounts
                (1000, float('inf')): 0.10/grid_levels  # 10% total for larger accounts
            }

            # Find appropriate allocation percentage
            allocation_percent = 0.05  # Default to 5%
            for (min_val, max_val), alloc in allocation_scale.items():
                if min_val <= account_value < max_val:
                    allocation_percent = alloc
                    break

            # Get Kraken's minimum requirements
            try:
                markets = await self.exchange.fetch_markets()
                market_info = next((m for m in markets if m['symbol'] == symbol), None)
                if market_info:
                    min_amount = float(market_info.get('limits', {}).get('amount', {}).get('min', 0))
                    min_cost = float(market_info.get('limits', {}).get('cost', {}).get('min', 0))
                    
                    logger.info(f"Exchange Minimums:")
                    logger.info(f"Min Amount: {min_amount}")
                    logger.info(f"Min Cost: ${min_cost}")
                else:
                    raise Exception(f"Market info not found for {symbol}")
                    
            except Exception as e:
                logger.warning(f"Could not fetch market limits: {e}")
                # Strict fallback minimums based on logs
                min_requirements = {
                    'XRP/USD': 2.0,    # From logs: Min Amount: 2.0
                    'SPX/USD': 3.5,    # From logs: Min Amount: 3.5
                    'USDT/USD': 5.0,   # From logs: Min Amount: 5.0
                    'SOL/USD': 0.1,    # Conservative estimate
                    'BTC/USD': 0.0001, # Standard BTC minimum
                    'ETH/USD': 0.01,   # Standard ETH minimum
                    'default': 5.0     # Safe default
                }
                min_amount = min_requirements.get(symbol, min_requirements['default'])
                min_cost = min_amount * price

            # Calculate initial allocation (40% for small accounts)
            initial_allocation = account_value * allocation_percent
            position_size = initial_allocation / price

            # CRITICAL: Ensure we meet minimum size requirements
            position_size = max(position_size, min_amount)
            
            logger.info(f"Base size before minimums: {position_size:.4f}")
            logger.info(f"Required minimum: {min_amount:.4f}")

            # Apply volatility scaling if we're above minimums
            if position_size > min_amount:
                try:
                    vol_scale = {
                        (0, 1): 1.0,      # Very low vol: normal size
                        (1, 2): 0.9,      # Low vol: slightly reduced
                        (2, 3): 0.8,      # Moderate vol: reduced
                        (3, 5): 0.7,      # High vol: significantly reduced
                        (5, 8): 0.6,      # Very high vol: minimum size
                        (8, float('inf')): 0.4  # Extreme vol: ultra conservative
                    }
                    
                    vol_factor = 0.4  # Default to most conservative
                    for (vol_min, vol_max), scale in vol_scale.items():
                        if vol_min <= volatility < vol_max:
                            vol_factor = scale
                            break
                    
                    position_size *= vol_factor
                    position_size = max(position_size, min_amount)  # Ensure we still meet minimums
                    logger.info(f"Volatility: {volatility:.2f}% (Scale: {vol_factor}x)")
                    
                except Exception as e:
                    logger.warning(f"Using default volatility scaling: {e}")

            # Calculate base position size
            position_size = initial_allocation / price
            
            # Ensure we meet exchange minimums
            min_required = max(
                min_amount,
                min_cost / price if min_cost else 0
            )
            position_size = max(position_size, min_required)
            
            logger.info(f"Base size: {position_size:.4f}")
            logger.info(f"Min required: {min_required:.4f}")

            # Keep your existing entry scaling logic
            if not hasattr(self, 'entry_counts'):
                self.entry_counts = {}
            entry_count = self.entry_counts.get(symbol, 0) + 1
            self.entry_counts[symbol] = entry_count

            # Enhanced scaling based on volatility
            if entry_count <= 3:
                if volatility <= 3:
                    base_scale = {1: 1.0, 2: 1.5, 3: 2.0}.get(entry_count, 1.0)
                elif volatility <= 5:
                    base_scale = {1: 1.0, 2: 1.3, 3: 1.6}.get(entry_count, 1.0)
                else:
                    base_scale = {1: 1.0, 2: 1.2, 3: 1.4}.get(entry_count, 1.0)
            
                position_size *= base_scale
                logger.info(f"Entry scaling: {base_scale}x (Entry #{entry_count})")

            # Final rounding based on price range
            if price >= 20000:  # BTC
                position_size = round(position_size, 3)
            elif price >= 1000:  # ETH
                position_size = round(position_size, 2)
            else:
                position_size = round(position_size, 1)

            # Final validation
            position_value = position_size * price
            if position_value > account_value * 0.40:  # Cap at 40% of account
                position_size = (account_value * 0.40) / price
                position_size = max(position_size, min_amount)  # Still ensure minimum
                logger.warning("Position size reduced to stay within 40% account limit")

            logger.info(f"\nFinal Position Details:")
            logger.info(f"Size: {position_size:.4f}")
            logger.info(f"Value: ${position_size * price:.2f}")
            logger.info(f"Account %: {(position_size * price / account_value)*100:.1f}%")
            logger.info(f"{'='*50}\n")

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def manage_stop_loss(self, symbol: str, position: Dict, current_price: float) -> None:
        """Manage stop loss and take profits for both spot and margin positions"""
        try:
            # Get position details
            entry_price = float(position['info']['price'])
            position_side = position['info'].get('side', 'long')  # Default to 'long' for spot
            position_size = float(position['info'].get('size', 0))
            
            # Initialize position tracking if not exists
            if symbol not in self.positions:
                # Get accurate entry from recent orders first
                try:
                    # FORCE spot check first before anything else
                    balance = await self.exchange.fetch_balance()
                    raw_asset = symbol.split('/')[0].replace(':USD', '')  # Remove :USD suffix
                    asset = raw_asset
                    spot_balance = float(balance.get(asset, {}).get('free', 0))
                    total_balance = float(balance.get(asset, {}).get('total', 0))
                    is_spot = True  # FORCE SPOT for now
                    
                    logger.info(f"\n{'='*50}")
                    logger.info(f"FORCING SPOT MODE for {symbol}:")
                    logger.info(f"Asset: {asset}")
                    logger.info(f"Free Balance: {spot_balance}")
                    logger.info(f"Total Balance: {total_balance}")
                    logger.info(f"Is Spot: {is_spot}")
                    logger.info(f"{'='*50}\n")
                    
                    # Then get entry price with correct symbol format
                    clean_symbol = f"{raw_asset}/USD"  # Use clean symbol format
                    orders = await self.exchange.fetch_closed_orders(clean_symbol, limit=5)
                    entry_orders = [order for order in orders 
                                  if order['status'] == 'closed' 
                                  and order['side'] == 'buy']
                    
                    if entry_orders:
                        entry_price = float(entry_orders[0]['price'])
                    else:
                        # Fallback to position info
                        entry_price = float(position['info'].get('price', current_price))
                    
                    self.positions[symbol] = {
                        'trailing_stop': None,
                        'tp1_triggered': False,
                        'tp2_triggered': False,
                        'breakeven_triggered': False,
                        'trailing_active': False,
                        'stop_buffer': 0.002,
                        'original_size': position_size,
                        'entry_price': entry_price,
                        'is_spot': is_spot,
                        'spot_balance': max(spot_balance, total_balance)
                    }
                    
                    logger.info(f"Position Initialized:")
                    logger.info(f"Type: {'Spot' if is_spot else 'Margin'}")
                    logger.info(f"Entry: ${entry_price}")
                    logger.info(f"Size: {position_size}")
                    
                except Exception as e:
                    logger.error(f"Error initializing position tracking: {e}")
                    return
                
            position_data = self.positions[symbol]
            entry_price = position_data['entry_price']  # Changed from initial_entry_price to match
            is_spot = position_data.get('is_spot', False)
            
            # Get mark price with validation
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                if is_spot:
                    current_price = float(ticker['last'])  # Use last price for spot
                else:
                    mark_price = float(ticker.get('mark', ticker['last']))  # Use mark price for futures
                    last_price = float(ticker['last'])
                    
                    # Add price sanity check for futures
                    if abs((mark_price - last_price) / last_price) > 0.01:  # 1% deviation
                        logger.warning(f"Large price deviation for {symbol}:")
                        logger.warning(f"Mark Price: {mark_price}")
                        logger.warning(f"Last Price: {last_price}")
                        # Use the more conservative price
                        current_price = min(mark_price, last_price) if position_side == 'long' else max(mark_price, last_price)
                    else:
                        current_price = mark_price
                
                # Add market data validation
                if current_price <= 0 or current_price > 1000000:  # Basic sanity check
                    logger.error(f"Invalid price received for {symbol}: {current_price}")
                    return
                
            except Exception as e:
                logger.error(f"Error getting market price: {e}")
                return

            # Calculate profit percentage with detailed logging
            try:
                if is_spot:
                    # For spot positions: simple calculation without leverage
                    raw_calc = (current_price - entry_price) / entry_price
                    profit_pct = raw_calc * 100
                else:
                    if position_side == 'short':
                        raw_calc = (entry_price - current_price) / entry_price
                        leverage = self.calculate_safe_leverage(self.last_balance)
                        profit_pct = raw_calc * leverage * 100
                    else:
                        raw_calc = (current_price - entry_price) / entry_price
                        leverage = self.calculate_safe_leverage(self.last_balance)
                        profit_pct = raw_calc * leverage * 100
                
                logger.info(f"\n{'='*50}")
                logger.info(f"PROFIT CALCULATION - {symbol} ({'Spot' if is_spot else 'Margin'})")
                logger.info(f"Account Size: ${self.last_balance:.2f}")
                if not is_spot:
                    logger.info(f"Dynamic Leverage: {leverage}x")
                logger.info(f"Entry: {entry_price}")
                logger.info(f"Current: {current_price}")
                logger.info(f"Raw %: {raw_calc * 100:.2f}%")
                logger.info(f"Final ROE: {profit_pct:.2f}%")
                
            except Exception as e:
                logger.error(f"Error calculating profit: {e}")
                return

            # Get most recent entry price from order history
            try:
                orders = await self.exchange.fetch_open_orders(symbol, limit=5)
                recent_buys = [o for o in orders if o['side'] == 'buy' and o['status'] == 'open']
                if recent_buys:
                    entry_price = float(recent_buys[0]['price'])
                    logger.info(f"Using most recent entry price: ${entry_price}")
                
                # Add base stop loss check (-10% from ACTUAL entry)
                stop_price = entry_price * 0.90  # Fixed -10% stop
                
                # Check if we should stop out
                if current_price <= stop_price:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"🚨 STOP LOSS CHECK for {symbol}")
                    logger.info(f"Recent Entry: ${entry_price:.4f}")
                    logger.info(f"Current: ${current_price:.4f}")
                    logger.info(f"Stop Price: ${stop_price:.4f}")
                    logger.info(f"P&L: {profit_pct:.2f}%")
                    
                    # Only execute if actually in loss
                    if current_price < entry_price:
                        logger.info("Confirming actual loss before stop")
                        
                        # Get minimum volume requirements
                        min_volumes = {
                            'XBT': 0.0001,  # Bitcoin
                            'ETH': 0.01,    # Ethereum
                            'SOL': 0.1,     # Solana
                            'XRP': 30.0,    # Ripple minimum
                            'default': 5.0   # Default minimum
                        }
                        
                        # Get asset and its minimum volume
                        asset = symbol.split('/')[0]
                        min_volume = min_volumes.get(asset, min_volumes['default'])
                        
                        # Get actual balance
                        try:
                            balance = await self.exchange.fetch_balance()
                            actual_balance = float(balance.get(asset, {}).get('free', 0))
                            logger.info(f"Actual {asset} balance: {actual_balance}")
                            logger.info(f"Minimum volume required: {min_volume}")
                            
                            # Take full position if our balance is less than minimum requirement
                            if actual_balance < min_volume:
                                logger.info(f"Balance {actual_balance} below minimum {min_volume}, taking full position")
                                position_size = actual_balance
                            
                            # Execute the stop loss with updated size
                            await self.execute_stop_loss(
                                symbol=symbol,
                                position_size=position_size,
                                is_spot=is_spot,
                                reason="stop loss"
                            )
                            return
                            
                        except Exception as e:
                            logger.error(f"Error checking balance for minimum volume: {e}")
                            return
                    else:
                        logger.info("Position in profit, skipping stop loss")
                        return
                
            except Exception as e:
                logger.error(f"Error checking stop loss: {e}")
                return

            # Move stop to breakeven (only if we're in profit)
            if not position_data.get('breakeven_triggered', False) and profit_pct > 0:
                # Calculate fees for buffer
                total_fees = self.calculate_total_fees(current_price, position_size, symbol)
                fee_percentage = (total_fees / (current_price * position_size)) * 100
                dynamic_buffer = max(0.002, fee_percentage * 1.1)
                
                # Higher threshold for breakeven (5% for spot)
                breakeven_threshold = 0.05 if is_spot else 0.04
                
                if profit_pct >= (breakeven_threshold * 100):
                    buffer = entry_price * dynamic_buffer
                    breakeven_stop = entry_price + buffer  # Small buffer above entry
                    position_data['trailing_stop'] = breakeven_stop
                    position_data['breakeven_triggered'] = True
                    logger.info(f"Moving stop to breakeven at ${breakeven_stop:.4f} ({profit_pct:.2f}% profit)")

            # Take profit logic (adjusted for spot)
            tp_threshold = 5.0 if is_spot else 33.0  # 5% for spot, 33% for margin
            if profit_pct >= tp_threshold and not position_data['tp1_triggered']:
                logger.info(f"🎯 TP1 TRIGGERED at {profit_pct:.2f}%")
                
                try:
                    # Exchange minimum precision requirements
                    min_precision = {
                        'XBTUSD': 0.0001,   # BTC minimum
                        'ETHUSD': 0.01,     # ETH minimum
                        'SOLUSD': 0.01,     # SOL minimum
                        'AVAXUSD': 0.1,     # AVAX minimum
                        'DOTUSD': 0.1,      # DOT minimum
                        'FILUSD': 0.1,      # FIL minimum
                        'UNIUSD': 0.1,      # UNI minimum
                        'APEUSD': 0.1,      # APE minimum
                        'XRPUSD': 0.1,      # XRP minimum
                        'XLMUSD': 1.0,      # XLM minimum
                        'ATOMUSD': 0.1,     # ATOM minimum
                        'NEARUSD': 0.1,     # NEAR minimum
                        'OPUSD': 1.0,       # OP minimum
                        'DOGEUSD': 1.0,     # DOGE minimum
                        'FTMUSD': 1.0,      # FTM minimum
                        'LDOUSD': 0.1,      # LDO minimum
                        'MATICUSD': 1.0,    # MATIC minimum
                        'XMRUSD': 0.01,     # XMR minimum (higher value coin)
                        'LINKUSD': 0.1,     # LINK minimum
                        'AAVEUSD': 0.01,    # AAVE minimum (higher value coin)
                        'BATUSD': 1.0,      # BAT minimum
                        'ADAUSD': 1.0       # ADA minimum
                    }
                    
                    # Get minimum precision for this asset
                    asset = symbol.replace('PF_', '').replace('PI_', '').replace('/USD', '')
                    min_size = min_precision.get(asset, 0.1)
                    
                    # Calculate TP size (smaller for spot)
                    tp_size = round(position_size * (0.30 if is_spot else 0.33), 8)  # 30% for spot, 33% for margin
                    
                    # If position is too small for 33%, take full position
                    if tp_size < min_size:
                        logger.info(f"Position too small for partial TP ({tp_size} < {min_size}), taking full position")
                        tp_size = position_size
                        position_data['tp2_triggered'] = True
                    
                    logger.info(f"🎯 EXECUTING TP1 at {profit_pct:.2f}% with size {tp_size}")
                    
                    # Execute the TP
                    tp_executed = await self.execute_take_profit(
                        symbol=symbol,
                        position_side=position_side,
                        size=tp_size,
                        price=current_price,
                        tp_type="TP1",
                        is_spot=is_spot
                    )
                    
                    if tp_executed:
                        position_data['tp1_triggered'] = True
                        if tp_size == position_size:  # If we took full position
                            position_data['tp2_triggered'] = True  # Skip remaining TPs
                            position_data['current_position_size'] = 0
                        else:
                            position_data['current_position_size'] = position_size * 0.70  # Track remaining 70%
                        logger.info(f"✅ TP1 executed successfully")
                    else:
                        logger.error(f"❌ Failed to execute TP1")
                    
                except Exception as e:
                    logger.error(f"Error executing TP1: {e}")

            # TP2 logic (adjusted for spot)
            tp2_threshold = 10.0 if is_spot else 66.0  # 10% for spot, 66% for margin
            if position_data['tp1_triggered'] and not position_data['tp2_triggered'] and profit_pct >= tp2_threshold:
                logger.info(f"🎯 EXECUTING TP2 for {symbol}")
                logger.info(f"Entry: {entry_price}")
                logger.info(f"Current: {current_price}")
                logger.info(f"P&L: {profit_pct:.2f}%")
                
                try:
                    # Get minimum volume requirements
                    min_volumes = {
                        'XBT': 0.0001,  # Bitcoin
                        'ETH': 0.01,    # Ethereum
                        'SOL': 0.1,     # Solana
                        'XRP': 30.0,    # Ripple minimum
                        'SPX': 5.0,     # SPX minimum
                        'default': 5.0   # Default minimum
                    }
                    
                    # Get asset and its minimum volume
                    asset = symbol.split('/')[0]
                    min_volume = min_volumes.get(asset, min_volumes['default'])
                    
                    # Get actual balance
                    balance = await self.exchange.fetch_balance()
                    actual_balance = float(balance.get(asset, {}).get('free', 0))
                    logger.info(f"Current balance: {actual_balance} {asset}")
                    logger.info(f"Minimum volume required: {min_volume}")
                    
                    # Calculate TP2 size (30% of original position)
                    tp_size = round(position_size * 0.30, 8)  # 30% for TP2
                    logger.info(f"Calculated TP2 size: {tp_size}")
                    
                    # If either TP size or remaining balance is below minimum, take full position
                    if tp_size < min_volume or actual_balance < min_volume:
                        logger.info(f"Position/TP size below minimum {min_volume}, taking full position")
                        tp_size = actual_balance
                    
                    # Execute TP2 only if we have enough to meet minimum
                    if tp_size >= min_volume:
                        logger.info(f"Executing TP2 with size: {tp_size}")
                        tp_executed = await self.execute_take_profit(
                            symbol=symbol,
                            position_side=position_side,
                            size=tp_size,
                            price=current_price,
                            tp_type="TP2",
                            is_spot=is_spot
                        )
                        
                        if tp_executed:
                            position_data['tp2_triggered'] = True
                            position_data['trailing_active'] = True
                            position_data['current_position_size'] = actual_balance - tp_size
                            logger.info(f"✅ TP2 executed successfully")
                            logger.info(f"Remaining position size: {position_data['current_position_size']}")
                        else:
                            logger.error(f"❌ Failed to execute TP2")
                    else:
                        logger.warning(f"Cannot execute TP2 - Position size {tp_size} below minimum {min_volume}")
                        
                except Exception as e:
                    logger.error(f"Error executing TP2: {e}")

            # Trailing stop logic (TP3 - final 40%)
            if position_data['tp2_triggered']:
                try:
                    volatility = await self.get_asset_volatility(symbol)
                    position_data['trailing_active'] = True
                    remaining_size = position_data['current_position_size']  # Should be 40% of original
                    logger.info(f"Monitoring trailing stop for remaining {remaining_size} position")
                    
                    # Tighter trailing distances for spot
                    if is_spot:
                        if profit_pct >= 15.0:  # Start trailing at 15%
                            trailing_distance = 0.008  # 0.8%
                        elif profit_pct >= 12.0:
                            trailing_distance = 0.01   # 1.0%
                        else:
                            trailing_distance = 0.012  # 1.2%
                    else:
                        # Margin trailing distances
                        if profit_pct >= 66.0:
                            trailing_distance = min(0.010, max(0.008, volatility * 0.3))
                        elif profit_pct >= 50.0:
                            trailing_distance = min(0.015, max(0.012, volatility * 0.4))
                        else:
                            trailing_distance = min(0.018, max(0.015, volatility * 0.5))
                    
                    new_stop = current_price * (1 - trailing_distance)
                    
                    # Update trailing stop if new stop is higher
                    if not position_data.get('trailing_stop') or new_stop > position_data['trailing_stop']:
                        position_data['trailing_stop'] = new_stop
                        logger.info(f"🔄 Updated trailing stop: {new_stop:.4f} - Distance: {trailing_distance*100:.1f}%")
                    
                    # Check if price hits trailing stop
                    if current_price <= position_data['trailing_stop']:
                        logger.info(f"🎯 TP3 (Trailing Stop) TRIGGERED at {profit_pct:.2f}%")
                        
                        # Execute final position close
                        tp_executed = await self.execute_take_profit(
                            symbol=symbol,
                            position_side=position_side,
                            size=remaining_size,  # Close remaining 40%
                            price=current_price,
                            tp_type="TP3",
                            is_spot=is_spot
                        )
                        
                        if tp_executed:
                            logger.info(f"✅ TP3 (Trailing Stop) executed successfully")
                            position_data['current_position_size'] = 0
                            return  # Exit after trailing stop hit
                    
                except Exception as e:
                    logger.error(f"Error managing trailing stop: {e}")

            # Log position status
            logger.info(f"\nPosition Status - {symbol} ({'Spot' if is_spot else 'Margin'}):")
            logger.info(f"P&L: {profit_pct:.2f}%")
            logger.info(f"Stop: {position_data.get('trailing_stop', 'Not Set')}")
            logger.info(f"TP1: {'✅' if position_data['tp1_triggered'] else '⏳'}")
            logger.info(f"TP2: {'✅' if position_data['tp2_triggered'] else '⏳'}")
            logger.info(f"Trailing: {'✅' if position_data.get('trailing_active') else '⏳'}")

        except Exception as e:
            logger.error(f"Error in manage_stop_loss for {symbol}: {e}")

    async def execute_take_profit(self, symbol: str, position_side: str, size: float, price: float, tp_type: str, is_spot: bool = False) -> bool:
        """Execute a take profit order with retries and confirmation"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # First verify if this is actually a spot position
                try:
                    balance = await self.exchange.fetch_balance()
                    asset = symbol.split('/')[0]
                    spot_balance = float(balance.get(asset, {}).get('free', 0))
                    is_spot = spot_balance > 0  # Override is_spot based on actual balance
                    logger.info(f"\nVerified {asset} spot balance: {spot_balance}")
                except Exception as e:
                    logger.error(f"Error checking spot balance: {e}")

                logger.info(f"\n{'='*50}")
                logger.info(f"🎯 EXECUTING {tp_type.upper()} - Attempt {attempt + 1}/{max_retries}")
                logger.info(f"Symbol: {symbol}")
                logger.info(f"Side: {'sell' if position_side == 'long' else 'buy'}")
                logger.info(f"Size: {size}")
                logger.info(f"Type: {'Spot' if is_spot else 'Futures'}")
                
                # For spot orders, use the verified balance
                if is_spot:
                    if spot_balance > 0:
                        # Kraken minimum volume requirements
                        min_volumes = {
                            'XBT': 0.0001,  # Bitcoin
                            'ETH': 0.01,    # Ethereum
                            'POPCAT': 5.0,  # POPCAT minimum 5
                            'XLM': 30.0,    # Stellar minimum 30
                            'default': 5.0  # Default minimum for unknown assets
                        }
                        
                        min_volume = min_volumes.get(asset, min_volumes['default'])
                        logger.info(f"Minimum volume requirement for {asset}: {min_volume}")
                        
                        # Get position history to implement FIFO
                        try:
                            # Fetch recent orders to identify entry points
                            orders = await self.exchange.fetch_closed_orders(symbol, limit=10)
                            buy_orders = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed']
                            
                            # Sort buy orders by timestamp (oldest first for FIFO)
                            buy_orders.sort(key=lambda x: x['timestamp'])
                            
                            logger.info(f"Found {len(buy_orders)} buy orders for FIFO analysis")
                            
                            # Calculate appropriate size based on FIFO
                            if tp_type == 'TP1':
                                # For TP1, sell 30% of the oldest position
                                if buy_orders:
                                    oldest_order = buy_orders[0]
                                    oldest_filled = float(oldest_order['filled'])
                                    oldest_price = float(oldest_order['price'])
                                    
                                    logger.info(f"FIFO: Oldest position from {datetime.fromtimestamp(oldest_order['timestamp']/1000)}")
                                    logger.info(f"FIFO: Size {oldest_filled} @ ${oldest_price}")
                                    
                                    # Calculate 30% of the oldest position
                                    size = min(oldest_filled * 0.3, spot_balance)
                                    logger.info(f"FIFO: Taking 30% of oldest position: {size} {asset}")
                                else:
                                    # Fallback if no order history
                                    size = spot_balance * 0.3
                                    logger.info(f"No order history, using 30% of current balance: {size} {asset}")
                            
                            elif tp_type == 'TP2':
                                # For TP2, sell another 30% of the oldest remaining position
                                if len(buy_orders) >= 2:
                                    # If we have multiple buy orders, take from the second oldest
                                    second_oldest = buy_orders[1]
                                    second_filled = float(second_oldest['filled'])
                                    second_price = float(second_oldest['price'])
                                    
                                    logger.info(f"FIFO: Second position from {datetime.fromtimestamp(second_oldest['timestamp']/1000)}")
                                    logger.info(f"FIFO: Size {second_filled} @ ${second_price}")
                                    
                                    # Calculate 30% of the second oldest position
                                    size = min(second_filled * 0.3, spot_balance)
                                    logger.info(f"FIFO: Taking 30% of second oldest position: {size} {asset}")
                                else:
                                    # If only one buy order, take another 30% from it
                                    size = spot_balance * 0.3
                                    logger.info(f"Only one position found, using 30% of current balance: {size} {asset}")
                            
                            else:  # Final TP or trailing stop
                                # For final TP, sell remaining balance
                                size = spot_balance
                                logger.info(f"Final TP: Selling entire remaining balance: {size} {asset}")
                        
                        except Exception as e:
                            logger.error(f"Error implementing FIFO for {symbol}: {e}")
                            # Fallback to original percentage-based approach
                            if tp_type == 'TP1':
                                size = spot_balance * 0.3
                            elif tp_type == 'TP2':
                                size = spot_balance * 0.3
                            else:
                                size = spot_balance
                        
                        # Force market order for spot without reduceOnly
                        params = {
                            'ordertype': 'market',
                            'trading_agreement': 'agree'
                        }
                        
                        try:
                            # Check if size meets minimum requirements
                            if size < min_volume:
                                logger.warning(f"❗ Size {size} is below minimum {min_volume} for {asset}")
                                if spot_balance >= min_volume:
                                    logger.info(f"Adjusting to minimum volume: {min_volume}")
                                    size = min_volume
                                else:
                                    logger.info(f"Total balance {spot_balance} below minimum, selling entire position")
                                    size = spot_balance
                            
                            # Execute the order
                            order = await self.exchange.create_market_order(
                                symbol=symbol,
                                side='sell',
                                amount=size,
                                params=params
                            )
                            logger.info(f"TP executed: {size} {asset}")
                            
                            # Update position tracking for FIFO
                            if symbol in self.positions:
                                # Record which TP was triggered
                                if tp_type == 'TP1':
                                    self.positions[symbol]['tp1_triggered'] = True
                                    self.positions[symbol]['tp1_size'] = size
                                    self.positions[symbol]['tp1_time'] = time.time()
                                elif tp_type == 'TP2':
                                    self.positions[symbol]['tp2_triggered'] = True
                                    self.positions[symbol]['tp2_size'] = size
                                    self.positions[symbol]['tp2_time'] = time.time()
                                
                                # Update remaining position size
                                self.positions[symbol]['position_size'] -= size
                                logger.info(f"Updated position tracking: {self.positions[symbol]['position_size']} {asset} remaining")
                            
                            return True
                            
                        except Exception as e:
                            if "volume minimum not met" in str(e).lower():
                                logger.warning(f"❗ Partial TP failed due to minimum volume, trying full position")
                                # If partial fails, try full position
                                size = spot_balance
                                logger.info(f"Attempting full position close: {size} {asset}")
                            else:
                                raise e  # Re-raise if it's not a minimum volume issue
                                
                    else:
                        logger.error(f"No spot balance found for {asset}")
                        return False
                else:
                    # For futures/margin orders, include reduceOnly
                    params = {
                        'ordertype': 'market',
                        'reduceOnly': True,
                        'trading_agreement': 'agree'
                    }
                
                # Verify size is valid
                if size <= 0:
                    logger.error(f"Invalid size ({size}) for {tp_type} on {symbol}")
                    return False
                
                # Create the market order
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side='sell',
                    amount=size,
                    params=params
                )
                
                if order and 'id' in order:
                    logger.info(f"✅ {tp_type} executed successfully for {symbol}:")
                    logger.info(f"Order ID: {order['id']}")
                    logger.info(f"Size: {size}")
                    logger.info(f"Type: Market")
                    await self.update_trade_history(symbol, order)
                    logger.info(f"{'='*50}\n")
                    return True
                
                logger.error(f"❌ No order ID returned - Attempt {attempt + 1}")
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error executing {tp_type}: {e}")
                logger.error(f"Full error: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                continue
        
        logger.error(f"❌ Failed to execute {tp_type} after {max_retries} attempts")
        return False

    async def monitor_positions(self, symbols: List[str] = None) -> None:
        """Monitor and manage open positions with batch processing"""
        try:
            # If no symbols provided, check all active positions
            if not symbols:
                symbols = list(self.active_positions.keys())
            
            # Create batches of 4 symbols with 8 max workers
            batch_size = 4  # Balanced batch size
            max_workers = 8  # More conservative worker count
            
            async def process_symbol(symbol: str):
                try:
                    # Get mapped symbol from class attribute
                    mapped_symbol = self.symbol_map.get(symbol)
                    logger.info(f"Checking position for {symbol} (mapped: {mapped_symbol})")
                    
                    # First check and cancel any stale orders (older than 24 hours)
                    await self.cancel_stale_orders(symbol)  # Use class parameter by default
                    
                    # Check both mapped and original symbol
                    position = None
                    if mapped_symbol in self.active_positions:
                        position = self.active_positions[mapped_symbol]
                        logger.info(f"Found position using mapped symbol: {mapped_symbol}")
                    elif symbol in self.active_positions:
                        position = self.active_positions[symbol]
                        mapped_symbol = symbol
                        logger.info(f"Found position using original symbol: {symbol}")
                        
                    if position:
                        logger.error(f"\n{'='*50}")
                        logger.error(f"MONITORING POSITION - {symbol}")
                        logger.error(f"Size: {position['info'].get('size', 0)}")
                        logger.error(f"Entry: {position['info'].get('price', 'N/A')}")
                        logger.error(f"Side: {position['info'].get('side', 'N/A')}")
                        logger.error(f"{'='*50}\n")
                        
                        # Get current price with proper error handling
                        current_price = await self.get_current_price(symbol)
                        if current_price is None:
                            logger.error(f"Unable to get current price for {symbol}, skipping position monitoring")
                            return
                        
                        # Calculate raw P&L first
                        try:
                            entry_price = float(position['info']['price'])
                            position_side = position['info']['side']
                            position_size = float(position['info'].get('size', 0))
                            
                            raw_pnl_pct = (
                                ((current_price - entry_price) / entry_price * 100)
                                if position_side == 'long'
                                else ((entry_price - current_price) / entry_price * 100)
                            )
                            
                            # Calculate leveraged P&L for display
                            leveraged_pnl = raw_pnl_pct * self.leverage
                            
                            # Log both raw and leveraged P&L
                            logger.info(f"\nP&L Analysis for {symbol}:")
                            logger.info(f"Entry: ${entry_price:.4f}")
                            logger.info(f"Current: ${current_price:.4f}")
                            logger.info(f"Raw P&L: {raw_pnl_pct:.2f}%")
                            logger.info(f"Leveraged P&L: {leveraged_pnl:.2f}%")
                            
                            # QUICK PROFIT CHECK - Do this before manage_stop_loss
                            if raw_pnl_pct >= 5.0:  # TP1 at 5%
                                logger.info(f"🎯 Quick TP1 Check Triggered at {raw_pnl_pct:.2f}%")
                                tp_size = position_size * 0.3
                                tp_executed = await self.execute_take_profit(
                                    symbol=symbol,
                                    position_side=position_side,
                                    size=tp_size,
                                    price=current_price,
                                    tp_type="TP1",
                                    is_spot=True
                                )
                                if tp_executed:
                                    return
                            
                            if raw_pnl_pct >= 10.0:  # TP2 at 10%
                                logger.info(f"🎯 Quick TP2 Check Triggered at {raw_pnl_pct:.2f}%")
                                tp_size = position_size * 0.3
                                tp_executed = await self.execute_take_profit(
                                    symbol=symbol,
                                    position_side=position_side,
                                    size=tp_size,
                                    price=current_price,
                                    tp_type="TP2",
                                    is_spot=True
                                )
                                if tp_executed:
                                    return
                            
                            # Continue with regular position management if no TPs executed
                            await self.manage_stop_loss(symbol, position, current_price)
                            
                        except Exception as e:
                            logger.error(f"Error calculating P&L for {symbol}: {e}")
                        
                    else:
                        logger.info(f"No active position found for {symbol} ({mapped_symbol})")
                        
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
            
            # Process symbols in batches with semaphore for max concurrency
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_with_semaphore(symbol):
                async with semaphore:
                    await process_symbol(symbol)
            
            # Process batches
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                tasks = [process_with_semaphore(symbol) for symbol in batch]
                await asyncio.gather(*tasks)
                
        except Exception as e:
            logger.error(f"Error in batch position monitoring: {e}")

    async def start(self):
        while True:  # Add this outer loop
            try:
                # Add a new flag to track if we have sufficient funds
                self.has_sufficient_funds = True
                
                logger.info("Starting grid trading strategy...")
                
                # Initialize symbol mappings
                self.symbol_map = {
                    'PF_XBTUSD': 'BTC/USD:USD',
                    'PF_XRPUSD': 'XRP/USD:USD',
                    'PI_ETHUSD': 'ETH/USD:ETH',
                    'PF_DOGEUSD': 'DOGE/USD:USD',
                    'PF_LDOUSD': 'LDO/USD:USD',
                    'PF_ADAUSD': 'ADA/USD:USD',
                    'PF_MATICUSD': 'MATIC/USD:USD',
                    'PF_FILUSD': 'FIL/USD:USD',
                    'PF_APEUSD': 'APE/USD:USD',
                    'PF_GMXUSD': 'GMX/USD:USD',
                    'PF_BATUSD': 'BAT/USD:USD',
                    'PF_XLMUSD': 'XLM/USD:USD',
                    'PF_EOSUSD': 'EOS/USD:USD',
                    'PF_OPUSD': 'OP/USD:USD',
                    'PF_AAVEUSD': 'AAVE/USD:USD',
                    'PF_LINKUSD': 'LINK/USD:USD',
                    'PF_XMRUSD': 'XMR/USD:USD',
                    'PF_ATOMUSD': 'ATOM/USD:USD',
                    'PF_DOTUSD': 'DOT/USD:USD',
                    'PF_ALGOUSD': 'ALGO/USD:USD',
                    'PF_TRXUSD': 'TRX/USD:USD',
                    'PF_SOLUSD': 'SOL/USD:USD',
                    'PF_AVAXUSD': 'AVAX/USD:USD',
                    'PF_UNIUSD': 'UNI/USD:USD',
                    'PF_SNXUSD': 'SNX/USD:USD',
                    'PF_NEARUSD': 'NEAR/USD:USD',
                    'PF_FTMUSD': 'FTM/USD:USD',
                    'PF_ARBUSD': 'ARB/USD:USD',
                    'PF_COMPUSD': 'COMP/USD:USD',
                    'PF_YFIUSD': 'YFI/USD:USD'
                }
                self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
                
                # Initialize tracking dictionaries
                self.positions = {}
                self.last_grid_check = {}
                self.active_positions = {}
                
                # Retry loop for exchange connection
                retry_count = 0
                while retry_count < 5:
                    try:
                        # Get initial balance
                        self.last_balance = await self.get_account_balance()
                        logger.info(f"Initial balance: {self.last_balance}")
                        
                        # Verify spot positions
                        balance = await self.exchange.fetch_balance()
                        for symbol in self.symbol_map.values():
                            asset = symbol.split('/')[0]
                            spot_balance = float(balance.get(asset, {}).get('free', 0))
                            
                            if spot_balance > 0:
                                logger.error(f"\n{'='*50}")
                                logger.error(f"FOUND SPOT POSITION - {symbol}")
                                logger.error(f"Asset: {asset}")
                                logger.error(f"Balance: {spot_balance}")
                                
                                # Get entry price from recent orders
                                try:
                                    orders = await self.exchange.fetch_closed_orders(symbol, limit=5)
                                    buy_orders = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed']
                                    entry_price = float(buy_orders[0]['price']) if buy_orders else None
                                except Exception as e:
                                    logger.error(f"Error getting entry price: {e}")
                                    entry_price = None
                                
                                if entry_price:
                                    logger.error(f"Entry Price: ${entry_price}")
                                    # Initialize position tracking
                                    self.positions[symbol] = {
                                        'entry_price': entry_price,
                                        'position_size': spot_balance,
                                        'side': 'buy',
                                        'trailing_stop': None,
                                        'tp1_triggered': False,
                                        'tp2_triggered': False,
                                        'breakeven_triggered': False,
                                        'trailing_active': False,
                                        'is_spot': True,
                                        'stop_buffer': 0.02
                                    }
                                logger.error(f"{'='*50}\n")
                        
                        # Get initial positions
                        positions = await self.verify_actual_positions() or []
                        logger.info(f"Found {len(positions)} positions on startup")
                        
                        break  # Break retry loop on success
                    except ccxt.RequestTimeout:
                        retry_count += 1
                        logger.warning(f"Timeout connecting to exchange. Retry {retry_count}/5")
                        await asyncio.sleep(30)
                        if retry_count == 5:
                            logger.error("Max retries reached, waiting 5 minutes before restarting")
                            await asyncio.sleep(300)
                            continue  # Restart from the beginning
                            
                # Start background market scanner
                self.market_scan_task = asyncio.create_task(self.background_market_scan())
                logger.info("Started background market scanner")
                
                # Initial high-priority symbols
                self.active_symbols = ['BTC/USD:USD', 'ETH/USD:ETH', 'SOL/USD:USD']
                logger.info(f"Starting with initial symbols: {self.active_symbols}")
                
                # Main trading loop
                while True:
                    try:
                        current_time = time.time()
                        
                        # Get fresh positions data
                        positions = await self.get_open_positions()
                        self.active_positions = {p['symbol']: p for p in positions}
                        
                        # Monitor existing positions
                        logger.info("\n=== MONITORING ACTIVE POSITIONS ===")
                        if self.active_positions:
                            await self.monitor_positions()  # Will now process all positions in batches
                        logger.info("=== FINISHED POSITION MONITORING ===\n")
                        
                        # Update and process trading symbols
                        await self.update_trading_symbols()
                        for symbol in self.active_symbols:
                            try:
                                df = await self.get_historical_data(symbol)
                                if df is not None and current_time - self.last_grid_check.get(symbol, 0) > 900:
                                    logger.info(f"Reassessing grids for {symbol}")
                                    await self.reassess_grids(symbol)
                                    self.last_grid_check[symbol] = current_time
                                    continue
                                    
                                await self.execute_grid_orders(symbol)
                            except Exception as e:
                                logger.error(f"Error processing symbol {symbol}: {e}")
                                continue
                            
                        await asyncio.sleep(5)  # 5-second interval
                        
                    except ccxt.RequestTimeout:
                        logger.warning("Timeout in main loop, continuing...")
                        await asyncio.sleep(30)
                        continue
                    except ccxt.InsufficientFunds:
                        logger.warning("Insufficient funds detected - switching to position management only mode")
                        self.has_sufficient_funds = False
                        # Continue execution to manage existing positions
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}")
                        await asyncio.sleep(30)
                        continue
                        
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                logger.error("Restarting strategy in 5 minutes...")
                await asyncio.sleep(300)
                continue  # Restart from the beginning
            finally:
                if hasattr(self, 'market_scan_task'):
                    self.market_scan_task.cancel()
                try:
                    await self.exchange.close()
                except:
                    pass

    async def background_market_scan(self):
        """Background task to scan and update available markets"""
        try:
            while True:
                logger.info("Running background market scan...")
                
                # Only scan for new symbols if we have sufficient funds
                if self.has_sufficient_funds:
                    # Update available symbols based on balance
                    available_balance = await self.get_available_balance()
                    
                    new_symbols = ['BTC/USD:USD', 'ETH/USD:ETH', 'SOL/USD:USD']  # Base tier
                    
                    if available_balance >= 500:
                        new_symbols.extend(['ATOM/USD:USD', 'LINK/USD:USD'])
                    if available_balance >= 2000:
                        new_symbols.extend(['AAVE/USD:USD', 'UNI/USD:USD'])
                    if available_balance >= 5000:
                        new_symbols.extend(['DOT/USD:USD', 'AVAX/USD:USD'])
                        
                    # Update active symbols
                    self.active_symbols = list(set(new_symbols))
                    logger.info(f"Updated active symbols: {self.active_symbols}")
                else:
                    # Only track symbols with active positions
                    self.active_symbols = list(self.active_positions.keys())
                    logger.info(f"Insufficient funds - Only monitoring active positions: {self.active_symbols}")
                
                # Check for and cancel stale orders across all active symbols
                logger.info("Checking for stale orders across all active symbols...")
                for symbol in self.active_symbols:
                    try:
                        await self.cancel_stale_orders(symbol)  # Use class parameter by default
                        # Brief pause between API calls to avoid rate limits
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error checking stale orders for {symbol}: {e}")
                
                await asyncio.sleep(3600)  # Scan every hour
                
        except asyncio.CancelledError:
            logger.info("Background market scanner stopped")
        except Exception as e:
            logger.error(f"Error in background market scanner: {e}")

    async def verify_stop_orders(self, symbol: str, position: dict) -> None:
        try:
            open_orders = await self.exchange.fetch_closed_orders(symbol)
            entry_price = float(position['info']['price'])
            position_side = position['info']['side']
            
            # Check for stop orders
            has_stop = any(
                order['type'] == 'stop' and 
                (order['side'] == 'sell' if position_side == 'long' else order['side'] == 'buy')
                for order in open_orders
            )
            
            if not has_stop:
                logger.error(f"No stop loss found for {symbol} position at {entry_price}")
                # Place emergency stop
        except Exception as e:
            logger.error(f"Stop verification failed: {e}")

    async def check_active_stops(self, symbol: str) -> None:
        """Check active stops and take profits for spot positions"""
        try:
            # Get open orders
            open_orders = await self.exchange.fetch_closed_orders(symbol)
            
            # Get current balance and entry price
            balance = await self.exchange.fetch_balance()
            asset = symbol.split('/')[0]  # e.g., 'XRP' from 'XRP/USD'
            position_size = float(balance.get(asset, {}).get('total', 0))
            
            if position_size > 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Position Check for {symbol}:")
                logger.info(f"Current {asset} Balance: {position_size}")
                
                # Categorize orders with progression
                initial_stops = [o for o in open_orders if o['type'] == 'stop-loss' and not o.get('info', {}).get('triggered')]
                breakeven_stops = [o for o in open_orders if o['type'] == 'stop-loss' and o.get('info', {}).get('triggered')]
                tp1_orders = [o for o in open_orders if o['type'] == 'take-profit' and float(o['price']) <= self.tp1_level]
                tp2_orders = [o for o in open_orders if o['type'] == 'take-profit' and self.tp1_level < float(o['price']) <= self.tp2_level]
                trailing_stops = [o for o in open_orders if o['type'] in ['trailing-stop', 'trailing-stop-limit']]
                
                logger.info("\nStop Loss Progression:")
                logger.info(f"Initial Stop (-10%): {'✅' if initial_stops else '❌'}")
                for stop in initial_stops:
                    logger.info(f"- Stop at ${float(stop['price']):.4f}")
                    
                logger.info(f"Breakeven Stop: {'✅' if breakeven_stops else '❌'}")
                for stop in breakeven_stops:
                    logger.info(f"- Stop at ${float(stop['price']):.4f}")
                
                logger.info("\nTake Profit Progression:")
                logger.info(f"TP1 (33%): {'✅' if tp1_orders else '❌'}")
                for tp in tp1_orders:
                    logger.info(f"- Take profit at ${float(tp['price']):.4f}")
                    
                logger.info(f"TP2 (33%): {'✅' if tp2_orders else '❌'}")
                for tp in tp2_orders:
                    logger.info(f"- Take profit at ${float(tp['price']):.4f}")
                
                logger.info(f"Trailing Stop (34%): {'✅' if trailing_stops else '❌'}")
                for trail in trailing_stops:
                    logger.info(f"- Trailing stop at ${float(trail['price']):.4f}")
                
                logger.info(f"{'='*50}\n")
                    
            else:
                logger.info(f"No active position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error checking stops: {e}")

    async def verify_stop_placement(self, symbol: str, position_side: str, stop_price: float, size: float) -> bool:
        """Verify stop order was placed and place emergency stop if needed"""
        try:
            # Get all open orders
            open_orders = await self.exchange.fetch_closed_orders(symbol)
            
            # Check for existing stop orders
            stop_orders = [
                order for order in open_orders 
                if order['type'] == 'stop' and 
                abs(float(order['price']) - stop_price) / stop_price < 0.001  # Within 0.1% of target price
            ]
            
            if not stop_orders:
                logger.warning(f"No stop order found at {stop_price} for {symbol}, placing emergency stop")
                
                try:
                    # Place emergency stop market order
                    emergency_stop = await self.exchange.create_order(
                        symbol=symbol,
                        type='stop',
                        side='sell' if position_side == 'long' else 'buy',
                        amount=size,
                        price=stop_price,
                        params={
                            'reduceOnly': True,
                            'stopPrice': stop_price
                        }
                    )
                    
                    logger.info(f"Emergency stop placed for {symbol}:")
                    logger.info(f"Side: {'sell' if position_side == 'long' else 'buy'}")
                    logger.info(f"Size: {size}")
                    logger.info(f"Stop Price: {stop_price}")
                    logger.info(f"Order ID: {emergency_stop['id']}")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to place emergency stop for {symbol}: {e}")
                    return False
            
            # Log existing stop orders
            for stop in stop_orders:
                logger.info(f"Verified stop order for {symbol}:")
                logger.info(f"Price: {stop['price']}")
                logger.info(f"Size: {stop['amount']}")
                logger.info(f"Order ID: {stop['id']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying stop placement for {symbol}: {e}")
            return False

    async def calculate_hybrid_grids(self, symbol: str, df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float]) -> List[float]:
        """Calculate grid levels focusing exclusively on support levels for buy orders"""
        try:
            # Validate dataframe and price
            if df.empty or df['close'].iloc[-1] == 0:
                logger.error(f"Invalid price data for {symbol}")
                return []

            current_price = float(df['close'].iloc[-1])
            if current_price <= 0:
                logger.error(f"Invalid current price ({current_price}) for {symbol}")
                return []

            # Get multi-timeframe data for technical analysis
            timeframes = {
                '1d': await self.get_historical_data(symbol, '1d', 100),
                '4h': await self.get_historical_data(symbol, '4h', 100),
                '1h': await self.get_historical_data(symbol, '1h', 100),
                '15m': df
            }

            # Calculate technical indicators for each timeframe
            tech_scores = {}
            for tf, tf_df in timeframes.items():
                if tf_df is not None and not tf_df.empty:
                    # EMAs (9, 21, 200) - Using talib
                    tf_df['ema9'] = talib.EMA(tf_df['close'], timeperiod=9)
                    tf_df['ema21'] = talib.EMA(tf_df['close'], timeperiod=21)
                    tf_df['ema200'] = talib.EMA(tf_df['close'], timeperiod=200)
                    
                    # RSI
                    tf_df['rsi'] = ta.momentum.RSIIndicator(tf_df['close'], window=14).rsi()
                    
                    # MACD
                    macd = ta.trend.MACD(tf_df['close'])
                    tf_df['macd'] = macd.macd()
                    tf_df['macd_signal'] = macd.macd_signal()
                    tf_df['macd_hist'] = macd.macd_diff()
                    
                    tech_scores[tf] = {
                        'rsi': tf_df['rsi'].iloc[-1],
                        'macd_hist': tf_df['macd_hist'].iloc[-1],
                        'prev_macd_hist': tf_df['macd_hist'].iloc[-2],
                        'ema_alignment': (
                            tf_df['ema9'].iloc[-1] > tf_df['ema21'].iloc[-1] and
                            tf_df['close'].iloc[-1] > tf_df['ema200'].iloc[-1]
                        ),
                        'is_bullish': (
                            tf_df['rsi'].iloc[-1] < 40 and
                            tf_df['macd_hist'].iloc[-1] > tf_df['macd_hist'].iloc[-2] and
                            tf_df['macd_hist'].iloc[-1] > 0 and
                            tf_df['ema9'].iloc[-1] > tf_df['ema21'].iloc[-1]
                        )
                    }

            levels_with_confidence = {}  # Price level -> Confidence score
            
            # Calculate fees and adjust grid spacing first
            self.adjust_grid_spacing_for_fees(current_price)
            avg_position_size = await self.calculate_position_size(current_price, symbol)
            
            if avg_position_size <= 0:
                logger.error(f"Invalid position size for {symbol}")
                return []

            total_fees = self.calculate_total_fees(current_price, avg_position_size, symbol)
            min_profit_needed = total_fees * 2

            # Filter support levels below current price
            valid_support_levels = [level for level in support_levels if level < current_price]
            
            levels_with_confidence = {}
            
            # Analyze each support level
            for support_level in valid_support_levels:
                # Skip if support is too far from current price (>10%)
                if abs(support_level - current_price) / current_price > 0.10:
                    continue
                
                # Calculate confidence score for support level
                base_conf = 0.5  # Higher base confidence for support levels
                
                # Add volume profile boost
                try:
                    price_bins = pd.qcut(df['close'], q=10, duplicates='drop')
                    volume_profile = df.groupby(price_bins, observed=True)['volume'].sum()
                    if any(abs(support_level - level.mid) / level.mid < 0.01 for level in volume_profile.index):
                        base_conf += 0.2
                except Exception as e:
                    logger.debug(f"Volume profile calculation skipped: {e}")

                # Add technical boost
                tech_boost = self.calculate_technical_boost(support_level, {
                    '1h': {
                        'rsi': df['rsi'].iloc[-1],
                        'macd_hist': df['macd'].iloc[-1],
                        'prev_macd_hist': df['macd'].iloc[-2] if len(df) > 2 else 0,
                        'ema_alignment': True,
                        'is_bullish': df['rsi'].iloc[-1] < 40
                    }
                })
                
                final_confidence = base_conf + tech_boost
                levels_with_confidence[support_level] = final_confidence

            # Filter and sort levels
            min_confidence = 0.4  # Higher minimum confidence
            valid_levels = {
                price: conf for price, conf in levels_with_confidence.items()
                if conf >= min_confidence
            }

            # Ensure minimum spacing between levels
            min_distance = max(current_price * 0.02, min_profit_needed / (current_price * avg_position_size))
            filtered_levels = []
            
            for price in sorted(valid_levels.keys()):
                if not filtered_levels or abs(price - filtered_levels[-1]) >= min_distance:
                    filtered_levels.append(price)

            # Log results
            logger.info(f"\nBuy Grid levels for {symbol} (Support-based):")
            logger.info(f"Current price: {current_price}")
            for level in filtered_levels:
                confidence = valid_levels.get(level, 'current')
                logger.info(f"Support Buy Level: {level}, Confidence: {confidence}")

            logger.info(f"\nSupport Level Analysis for {symbol}:")
            logger.info(f"Valid support levels below current price: {[round(s, 4) for s in valid_support_levels]}")
            logger.info(f"Current price: {current_price}")

            return filtered_levels

        except Exception as e:
            logger.error(f"Error calculating hybrid grids: {e}")
            # Return basic grid levels instead of awaiting
            if current_price and current_price > 0:
                return [
                    current_price * (1 + (i * self.grid_spacing))
                    for i in range(-2, 3)
                ]
            return []

    def calculate_technical_boost(self, price: float, tech_scores: dict) -> float:
        """Calculate confidence boost based on technical analysis"""
        tech_boost = 0
        
        # Weight different timeframes
        timeframe_weights = {
            '1d': 0.4,
            '4h': 0.3,
            '1h': 0.2,
            '15m': 0.1
        }
        
        for tf, scores in tech_scores.items():
            if scores['is_bullish']:
                tech_boost += timeframe_weights[tf]
                
            # Additional boost for strong RSI signals
            if scores['rsi'] < 30:  # Heavily oversold
                tech_boost += timeframe_weights[tf] * 0.5
                
            # MACD momentum boost
            if scores['macd_hist'] > scores['prev_macd_hist']:
                tech_boost += timeframe_weights[tf] * 0.3
        
        return tech_boost

    async def update_trading_symbols(self) -> None:
        """Dynamically update trading symbols based on account growth and volume"""
        try:
            current_time = time.time()
            if current_time - self.last_symbol_update < 3600:
                return

            logger.info("\n=== Updating Trading Symbols ===")
            balance = await self.get_account_balance()
            logger.info(f"Current Balance: ${balance:.2f}")

            # Get all possible symbols first
            new_symbols = []
            if balance >= 0:     # Always include tier 1
                logger.info("Including Tier 1 symbols")
                new_symbols.extend(self.tier1_symbols)
            if balance >= 500:
                logger.info("Including Tier 2 symbols")
                new_symbols.extend(self.tier2_symbols)
                logger.info("Account above $500 - Adding Tier 2 symbols (including BTC and ATOM)")
                
            if balance >= 2000:
                logger.info("Including Tier 3 symbols")
                new_symbols.extend(self.tier3_symbols)
                logger.info("Account above $2000 - Adding Tier 3 symbols (including AAVE)")
                
            if balance >= 5000:
                logger.info("Including Tier 4 symbols")
                new_symbols.extend(self.tier4_symbols)
                logger.info("Account above $5000 - Adding ETH and remaining symbols")

            # Apply volume and price filters BEFORE setting active symbols
            logger.info(f"Filtering {len(new_symbols)} initial symbols...")
            filtered_symbols = await self.filter_symbols(new_symbols)
            
            logger.info(f"Filtered down to top {len(filtered_symbols)} symbols by volume")
            self.active_symbols = filtered_symbols
            
            self.last_symbol_update = current_time
            logger.info(f"Current active symbols: {self.active_symbols}")
            logger.info(f"Current account balance: ${balance:.2f}")

        except Exception as e:
            logger.error(f"Error updating trading symbols: {e}")

    async def initialize_position_tracking(self, order: dict, symbol: str) -> None:
        """Initialize position tracking when a new position is opened"""
        try:
            if order and 'side' in order and 'price' in order and 'amount' in order:
                # Verify spot balance
                balance = await self.exchange.fetch_balance()
                asset = symbol.split('/')[0]
                spot_balance = float(balance.get(asset, {}).get('free', 0))
                
                entry_price = float(order['price'])
                
                # For spot, we only track long positions with initial 10% stop
                initial_stop = entry_price * 0.90  # 10% stop loss for spot
                
                logger.error(f"\n{'='*50}")
                logger.error(f"NEW SPOT POSITION OPENED - {symbol}")
                logger.error(f"Asset Balance: {spot_balance} {asset}")
                logger.error(f"Entry: ${entry_price:.4f}")
                logger.error(f"Initial Stop: ${initial_stop:.4f}")
                logger.error(f"Size: {order['amount']}")
                logger.error(f"{'='*50}\n")
                
                self.positions[symbol] = {
                    'entry_price': entry_price,
                    'position_size': float(order['amount']),
                    'side': 'buy',  # Always buy for spot
                    'trailing_stop': initial_stop,
                    'tp1_triggered': False,
                    'tp2_triggered': False,
                    'breakeven_triggered': False,
                    'trailing_active': False,
                    'stop_buffer': 0.02,  # 2% buffer for spot stops
                    'is_spot': True
                }
                
                # Update active positions based on actual spot balances
                self.active_positions = {
                    symbol: {
                        'symbol': symbol,
                        'size': spot_balance,
                        'entry_price': entry_price,
                        'is_spot': True
                    }
                }
                
        except Exception as e:
            logger.error(f"Error initializing position tracking for {symbol}: {e}")

    async def reassess_grids(self, symbol: str) -> None:
        """Reassess and reposition grids based on high-confidence levels and multi-timeframe analysis"""
        try:
            # Get historical data first
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                logger.warning(f"No historical data for {symbol}")
                return

            # Get multi-timeframe trend analysis
            trend_data = await self.analyze_multi_timeframe_trend(symbol)
            
            # Log trend analysis
            logger.info(f"\nMulti-Timeframe Analysis for {symbol}:")
            for timeframe, data in trend_data.items():
                logger.info(f"{timeframe}: {data['trend'].upper()} (Strength: {data['strength']:.2f}%)")

            # Calculate S/R levels with validation
            support_levels, resistance_levels = await self.calculate_support_resistance(df, symbol)
            
            # Validate S/R levels aren't identical
            if support_levels and resistance_levels:
                if any(abs(s - r) / s < 0.001 for s in support_levels for r in resistance_levels):
                    logger.warning(f"Identical S/R levels detected for {symbol}, recalculating...")
                    support_levels = [level for level in support_levels 
                                    if not any(abs(level - r) / level < 0.001 for r in resistance_levels)]
            
            # Get current price with retry logic
            current_price = await self.get_current_price(symbol)
            if not current_price:
                logger.warning(f"Could not get current price for {symbol}")
                return
            
            # Calculate new high-confidence grid levels with S/R
            new_grid_prices = await self.calculate_hybrid_grids(symbol, df, support_levels, resistance_levels)
            
            if not new_grid_prices or not current_price:
                return
            
            # Get existing orders
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            # Validate grid levels against trend
            validated_levels = self.validate_grid_levels(
                [{
                    'price': price,
                    'confidence': getattr(price, 'confidence', 0.0)
                } for price in new_grid_prices],
                trend_data
            )
            
            # Sort validated levels by confidence
            confidence_levels = [
                (level['price'], level['confidence'])
                for level in validated_levels
            ]
            
            # Sort by confidence (highest first)
            confidence_levels.sort(key=lambda x: x[1], reverse=True)
            highest_confidence = confidence_levels[0][1] if confidence_levels else 0
            
            # Check for exceptional opportunities (confidence > 2.0)
            if highest_confidence >= 2.0:
                logger.error(f"\n{'='*50}")
                logger.error(f"EXCEPTIONAL OPPORTUNITY DETECTED - {symbol}")
                logger.error(f"Current Price: {current_price}")
                logger.error("\nConfidence Analysis:")
                for price, conf in confidence_levels:
                    if conf >= 2.0:
                        rating = "🔥 EXCEPTIONAL"
                    elif conf >= 1.5:
                        rating = "✨ HIGH"
                    else:
                        rating = "⚠️ LOW"
                    logger.error(f"Level: {price:.4f}, Confidence: {conf:.2f} - {rating}")
                
                # Cancel ALL existing orders for high confidence opportunities
                if open_orders:
                    logger.error(f"\nCancelling all orders to focus on exceptional opportunity")
                    for order in open_orders:
                        try:
                            await self.exchange.cancel_order(order['id'], symbol)
                            logger.error(f"Cancelled order at {order['price']}")
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.error(f"Error cancelling order: {e}")
                
                # Clear grid tracking for this symbol
                self.grid_orders = {k: v for k, v in self.grid_orders.items() if v['symbol'] != symbol}
                
                logger.error(f"\nRepositioning grids for exceptional opportunity:")
                for price, conf in confidence_levels:
                    if conf >= 2.0:
                        logger.error(f"Target: {price:.4f} (Confidence: {conf:.2f})")
                logger.error(f"{'='*50}\n")
                
                # Force create new grids
                await self.execute_grid_orders(symbol, force_create=True)
                return
                
            # Regular high confidence check (1.5-2.0)
            elif highest_confidence >= 1.5:
                logger.error(f"\n{'='*50}")
                logger.error(f"HIGH CONFIDENCE OPPORTUNITY - {symbol}")
                logger.error(f"Current Price: {current_price}")
                logger.error("\nConfidence Analysis:")
                for price, conf in confidence_levels:
                    if conf >= 1.5:
                        rating = "✨ HIGH"
                    else:
                        rating = "⚠️ LOW"
                    logger.error(f"Level: {price:.4f}, Confidence: {conf:.2f} - {rating}")
                
                # Log trend alignment
                logger.error("\nTrend Alignment:")
                aligned = all(data['trend'] == trend_data['1d']['trend'] for data in trend_data.values())
                logger.error(f"All Timeframes Aligned: {'✅' if aligned else '❌'}")
                logger.error(f"Daily Trend: {trend_data['1d']['trend'].upper()}")
                logger.error(f"{'='*50}\n")
                
        except Exception as e:
            logger.error(f"Error reassessing grids for {symbol}: {e}")

    def calculate_total_fees(self, price: float, size: float, is_maker: bool = True) -> float:
        """Calculate total fees including slippage and volatility buffer"""
        try:
            # Base fee calculation
            fee_rate = self.maker_fee if is_maker else self.taker_fee
            base_fee = price * size * fee_rate
            
            # Add slippage and volatility buffers
            total_buffer = self.slippage_buffer + self.volatility_buffer
            buffer_cost = price * size * total_buffer
            
            total_cost = base_fee + buffer_cost
            
            logger.info(f"Fee Calculation:")
            logger.info(f"Base Fee: ${base_fee:.4f} ({fee_rate*100}%)")
            logger.info(f"Buffer Cost: ${buffer_cost:.4f} ({total_buffer*100}%)")
            logger.info(f"Total Cost: ${total_cost:.4f}")
            
            return total_cost
        except Exception as e:
            logger.error(f"Error calculating fees: {e}")
            return 0.0

    def adjust_grid_spacing_for_fees(self, current_price: float) -> None:
        """Adjust grid spacing to account for fees and buffers"""
        try:
            # Calculate minimum profitable spacing
            total_cost_ratio = (self.maker_fee * 2) + (self.slippage_buffer + self.volatility_buffer)
            min_spacing = total_cost_ratio * 3  # Minimum 3x cost coverage
            
            # Adjust grid spacing if needed
            self.grid_spacing = max(self.grid_spacing, min_spacing)
            logger.info(f"Adjusted grid spacing to {self.grid_spacing*100:.2f}% to account for fees")
            
        except Exception as e:
            logger.error(f"Error adjusting grid spacing: {e}")

    def get_sr_boost(self, price: float, support_levels: List[float], resistance_levels: List[float]) -> float:
        """Calculate confidence boost based on S/R proximity"""
        boost = 0.0
        
        # Support level analysis
        for support in support_levels:
            prox_pct = abs(price - support) / support
            if prox_pct <= 0.005:  # Within 0.5%
                boost += 1.0
            elif prox_pct <= 0.01:  # Within 1%
                boost += 0.7
            elif prox_pct <= 0.02:  # Within 2%
                boost += 0.4
        
        # Resistance level analysis
        for resistance in resistance_levels:
            prox_pct = abs(price - resistance) / resistance
            if prox_pct <= 0.005:  # Within 0.5%
                boost += 1.0
            elif prox_pct <= 0.01:  # Within 1%
                boost += 0.7
            elif prox_pct <= 0.02:  # Within 2%
                boost += 0.4
        
        # Multiple level confluence bonus
        if len(support_levels) > 1 or len(resistance_levels) > 1:
            boost *= 1.2  # 20% bonus for multiple levels
        
        return boost

    def calculate_safe_leverage(self, account_size: float) -> int:
        """Calculate safe leverage based on account size"""
        if account_size <= 100:
            leverage = 5  # Very conservative for small accounts
        elif account_size <= 250:
            leverage = 10
        elif account_size <= 500:
            leverage = 20
        elif account_size <= 1000:
            leverage = 30
        elif account_size <= 2500:
            leverage = 40
        else:
            leverage = 50  # Maximum leverage
            
        logger.info(f"\nLEVERAGE CALCULATION:")
        logger.info(f"Account Size: ${account_size:.2f}")
        logger.info(f"Selected Leverage: {leverage}x")
        logger.info(f"Max Position Value: ${(account_size * leverage):.2f}")
        
        return leverage

    async def calculate_grid_profit(self, price: float, grid_spacing: float, position_size: float) -> float:
        """Calculate expected profit for a grid trade"""
        price_move = price * grid_spacing
        expected_profit = price_move * position_size
        
        # Add safety margin for slippage
        safety_margin = 1.1  # 10% safety margin
        return expected_profit * safety_margin

    async def get_tradeable_instruments(self) -> Dict[str, List[str]]:
        """Fetch and categorize instruments from Kraken Spot"""
        try:
            # Use CCXT to fetch markets and tickers
            markets = await self.exchange.fetch_markets()
            tickers = await self.exchange.fetch_tickers()
            
            logger.info(f"Fetched {len(markets)} markets and {len(tickers)} tickers")
            
            categorized_symbols = {
                'tier1': [],  # $100-500
                'tier2': [],  # $500-2000
                'tier3': [],  # $2000-5000
                'tier4': []   # $5000+
            }
            
            # Process each market
            for market in markets:
                try:
                    symbol = market['symbol']
                    # Only include USD pairs and spot markets
                    if not symbol.endswith('/USD') or market.get('type') != 'spot':
                        continue
                        
                    # Get current price from tickers
                    if symbol in tickers and tickers[symbol].get('last'):
                        price = float(tickers[symbol]['last'])
                        
                        if price > 0:
                            if price < 500:
                                categorized_symbols['tier1'].append(symbol)
                            elif price < 2000:
                                categorized_symbols['tier2'].append(symbol)
                            elif price < 5000:
                                categorized_symbols['tier3'].append(symbol)
                            else:
                                categorized_symbols['tier4'].append(symbol)
                            logger.info(f"Added {symbol} at ${price:.2f}")
                            
                except Exception as e:
                    logger.error(f"Error processing market {market.get('symbol', 'unknown')}: {e}")
                    continue
            
            # Log the categorization
            logger.info("\nTradeable Spot Instruments by Tier:")
            for tier, symbols in categorized_symbols.items():
                logger.info(f"\n{tier.upper()} (Count: {len(symbols)}):")
                for symbol in sorted(symbols):
                    if symbol in tickers and tickers[symbol].get('last'):
                        price = float(tickers[symbol]['last'])
                        logger.info(f"- {symbol} (${price:.2f})")
            
            return categorized_symbols
            
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            logger.exception("Full traceback:")
            
            # Return empty lists if something went wrong
            return {
                'tier1': [],
                'tier2': [],
                'tier3': [],
                'tier4': []
            }

    async def analyze_multi_timeframe_trend(self, symbol: str) -> dict:
        timeframes = {
            '15m': {'fast_ema': 20, 'slow_ema': 50},
            '1h': {'fast_ema': 50, 'slow_ema': 200},
            '1d': {'fast_ema': 50, 'slow_ema': 200}
        }
        
        trend_results = {}
        
        for tf, emas in timeframes.items():
            try:
                # Add timeout and retry logic for data fetch
                for attempt in range(3):  # 3 retries
                    try:
                        df = await asyncio.wait_for(
                            self.get_historical_data(symbol, timeframe=tf, limit=210),
                            timeout=30  # 30 second timeout
                        )
                        if df is not None:
                            break
                        await asyncio.sleep(1)
                    except asyncio.TimeoutError:
                        if attempt == 2:  # Last attempt
                            logger.error(f"Timeout getting {tf} data for {symbol}")
                            continue
                        await asyncio.sleep(2)
                        
                if df is not None:
                    # Calculate EMAs
                    df[f'ema_{emas["fast_ema"]}'] = df['close'].ewm(span=emas['fast_ema'], adjust=False).mean()
                    df[f'ema_{emas["slow_ema"]}'] = df['close'].ewm(span=emas['slow_ema'], adjust=False).mean()
                    
                    # Add RSI
                    df['rsi'] = ta.momentum.RSIIndicator(df['close'], self.rsi_period).rsi()
                    
                    # Add MACD
                    macd = ta.trend.MACD(
                        df['close'], 
                        window_slow=self.macd_slow, 
                        window_fast=self.macd_fast, 
                        window_sign=self.macd_signal
                    )
                    df['macd'] = macd.macd()
                    df['macd_signal'] = macd.macd_signal()
                    df['macd_histogram'] = macd.macd_diff()
                    
                    # Get S/R levels with Fibonacci
                    support_levels, resistance_levels = await self.calculate_support_resistance(df, symbol)
                    
                    # Get last values
                    fast_ema = df[f'ema_{emas["fast_ema"]}'].iloc[-1]
                    slow_ema = df[f'ema_{emas["slow_ema"]}'].iloc[-1]
                    current_price = df['close'].iloc[-1]
                    
                    # Determine trend
                    trend_results[tf] = {
                        'trend': 'bullish' if fast_ema > slow_ema else 'bearish',
                        'strength': abs(fast_ema - slow_ema) / slow_ema * 100,
                        'rsi': df['rsi'].iloc[-1],
                        'macd_hist': df['macd_histogram'].iloc[-1],
                        'is_oversold': df['rsi'].iloc[-1] < 30,
                        'is_overbought': df['rsi'].iloc[-1] > 70,
                        'support_levels': support_levels,
                        'resistance_levels': resistance_levels,
                        'near_support': any(abs(current_price - s) / s < 0.01 for s in support_levels),
                        'near_resistance': any(abs(current_price - r) / r < 0.01 for r in resistance_levels)
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing {tf} timeframe for {symbol}: {e}")
                continue
        
        return trend_results

    def validate_grid_levels(self, levels: List[dict], trend_data: dict) -> List[dict]:
        """Validate grid levels against trend and Fibonacci"""
        validated_levels = []
        
        # Get overall trend bias
        bullish_timeframes = sum(1 for tf in trend_data.values() if tf['trend'] == 'bullish')
        bearish_timeframes = len(trend_data) - bullish_timeframes
        trend_bias = 'bullish' if bullish_timeframes > bearish_timeframes else 'bearish'
        
        # Fibonacci levels (you can adjust these)
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in levels:
            price = level['price']
            
            # Calculate Fibonacci zones
            is_fib_zone = False
            for fib in fib_levels:
                fib_price = price * (1 + fib * 0.1)  # 10% range for each fib level
                if abs(price - fib_price) / price < 0.02:  # 2% tolerance
                    is_fib_zone = True
                    break
            
            # Validate level based on trend and Fibonacci
            if trend_bias == 'bullish':
                if price <= level['price'] and is_fib_zone:
                    validated_levels.append(level)
            else:
                if price >= level['price'] and is_fib_zone:
                    validated_levels.append(level)
        
        return validated_levels

    async def get_asset_volatility(self, symbol: str, is_spot: bool = False) -> float:
        """Enhanced volatility calculation with protective downside focus for spot"""
        try:
            # Get recent OHLCV data (using existing 15m timeframe for consistency)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=24)
            
            # Calculate both ATR and returns volatility
            ranges = []
            returns = []
            downside_returns = []
            
            for i in range(1, len(ohlcv)):
                # ATR calculation
                high, low = ohlcv[i][2], ohlcv[i][3]
                ranges.append(high - low)
                
                # Returns calculation
                close_price = ohlcv[i][4]
                prev_close = ohlcv[i-1][4]
                ret = (close_price - prev_close) / prev_close
                returns.append(ret)
                
                # Track downside returns for spot
                if is_spot and ret < 0:
                    downside_returns.append(abs(ret))  # Use absolute value of negative returns
            
            # ATR-based volatility
            atr_volatility = (sum(ranges) / len(ranges) / ohlcv[-1][4]) * 100
            returns_vol = np.std(returns) * np.sqrt(24) * 100  # Adjusted to daily
            
            if is_spot and downside_returns:
                # Higher downside volatility = tighter stops to protect profits
                downside_vol = np.std(downside_returns) * np.sqrt(24) * 100
                # Inverse relationship: higher downside vol = lower final volatility = tighter stops
                volatility = atr_volatility / (1 + downside_vol)
            else:
                # Regular volatility calculation for non-spot
                returns_vol = np.std(returns) * np.sqrt(24) * 100
                volatility = (atr_volatility * 0.6) + (returns_vol * 0.4)
            
            logger.info(f"Volatility metrics for {symbol} ({'Spot' if is_spot else 'Margin'}):")
            logger.info(f"ATR-based: {atr_volatility:.2f}%")
            if is_spot:
                logger.info(f"Downside Vol: {downside_vol:.2f}%")
                logger.info(f"Final (Tightened): {volatility:.2f}%")
            else:
                logger.info(f"Returns Vol: {returns_vol:.2f}%")
                logger.info(f"Final: {volatility:.2f}%")
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02  # Default 2% volatility

    async def check_correlation_risk(self, symbol: str) -> bool:
        try:
            active_positions = [pos for pos in self.positions.values() if pos['size'] != 0]
            if not active_positions:
                return True
                
            # Get price history for new symbol
            new_prices = await self.get_recent_prices(symbol)
            if not new_prices:
                return True
                
            for pos in active_positions:
                pos_symbol = pos['symbol']
                pos_prices = await self.get_recent_prices(pos_symbol)
                if not pos_prices:
                    continue
                    
                correlation = np.corrcoef(new_prices, pos_prices)[0,1]
                if abs(correlation) > 0.7:  # High correlation threshold
                    logger.warning(f"High correlation ({correlation:.2f}) between {symbol} and {pos_symbol}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error in correlation check: {e}")
            return True

    def get_symbol_prefix(self):
        """Get the appropriate symbol prefix based on mode"""
        return 'PF_' if self.demo_mode else ''

    def normalize_confidence_scores(self, levels: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Normalize confidence scores to a 0-10 scale"""
        if not levels:
            return []
        
        # Get min/max confidence
        confidences = [conf for _, conf in levels]
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        # Normalize to 0-10 scale
        if max_conf > min_conf:
            normalized = [(price, (conf - min_conf) / (max_conf - min_conf) * 10)
                         for price, conf in levels]
        else:
            normalized = [(price, 5.0) for price, conf in levels]  # Default to middle confidence
            
        return normalized

    async def start_market_analysis(self, symbol: str) -> Tuple[bool, float]:
        """Start market analysis process"""
        try:
            # Get historical data first
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                logger.warning(f"Could not get historical data for {symbol}")
                return False, 0.0
                
            # Analyze market conditions using the data
            should_trade, volatility = await self.analyze_market_conditions(df, symbol)
            return should_trade, volatility
            
        except Exception as e:
            logger.error(f"Error in start_market_analysis: {e}")
            return False, 0.0

    async def filter_symbols(self, symbols: List[str]) -> List[str]:
        """Pre-filter symbols before detailed analysis - rescans hourly"""
        filtered = []
        current_time = time.time()
        
        # Check if an hour has passed since last scan
        if hasattr(self, 'last_scan_time') and current_time - self.last_scan_time < 3600:
            logger.info("Using existing filtered symbols (next scan in {:.1f} minutes)".format(
                (3600 - (current_time - self.last_scan_time)) / 60
            ))
            return self.last_filtered_symbols if hasattr(self, 'last_filtered_symbols') else []
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 HOURLY SYMBOL SCAN - PRE-FILTERING {len(symbols)} SYMBOLS")
        logger.info(f"{'='*50}")
        
        # Get available balance first
        available_balance = await self.get_available_balance()  # Changed from get_account_balance
        logger.info(f"Available Balance: ${available_balance:.2f}")
        
        # Calculate max position size (e.g., 20% of account)
        max_position_size = available_balance * 0.20
        logger.info(f"Max Position Size: ${max_position_size:.2f}")
        
        # Restricted pairs for US/TX
        restricted_pairs = [
            'EUR', 'GBP', 'CHF', 'AUD', 'CAD', 'JPY',  # Fiat pairs
            'USDC', 'USDT', 'DAI', 'BUSD'  # Stablecoins
        ]
        
        try:
            # Get all tickers at once for efficiency
            tickers = await self.exchange.fetch_tickers()
            
            # Filter and sort by volume, considering account size
            tradeable_symbols = []
            for symbol in symbols:
                try:
                    if (symbol in tickers and 
                        tickers[symbol]['quoteVolume'] is not None and 
                        not any(restricted in symbol for restricted in restricted_pairs)):
                        
                        ticker = tickers[symbol]
                        price = ticker['last']
                        
                        # Calculate minimum position size (e.g., 3 units)
                        min_position_cost = price * 3  # Minimum 3 units
                        
                        # Check if we can trade at least 3 units and price is reasonable
                        if (min_position_cost <= max_position_size and 
                            ticker['quoteVolume'] > 1000000 and  # Min 1M daily volume
                            'USD' in symbol):
                            
                            tradeable_symbols.append({
                                'symbol': symbol,
                                'price': price,
                                'volume': ticker['quoteVolume'],
                                'min_cost': min_position_cost
                            })
                            
                except Exception as e:
                    continue
            
            # Sort by volume and filter top symbols
            volume_sorted = sorted(
                tradeable_symbols,
                key=lambda x: x['volume'],
                reverse=True
            )[:20]  # Take top 20
            
            # Log detailed analysis
            logger.info("\nTradeable Symbols Analysis:")
            logger.info(f"{'Symbol':<12} {'Price':<10} {'Volume(M)':<12} {'Min Cost':<10}")
            logger.info("-" * 44)
            
            for item in volume_sorted:
                filtered.append(item['symbol'])
                logger.info(
                    f"{item['symbol']:<12} "
                    f"${item['price']:<9.2f} "
                    f"${item['volume']/1000000:<11.1f}M "
                    f"${item['min_cost']:<9.2f}"
                )
            
            # Store scan time and filtered symbols
            self.last_scan_time = current_time
            self.last_filtered_symbols = filtered
            
            logger.info(f"\n✨ Selected {len(filtered)} symbols within account trading range")
            logger.info(f"Account size: ${available_balance:.2f}")
            logger.info(f"Max position: ${max_position_size:.2f}")
            logger.info("Next scan in 60 minutes")
            return filtered
            
        except Exception as e:
            logger.error(f"Error in filter_symbols: {e}")
            # Return major pairs that are likely within our range
            return ['XRP/USD', 'ADA/USD', 'DOGE/USD', 'DOT/USD']  # Lower-priced majors as fallback

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for Kraken"""
        try:
            # Remove any suffixes like :USD or :ETH
            base_symbol = symbol.split(':')[0]
            
            # Map of common symbol corrections
            symbol_map = {
                'BTC/USD': 'BTC/USD',  # Kraken uses XBT instead of BTC
                'WBTC/USD': 'WBTC/USD',
                'ETH/USD': 'ETH/USD',
                'SOL/USD': 'SOL/USD',
                'XRP/USD': 'XRP/USD',
                'ADA/USD': 'ADA/USD',
                'DOGE/USD': 'DOGE/USD',
                'DOT/USD': 'DOT/USD',
                'LTC/USD': 'LTC/USD'
            }
            
            normalized = symbol_map.get(base_symbol, base_symbol)
            logger.info(f"Normalized {symbol} to {normalized}")
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing symbol {symbol}: {e}")
            return symbol  # Return original if normalization fails

    async def execute_stop_loss(self, symbol: str, position_size: float, is_spot: bool, reason: str = "stop loss") -> bool:
        """Execute stop loss order with retries and confirmation"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Verify spot balance first
                balance = await self.exchange.fetch_balance()
                asset = symbol.split('/')[0]
                spot_balance = float(balance.get(asset, {}).get('free', 0))
                
                if spot_balance <= 0:
                    logger.error(f"No spot balance found for {asset}, cleaning up tracking")
                    if symbol in self.positions:
                        del self.positions[symbol]
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                    return True  # Return True since position is already closed
                
                logger.error(f"\n{'='*50}")
                logger.error(f"🚨 EXECUTING {reason.upper()} - Attempt {attempt + 1}/{max_retries}")
                logger.error(f"Symbol: {symbol}")
                logger.error(f"Actual Balance: {spot_balance} {asset}")
                
                # Use actual spot balance for order
                params = {
                    'ordertype': 'market',
                    'trading_agreement': 'agree'
                }
                
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side='sell',
                    amount=spot_balance,
                    params=params
                )
                
                if order and 'id' in order:
                    logger.error(f"✅ {reason.upper()} executed successfully")
                    logger.error(f"Order ID: {order['id']}")
                    logger.error(f"Size: {spot_balance}")
                    logger.error(f"{'='*50}\n")
                    
                    # Clean up position tracking
                    if symbol in self.positions:
                        del self.positions[symbol]
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                    return True
                
                logger.error(f"❌ No order ID returned - Attempt {attempt + 1}")
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error executing {reason}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                continue
        
        logger.error(f"❌ Failed to execute {reason} after {max_retries} attempts")
        return False

    async def validate_order_size(self, symbol: str, amount: float) -> bool:
        """Always validate order size as true to allow any size orders"""
        try:
            logger.info(f"Allowing order for {symbol} with size {amount}")
            return True
            
        except Exception as e:
            logger.error(f"Error in order size validation: {e}")
            return True  # Still return True on error to allow order

    async def validate_grid_order_size(self, symbol: str, size: float) -> float:
        """Validate if grid order size meets exchange minimums"""
        try:
            # Get market info from exchange
            market = self.exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            
            if size < min_amount:
                logger.warning(f"Grid order size {size} is below minimum {min_amount} for {symbol}")
                # Calculate the adjusted size to meet minimum
                adjusted_size = min_amount * 1.01  # Add 1% buffer
                logger.info(f"Adjusting grid order size to {adjusted_size}")
                return adjusted_size
                
            return size
            
        except Exception as e:
            logger.error(f"Error validating grid order size: {e}")
            return size  # Return original size if validation fails

    async def verify_actual_positions(self) -> List[Dict]:
        """Verify actual positions vs open orders"""
        try:
            # Get actual balances
            balance = await self.exchange.fetch_balance()
            
            # Track actual positions
            actual_positions = []
            
            for currency, amount in balance.get('free', {}).items():
                if currency in ['USD', 'USDT']:
                    continue
                
                total_amount = float(amount)
                if total_amount > 0.00001:  # Minimum threshold
                    symbol = f"{currency}/USD"
                    
                    # Get actual entry price from order history
                    orders = await self.exchange.fetch_open_orders(symbol)
                    filled_buys = [o for o in orders if o['side'] == 'buy' and o['status'] == 'open']
                    
                    if filled_buys:
                        entry_price = float(filled_buys[0]['price'])
                        logger.info(f"Found actual position for {symbol}:")
                        logger.info(f"Balance: {total_amount} {currency}")
                        logger.info(f"Entry: ${entry_price}")
                        
                        position = {
                            'symbol': symbol,
                            'info': {
                                'symbol': symbol,
                                'size': total_amount,
                                'price': entry_price,
                                'side': 'long',
                                'is_spot': True
                            }
                        }
                        actual_positions.append(position)
            
            return actual_positions
                
        except Exception as e:
            logger.error(f"Error verifying actual positions: {e}")
            return []  # Return empty list on error

    async def validate_entry_conditions(self, symbol: str, df: pd.DataFrame) -> bool:
        """Enhanced multi-timeframe entry validation"""
        try:
            # Get multi-timeframe data
            daily_df = await self.get_historical_data(symbol, timeframe='1d')
            four_hour_df = await self.get_historical_data(symbol, timeframe='4h')
            hourly_df = await self.get_historical_data(symbol, timeframe='1h')
            five_min_df = df  # Current 5m dataframe

            # Add sentiment analysis
            sentiment_data = await self.market_analyzer.analyze_market_sentiment(symbol, '1d')
            if sentiment_data and sentiment_data.get('confidence', 0) > 0.8:
                sentiment_score = sentiment_data['sentiment']['positive']
                logger.info(f"Sentiment Score: {sentiment_score:.2f}")
                
                if sentiment_score < 0.6:
                    logger.info("❌ Market sentiment not bullish enough")
                    return False

            # Continue with existing validation logic...
            current_price = float(df['close'].iloc[-1])
            volume = float(df['volume'].iloc[-1])
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            
            logger.info(f"\n{'='*50}")
            logger.info(f"VALIDATING ENTRY - {symbol}")
            logger.info(f"Price: ${current_price:.4f}")
            
            # Calculate base technical indicators
            rsi = ta.RSI(df['close'], timeperiod=14).iloc[-1]
            bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20)
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            
            # Daily timeframe conditions
            daily_ema9 = ta.EMA(daily_df['close'], timeperiod=9)
            daily_ema21 = ta.EMA(daily_df['close'], timeperiod=21)
            
            # Define daily EMA cross
            daily_ema_cross = (
                daily_ema9.iloc[-1] > daily_ema21.iloc[-1] and
                daily_ema9.iloc[-2] <= daily_ema21.iloc[-2]
            )

            # Then use the condition
            if not daily_ema_cross:
                logger.info("❌ No daily 9/21 EMA bullish cross")
                return False
            
            daily_rsi = ta.RSI(daily_df['close'], timeperiod=14).iloc[-1]
            daily_rsi_prev = ta.RSI(daily_df['close'], timeperiod=14).iloc[-2]
            daily_macd = ta.MACD(daily_df['close'])
            daily_macd_cross = (
                daily_macd.macd().iloc[-1] > daily_macd.macd_signal().iloc[-1] and
                daily_macd.macd().iloc[-2] <= daily_macd.macd_signal().iloc[-2]
            )
            
            # Check if near daily fib support
            support_levels, resistance_levels = await self.calculate_support_resistance(daily_df, symbol)
            near_support = any(abs(current_price - s) / s < 0.02 for s in support_levels)
            
            logger.info("\nDaily Timeframe:")
            logger.info(f"EMA 9/21 Cross: {'✅' if daily_ema_cross else '❌'}")
            logger.info(f"RSI: {daily_rsi:.1f} (Previous: {daily_rsi_prev:.1f})")
            logger.info(f"MACD Bullish Cross: {'✅' if daily_macd_cross else '❌'}")
            logger.info(f"Near Fib Support: {'✅' if near_support else '❌'}")
            
            # Add this as a primary condition
            if not daily_ema_cross:
                logger.info("❌ No daily 9/21 EMA bullish cross")
                return False
            
            # Small cap specific checks
            if current_price < 5.0:  # Small cap threshold
                # 1. Check daily conditions first
                if not (daily_rsi > 30 and daily_rsi_prev <= 30):
                    logger.info("❌ Daily RSI not showing bullish cross above 30")
                    return False
                    
                if not daily_macd_cross:
                    logger.info("❌ Daily MACD not showing bullish cross")
                    return False
                    
                if not near_support:
                    logger.info("❌ Price not near daily fib support")
                    return False
                
                # 2. Check 4H and 1H RSI
                four_hour_rsi = ta.RSI(four_hour_df['close'], timeperiod=14).iloc[-1]
                hourly_rsi = ta.RSI(hourly_df['close'], timeperiod=14).iloc[-1]
                
                if four_hour_rsi < 50 or hourly_rsi < 50:
                    logger.info("❌ 4H or 1H RSI below 50")
                    return False
                
                # 3. Original small cap checks
                if rsi > 35:
                    logger.info(f"❌ RSI too high for small cap: {rsi:.1f}")
                    return False
                
                min_volume = volume_ma * 1.5
                if volume < min_volume:
                    logger.info(f"❌ Volume too low: {volume:,.0f} < {min_volume:,.0f}")
                    return False
                
                if bb_width > 0.4:
                    logger.info(f"❌ BB Width too high: {bb_width:.3f}")
                    return False
                
                if (atr/current_price) > 0.03:
                    logger.info(f"❌ ATR too high: {(atr/current_price)*100:.1f}%")
                    return False
                
                # 4. Check 5min EMAs and MACD
                ema9 = ta.EMA(five_min_df['close'], timeperiod=9).iloc[-1]
                ema21 = ta.EMA(five_min_df['close'], timeperiod=21).iloc[-1]
                ema200 = ta.EMA(five_min_df['close'], timeperiod=200).iloc[-1]
                
                if not (ema9 > ema21 > ema200):
                    logger.info("❌ EMAs not in bullish alignment")
                    return False
                
                five_min_macd = ta.MACD(five_min_df['close'])
                if five_min_macd.macd().iloc[-1] < five_min_macd.macd_signal().iloc[-1]:
                    logger.info("❌ 5min MACD not showing upward momentum")
                    return False
                
                logger.info("\n✅ Small cap entry conditions met:")
                logger.info(f"- Daily RSI cross and MACD bullish")
                logger.info(f"- 4H and 1H RSI above 50")
                logger.info(f"- 5min EMAs and MACD bullish")
                logger.info(f"- Strong volume and controlled volatility")
                
            else:  # Regular entry conditions for higher priced assets
                # Keep existing conditions and add multi-timeframe checks
                if not (daily_rsi > 30 and daily_rsi_prev <= 30):
                    logger.info("❌ Daily RSI not showing bullish cross above 30")
                    return False
                
                if not daily_macd_cross:
                    logger.info("❌ Daily MACD not showing bullish cross")
                    return False
                
                if rsi > 40:
                    logger.info(f"❌ RSI too high: {rsi:.1f}")
                    return False
                
                if volume < volume_ma:
                    logger.info(f"❌ Volume below average: {volume:,.0f} < {volume_ma:,.0f}")
                    return False
                
                # Technical Structure
                ema9 = ta.EMA(df['close'], timeperiod=9).iloc[-1]
                ema21 = ta.EMA(df['close'], timeperiod=21).iloc[-1]
                ema200 = ta.EMA(df['close'], timeperiod=200).iloc[-1]
                
                # Check EMA alignment for trend
                if current_price < ema200:
                    logger.info("❌ Price below 200 EMA - Not in uptrend")
                    return False
                
                if ema9 < ema21:
                    logger.info("❌ EMAs in bearish alignment (9 < 21)")
                    return False
                
                # Keep existing volatility and MACD checks
                if bb_width > 0.5:
                    logger.info(f"❌ BB Width too high: {bb_width:.3f}")
                    return False
                
                if (atr/current_price) > 0.04:
                    logger.info(f"❌ ATR too high: {(atr/current_price)*100:.1f}%")
                    return False
                
                # 5. MACD Momentum
                macd = ta.MACD(df['close'])
                if macd.macd().iloc[-1] < macd.macd_signal().iloc[-1]:
                    logger.info("❌ MACD below signal line - Weak momentum")
                    return False
                
                logger.info("\n✅ Regular asset entry conditions met:")
                logger.info(f"- Daily timeframe confirmation")
                logger.info(f"- RSI and volume criteria met")
                logger.info(f"- EMAs in bullish alignment")
                logger.info(f"- Controlled volatility")
                logger.info(f"- Strong momentum")
            
            # Final resistance check for both types
            try:
                if resistance_levels:
                    nearest_resistance = min(r for r in resistance_levels if r > current_price)
                    distance_to_resistance = (nearest_resistance - current_price) / current_price
                    
                    # If price is within 3% of resistance, reject entry
                    if distance_to_resistance < 0.03:
                        logger.info(f"❌ Too close to resistance: {distance_to_resistance*100:.1f}% away")
                        return False
                        
                    logger.info(f"✅ Clear distance to resistance: {distance_to_resistance*100:.1f}%")
                else:
                    logger.info("✅ No immediate resistance above")
                    
            except Exception as e:
                logger.error(f"Error checking resistance: {e}")
                return False
            
            logger.info(f"{'='*50}\n")
            return True
            
        except Exception as e:
            logger.error(f"Error validating entry conditions: {e}")
            return False

    async def check_all_positions(self) -> None:
        """Debug helper to check all positions and balances"""
        try:
            balance = await self.exchange.fetch_balance()
            
            logger.info("\n=== CHECKING ALL SPOT POSITIONS ===")
            for currency, amount in balance.items():
                total = float(amount.get('total', 0))
                free = float(amount.get('free', 0))
                used = float(amount.get('used', 0))
                
                if total > 0:
                    logger.info(f"\nFound {currency}:")
                    logger.info(f"Total: {total}")
                    logger.info(f"Free: {free}")
                    logger.info(f"Used: {used}")
                    
                    # Try to get recent orders for this asset
                    try:
                        symbol = f"{currency}/USD"
                        orders = await self.exchange.fetch_open_orders(symbol, limit=3)
                        if orders:
                            logger.info(f"Recent orders for {symbol}:")
                            for order in orders:
                                logger.info(f"Order: {order['side']} {order['amount']} @ {order['price']}")
                    except Exception as e:
                        logger.info(f"Could not fetch orders for {currency}: {e}")
                        
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")

    async def quick_profit_check(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Fast profit check and TP execution without other overhead"""
        try:
            entry_price = float(position['info']['price'])
            position_side = position['info'].get('side', 'long')
            position_size = float(position['info'].get('size', 0))
            is_spot = True  # For now we're forcing spot
            
            # Quick profit calc
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Check TP1 (5%)
            if profit_pct >= 5.0:
                tp_size = position_size * 0.3
                await self.execute_take_profit(symbol, position_side, tp_size, current_price, "TP1", is_spot)
                return True
                
            # Check TP2 (10%)
            if profit_pct >= 10.0:
                tp_size = position_size * 0.3
                await self.execute_take_profit(symbol, position_side, tp_size, current_price, "TP2", is_spot)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in quick profit check: {e}")
            return False

    async def get_available_balance(self) -> float:
        """Get actual available balance accounting for open orders"""
        try:
            balance = await self.get_account_balance()
            
            # Get all open orders to calculate committed funds
            open_orders = await self.exchange.fetch_open_orders()
            committed_funds = sum(
                float(order['price']) * float(order['amount'])
                for order in open_orders
                if order['side'] == 'buy'  # Only count buy orders
            )
            
            actual_available = balance - committed_funds
            logger.info(f"\nBalance Analysis:")
            logger.info(f"Total Balance: ${balance:.2f}")
            logger.info(f"Committed in Orders: ${committed_funds:.2f}")
            logger.info(f"Actually Available: ${actual_available:.2f}")
            
            # Update sufficient funds flag
            self.has_sufficient_funds = actual_available > 20  # Minimum $20 available
            
            return actual_available
            
        except Exception as e:
            logger.error(f"Error calculating available balance: {e}")
            return 0.0

    async def cancel_stale_orders(self, symbol: str, max_age_hours: int = None) -> None:
        """Cancel orders that have been open for longer than the specified time period"""
        try:
            # Use the class parameter if no specific timeout is provided
            if max_age_hours is None:
                max_age_hours = self.stale_order_timeout_hours

            # Fetch open orders for the symbol
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            if not open_orders:
                return
                
            current_time = time.time() * 1000  # Current time in milliseconds
            cancelled_count = 0
            
            for order in open_orders:
                # Calculate order age in hours
                order_time = order['timestamp']  # Order creation time in milliseconds
                order_age_hours = (current_time - order_time) / (1000 * 60 * 60)
                
                # Cancel orders older than max_age_hours
                if order_age_hours > max_age_hours:
                    order_id = order['id']
                    price = float(order['price'])
                    side = order['side']
                    
                    try:
                        await self.exchange.cancel_order(order_id, symbol)
                        cancelled_count += 1
                        logger.info(f"Cancelled stale {side} order for {symbol} at price ${price:.4f} (age: {order_age_hours:.1f} hours)")
                    except Exception as e:
                        logger.error(f"Failed to cancel stale order {order_id} for {symbol}: {e}")
            
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} stale orders for {symbol} older than {max_age_hours} hours")
        
        except Exception as e:
            logger.error(f"Error cancelling stale orders for {symbol}: {e}")

if __name__ == "__main__":
    try:
        # Set up proper event loop policy for Windows
        if platform.system() == 'Windows':
            import asyncio
            import nest_asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            nest_asyncio.apply()
        
        # Initialize strategy
        strategy = KrakenAdvancedGridStrategy(demo_mode=False)
        
        # Run with proper error handling
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(strategy.start())
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
        finally:
            loop.run_until_complete(strategy.exchange.close())
            loop.close()
            
    except Exception as e:
        logger.error(f"Fatal strategy error: {e}")
        logger.error(traceback.format_exc())
