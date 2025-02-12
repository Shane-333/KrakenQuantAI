import asyncio
import logging
import os
import platform
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import talib
from typing import Dict, List, Tuple, Union
from dotenv import load_dotenv
from datetime import datetime, timezone
import time
import traceback
import pytz
from market_analysis import MarketAnalyzer
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KrakenAdvancedGridStrategy:
    def __init__(self, demo_mode=False):  # Set default to False for live
        self.demo_mode = demo_mode
        self.STABLE_BLACKLIST = {
        'EUR/USD', 'USDC/USD', 'USDT/USD', 'AUD/USD', 'FARTCOIN/USD', 'PEPE/USD', 'BONK/USD', 'SHIB/USD', 'GBP/USD', 'JPY/USD', 'CHF/USD', 'NZD/USD', 'CAD/USD', 'HKD/USD', 'SGD/USD', 'NOK/USD', 'SEK/USD', 'AUD/JPY', 'NZD/JPY', 'CAD/JPY', 'HKD/JPY', 'SGD/JPY', 'NOK/JPY', 'SEK/JPY'
        'EUR/USD:USD', 'USDC/USD:USD', 'USDT/USD:USD', 'AUD/USD:USD', 'FARTCOIN/USD:USD', 'PEPE/USD:USD', 'BONK/USD:USD', 'SHIB/USD:USD', 'GBP/USD:USD', 'JPY/USD:USD', 'CHF/USD:USD', 'NZD/USD:USD', 'CAD/USD:USD', 'HKD/USD:USD', 'SGD/USD:USD', 'NOK/USD:USD', 'SEK/USD:USD', 'AUD/JPY:USD', 'NZD/JPY:USD', 'CAD/JPY:USD', 'HKD/JPY:USD', 'SGD/JPY:USD', 'NOK/JPY:USD', 'SEK/JPY:USD'
        }

        
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
        self.has_sufficient_funds = True  # Add this line to initialize the attribute

        # Add this line in your __init__
        self.atr_periods = 14  # Standard 14-period ATR
        self.is_spot = False  # Default to False

        # Add this line to initialize the attribute
        self.min_position_size = 0.01  # Default value

        # Initialize risk management parameters
        self.risk_per_trade = 0.05  # 10% 
        self.max_portfolio_risk = 0.15  # 10% default
        logger.info(f"Strategy initialized with risk_per_trade={self.risk_per_trade*100}%")

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
        """Get open positions including both spot and margin/futures"""
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
                        # Get proper normalized symbol
                        symbol = self.normalize_symbol(f"{currency}/USD")
                        
                        # Get current price from order book spread
                        orderbook = await self.exchange.fetch_order_book(symbol)
                        bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
                        ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
                        mid_price = (bid + ask) / 2
                        
                        # Calculate entry price using spread-adjusted mid price
                        entry_price = await self.get_average_entry_price(symbol, total_amount, mid_price)
                        
                        position = {
                            'symbol': symbol,
                            'info': {
                                'symbol': symbol,
                                'size': total_amount,
                                'price': entry_price,  # Use spread-adjusted price
                                'side': 'long',
                                'is_spot': True
                            }
                        }
                        active_positions.append(position)
                    except Exception as e:
                        logger.error(f"Error processing {currency} position: {str(e)[:100]}")
                        continue
            return active_positions
            
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
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
            
            # Calculate required periods for all indicators
            required_periods = max(
                self.rsi_period,
                self.ema_long_period,
                self.bb_period,
                self.atr_period,
                self.macd_slow
            )
            
            # Add buffer and fetch more data than needed
            fetch_limit = max(limit, required_periods * 2)
            
            # Get OHLCV data using Kraken's specific endpoint
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                timeframe,
                limit=fetch_limit,
                params={"interval": interval}
            )
            
            if not ohlcv or len(ohlcv) < required_periods:
                logger.error(f"Insufficient data for {symbol}. Need {required_periods} periods, got {len(ohlcv) if ohlcv else 0}")
                return pd.DataFrame()
                
            logger.info(f"Received {len(ohlcv)} candles for {symbol}")
            
            # Create DataFrame and calculate indicators
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure minimum required data
            if len(df) < required_periods:
                logger.warning(f"Insufficient data ({len(df)} rows) for {symbol}. Need {required_periods} periods")
                return pd.DataFrame()

            # Wrap each indicator in try/except
            try:
                df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            except Exception as e:
                logger.error(f"RSI Error for {symbol}: {e}")
                df['rsi'] = np.nan

            try:
                df['ema_short'] = ta.trend.EMAIndicator(df['close'], self.ema_short_period).ema_indicator()
                df['ema_long'] = ta.trend.EMAIndicator(df['close'], self.ema_long_period).ema_indicator()
            except Exception as e:
                logger.error(f"EMA Error for {symbol}: {e}")
                df['ema_short'] = df['close']
                df['ema_long'] = df['close']

            try:
                df['atr'] = ta.volatility.AverageTrueRange(
                    df['high'], 
                    df['low'], 
                    df['close'], 
                    window=self.atr_period
                ).average_true_range()
            except Exception as e:
                logger.error(f"ATR Error for {symbol}: {e}")
                df['atr'] = 0
            
            # Calculate MACD with error handling
            try:
                macd_line, signal_line, macd_hist = talib.MACD(
                    df['close'], 
                    fastperiod=self.macd_fast,  # Correct parameter name
                    slowperiod=self.macd_slow,   # Correct parameter name
                    signalperiod=self.macd_signal
                )
                df['macd'] = macd_line
                df['macd_signal'] = signal_line
                df['macd_histogram'] = macd_hist
            except Exception as e:
                logger.error(f"MACD Error for {symbol}: {e}")
                df['macd'] = 0
                df['macd_signal'] = 0
                df['macd_histogram'] = 0
            
            # Calculate Bollinger Bands with error handling
            try:
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
            except Exception as e:
                logger.error(f"Bollinger Bands Error for {symbol}: {e}")
                df['bb_upper'] = df['close']
                df['bb_middle'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_width'] = 0
            
            # Drop any NaN values and return a copy
            df_clean = df.dropna().copy()
            
            if len(df_clean) > 0:
                logger.info(f"Calculated indicators for {symbol}:")
                logger.info(f"Latest RSI: {df_clean['rsi'].iloc[-1]:.2f}")
                logger.info(f"Latest MACD: {df_clean['macd'].iloc[-1]:.4f}")
                logger.info(f"Latest BB Width: {df_clean['bb_width'].iloc[-1]:.4f}")
            
            return df_clean
                
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {str(e)[:200]}")
            return pd.DataFrame()

    async def analyze_market_conditions(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float]:
        """Analyze market conditions using technical indicators and FinBERT sentiment"""
        try:
            # Validate input DataFrame
            if df is None or len(df) < 1:
                logger.error(f"Invalid DataFrame for {symbol}")
                return False, 0.0

            # Add market state detection
            daily_df = await self.get_historical_data(symbol, '1d')
            if daily_df is not None and len(daily_df) >= 200:
                daily_ma_200 = daily_df['close'].rolling(200).mean().iloc[-1]
                current_daily_close = daily_df['close'].iloc[-1]
                market_state = "bull" if current_daily_close > daily_ma_200 else "bear"
                logger.info(f"Market State: {market_state.upper()}")
                
                min_sentiment = 0.55 if market_state == "bear" else 0.6
                rsi_floor = 35 if market_state == "bear" else 30
                max_volatility = 40 if market_state == "bear" else 60  # Allow higher volatility (40-60%)
            else:
                logger.warning("Using default thresholds - insufficient daily data")
                min_sentiment = 0.6
                rsi_floor = 30
                max_volatility = 50  # Higher default volatility threshold

            if not self.has_sufficient_funds and symbol not in self.active_positions:
                return False, 0.0

            if symbol in self.active_positions:
                current_price = float(df.iloc[-1]['close'])
                position = self.active_positions[symbol]
                await self.manage_stop_loss(symbol, position, current_price)

            if self.has_sufficient_funds or symbol in self.active_positions:
                tasks = []
                last_row = df.iloc[-1]
                
                # Format price data
                price_data = {
                    'timeframe': '1h',
                    'price_data': {
                        'open': df['open'].astype(float).tolist(),
                        'high': df['high'].astype(float).tolist(),
                        'low': df['low'].astype(float).tolist(),
                        'close': df['close'].astype(float).tolist()
                    }
                }
                
                # Get sentiment analysis
                try:
                    sentiment_score = await self.market_analyzer.get_market_sentiment(
                        data=price_data,  # Pass the price_data dictionary we created above
                        symbol=symbol
                    )
                    sentiment_valid = sentiment_score >= min_sentiment
                    logger.info(f"Sentiment score: {sentiment_score:.2f}")
                except Exception as e:
                    logger.error(f"Error getting sentiment: {e}")
                    sentiment_score = 0.5
                    sentiment_valid = False
                
                # Calculate technical signals
                rsi = float(last_row['rsi'])
                rsi_signal = rsi_floor <= rsi <= 70
                ema_trend = float(last_row['ema_short']) > float(last_row['ema_long'])
                macd_cross = float(last_row['macd']) > float(last_row['macd_signal'])
                current_bbw = float(last_row['bb_width'])
                bbw_valid = current_bbw < 0.5

                # Calculate volatility
                returns = df['close'].pct_change()
                volatility = returns.std() * np.sqrt(252) * 100

                # Calculate signal strength with sentiment
                signal_strength = (
                    (1 if rsi_signal else 0) +
                    (1 if ema_trend else 0) +
                    (1 if macd_cross else 0) +
                    (1 if bbw_valid else 0) +
                    (1 if sentiment_valid else 0)
                ) / 5.0

                should_trade = signal_strength >= 0.6 and volatility <= max_volatility

                logger.info(f"Analysis complete - Signal: {signal_strength:.2f}, "
                           f"Volatility: {volatility:.2f}%, Trade: {'✅' if should_trade else '❌'}")

                return should_trade, volatility

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return False, 0.0

    async def adjust_grid_parameters(self, volatility: float, current_price: float) -> None:
        """Adjust grid parameters based on volatility"""
        try:
            logger.info(f"\n{'='*30}")
            logger.info("ADJUSTING GRID PARAMETERS")
            logger.info(f"{'='*30}")
            logger.info(f"Current volatility: {volatility:.2f}%")
            
            # Base grid levels on volatility ranges
            if volatility < 1.0:  # Low volatility
                self.grid_levels = 3
                self.grid_spacing = 0.008  # 0.8%
                logger.info("Low volatility setup - Tighter grids")
            elif 1.0 <= volatility < 2.0:  # Medium-low volatility
                self.grid_levels = 4
                self.grid_spacing = 0.012  # 1.2%
                logger.info("Medium-low volatility setup")
            elif 2.0 <= volatility < 3.0:  # Medium volatility
                self.grid_levels = 5
                self.grid_spacing = 0.015  # 1.5%
                logger.info("Medium volatility setup")
            elif 3.0 <= volatility < 4.0:  # Medium-high volatility
                self.grid_levels = 6
                self.grid_spacing = 0.018  # 1.8%
                logger.info("Medium-high volatility setup")
            else:  # High volatility
                self.grid_levels = 7
                self.grid_spacing = 0.02  # 2%
                logger.info("High volatility setup - Wider grids")

            # Adjust for minimum profitability
            min_profit_needed = self.calculate_total_fees(current_price, self.min_position_size, "BTC/USD") * 2
            min_grid_spacing = min_profit_needed / current_price
            
            if self.grid_spacing < min_grid_spacing:
                self.grid_spacing = min_grid_spacing
                logger.info(f"Adjusted grid spacing to ensure profitability: {self.grid_spacing*100:.2f}%")

            # Update class attribute instead of local variable
            self.min_position_size = 0.001 if current_price >= 20000 else 0.01
            
            logger.info(f"Final grid spacing: {self.grid_spacing*100:.2f}%")
            logger.info(f"Minimum position size: {self.min_position_size}")
            logger.info(f"{'='*30}\n")

            return True

        except Exception as e:
            logger.error(f"Error adjusting grid parameters: {e}")
            return False
            

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
            # Fetch all open orders for the symbol
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            if open_orders:
                # Remove duplicates using set
                order_prices = sorted(set(order['price'] for order in open_orders))
                
                # Count orders on each side of current price
                current_price = await self.get_current_price(symbol)
                buy_orders = len([p for p in order_prices if p < current_price])
                sell_orders = len([p for p in order_prices if p > current_price])
                
                logger.info(f"Found existing grid orders for {symbol} - "
                           f"Buy orders: {buy_orders}, Sell orders: {sell_orders}, "
                           f"Prices: {order_prices}")
                return True
                
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
        """Calculate support and resistance levels using daily, 15m and 5m timeframes"""
        try:
            # Get multi-timeframe data
            daily_df = await self.get_historical_data(symbol, '1d')
            fifteen_min_df = await self.get_historical_data(symbol, timeframe='15m', limit=100)  # ~24h
            five_min_df = await self.get_historical_data(symbol, timeframe='5m', limit=72)      # Last 6h
            
            if daily_df is None or fifteen_min_df is None or five_min_df is None:
                logger.warning("No data available for S/R calculation")
                return [], []

            # Calculate daily Fibonacci levels (structural)
            daily_high = daily_df['high'].max()
            daily_low = daily_df['low'].min()
            daily_diff = daily_high - daily_low
            
            daily_fib_levels = {
                0: daily_low,
                0.236: daily_high - (daily_diff * 0.236),
                0.382: daily_high - (daily_diff * 0.382),
                0.5: daily_high - (daily_diff * 0.5),
                0.618: daily_high - (daily_diff * 0.618),
                0.786: daily_high - (daily_diff * 0.786),
                1: daily_high
            }

            # Calculate 15min S/R
            fifteen_high = fifteen_min_df['high'].rolling(window=20).max()
            fifteen_low = fifteen_min_df['low'].rolling(window=20).min()
            
            # Calculate 5min Fibonacci levels (short-term)
            recent_high = five_min_df['high'].max()
            recent_low = five_min_df['low'].min()
            diff = recent_high - recent_low
            
            short_term_fib_levels = {
                -1: recent_high + diff,          # Extension below
                0: recent_low,                   # Start level
                0.236: recent_high - (diff * 0.236),
                0.382: recent_high - (diff * 0.382),
                0.5: recent_high - (diff * 0.5),
                0.618: recent_high - (diff * 0.618),
                0.786: recent_high - (diff * 0.786),
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

            # Add daily levels first (highest priority)
            for level in daily_fib_levels.values():
                if level > recent_high:
                    resistance_levels.append(level)
                elif level < recent_low:
                    support_levels.append(level)

            # Add 15min levels
            resistance_levels.extend(fifteen_high.dropna().tolist())
            resistance_levels.extend([
                short_term_fib_levels[level] for level in [1.414, 1.618, 2.618, 3.618, 4.236]
            ])
            resistance_levels.append(recent_high)

            # Add support levels
            support_levels.extend(fifteen_low.dropna().tolist())
            support_levels.extend([
                short_term_fib_levels[level] for level in [0, 0.236, 0.382, 0.5, 0.618, 0.786]
            ])
            support_levels.append(recent_low)

            # Log all levels
            logger.info(f"\nS/R Levels for {symbol}:")
            logger.info("Daily Fibonacci Levels:")
            for k, v in daily_fib_levels.items():
                logger.info(f"- {k}: ${v:.4f}")
                
            logger.info(f"\n5min Fib Extensions: {[round(short_term_fib_levels[level], 4) for level in [1.414, 1.618, 2.618, 3.618, 4.236]]}")
            logger.info(f"5min Fib Retracements: {[round(short_term_fib_levels[level], 4) for level in [0.236, 0.382, 0.5, 0.618, 0.786]]}")
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
        """Execute grid orders with fee validation"""
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 STARTING GRID EXECUTION FOR {symbol}")
            
            # Get historical data first
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                logger.warning(f"❌ No historical data for {symbol}")
                return

            # Calculate volatility with default fallback
            try:
                volatility = await self.calculate_volatility(df)
                if volatility is None:
                    volatility = 2.0  # Default moderate volatility
                    logger.warning(f"Using default volatility of {volatility}%")
            except Exception as e:
                volatility = 2.0
                logger.error(f"Error calculating volatility: {e}")

            current_price = float(df['close'].iloc[-1])
            
            # Make sure adjust_grid_parameters is async
            await self.adjust_grid_parameters(volatility, current_price)

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

            # If position exists, use manage_stop_loss instead of direct execution
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                current_price = await self.get_current_price(symbol)
                await self.manage_stop_loss(symbol, position, current_price)
                return

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
                if trend_data['1h']['is_oversold'] and trend_data['1h']['near_support'] and trend_data['1h']['rsi'] > 45:
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
                return
            
            if not should_trade:
                logger.info(f"❌ Market conditions not favorable for {symbol}, skipping")
                return

            five_min_df = await self.get_historical_data(symbol, '5m', 50)
            if not await self._analyze_5m(five_min_df, symbol):
                logger.info("❌ 5min conditions not met")
                return
                
            # Adjust grid parameters based on volatility
            await self.adjust_grid_parameters(volatility, float(df['close'].iloc[-1]))
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
                rsi = talib.RSI(df['close'], 14).iloc[-1]  
                macd_indicator = talib.trend.MACD(df['close'])
                macd_line = macd_indicator.macd()
                signal_line = macd_indicator.macd_signal()
                histogram = macd_indicator.macd_diff()
                macd_bullish = (
                    (macd_line.iloc[-1] > signal_line.iloc[-1]) and
                    (histogram.iloc[-1] > histogram.iloc[-2]) and
                    (macd_line.iloc[-1] > 0)
                )
                valid_for_trade = (
                    (momentum_aligned or volume_confirmed) and  # AND instead of OR
                    (

                        (would_be_side == "buy" and is_near_support and rsi < 35) or
                        (would_be_side == "sell" and is_near_resistance and rsi > 65)
                    ) and 
                    macd_bullish and  
                    normalized_confidence > 8
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
                
                # Handle spot position logic
                if is_spot:
                    if side == "sell":
                        # Only allow sells if we have a position
                        if symbol in self.positions and self.positions[symbol]['size'] > 0:
                            # Verify actual spot balance
                            asset = symbol.split('/')[0]
                            spot_balance = (await self.exchange.fetch_balance()).get(asset, {}).get('free', 0)
                            if spot_balance < size:
                                logger.info(f"❌ Insufficient {asset} balance for sell order: {spot_balance} < {size}")
                                continue
                        else:
                            logger.info(f"❌ Skipping sell order at ${price:.4f} - No spot position")
                            continue
                    else:  # For buys in spot
                        # Check USD balance using the existing logic
                        pass
                
                # Get relevant S/R levels based on order side
                if side == "buy":
                    relevant_levels = support_levels
                    level_type = "support"
                    max_distance = 0.01  # 1% from support
                else:
                    relevant_levels = resistance_levels 
                    level_type = "resistance"
                    max_distance = 0.008  # 0.8% from resistance

                # Check proximity to relevant S/R
                near_level = any(abs(price - level)/level < max_distance 
                                for level in relevant_levels[:3])
                
                if not near_level:
                    logger.info(f"Skipping {side} order at {price} - not near {level_type}")
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

                # Add USD balance check for spot buys
                if is_spot and side == "buy":
                    try:
                        usd_balance = (await self.exchange.fetch_balance())['USD']['free']
                        order_cost = price * size
                        if order_cost > usd_balance:
                            logger.warning(f"❌ Insufficient USD balance: {usd_balance:.2f} < {order_cost:.2f}")
                            continue
                    except Exception as e:
                        logger.error(f"Balance check error: {e}")
                        continue
                
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
                    
                    # Refresh grid levels after each trade
                    await self.refresh_grid_levels(symbol)
                    
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
            traceback.print_exc()

    async def get_current_price(self, symbol: str) -> float:
        """Get current price using last trade price with bid/ask spread awareness"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            # Use last price but verify against spread
            spread = ticker['ask'] - ticker['bid']
            if spread > ticker['last'] * 0.001:  # More than 0.1% spread
                return (ticker['bid'] + ticker['ask']) / 2  # Use mid price
            return ticker['last']
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

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

    async def calculate_position_size(self, current_price: float, symbol: str) -> float:
        """Safe position sizing with USD balance check for spot buys"""
        try:
            # Split symbol into base/quote currencies (e.g. POPCAT/USD)
            base_currency, quote_currency = symbol.split('/')
            
            market = self.exchange.market(symbol)
            min_size = float(market['limits']['amount']['min'])
            precision = int(market['precision']['amount'])
            
            # Get balances
            balance = await self.exchange.fetch_balance()
            portfolio_value = float(balance['total']['USD'])
            
            # Check minimum required capital in USD
            min_usd_required = min_size * current_price
            if portfolio_value < min_usd_required:
                logger.warning(f"❌ Insufficient USD capital for {symbol}")
                logger.warning(f"Need ${min_usd_required:.2f}, have ${portfolio_value:.2f}")
                return 0.0
                
            # Calculate max USD to risk
            risk_usd = min(
                portfolio_value * self.risk_per_trade,
                float(balance.get(quote_currency, {}).get('free', 0))  # Available USD
            )
            
            # Convert USD risk to base currency size
            size = risk_usd / current_price
            
            # Enforce minimum size after USD conversion
            if size < min_size:
                logger.warning(f"⚠️ Increasing size to meet {symbol} minimum {min_size}")
                adjusted_size = min_size * 1.01  # Add 1% buffer
                
                # Validate adjusted USD risk
                adjusted_usd_risk = adjusted_size * current_price
                if adjusted_usd_risk > (portfolio_value * self.max_risk_per_trade):
                    logger.warning(f"❌ Minimum size exceeds max USD risk (${adjusted_usd_risk:.2f})")
                    return 0.0
                    
                size = adjusted_size
            
            # Convert to proper precision
            size = round(float(size), precision)
            
            # Spot balance check
            if self.is_spot:
                asset = symbol.split('/')[0]
                available = float(balance.get(asset, {}).get('free', 0))
                if available < size:
                    logger.warning(f"⚠️ Adjusting size to available {asset} balance: {available}")
                    size = min(size, available)
            
            # Final validation
            final_usd_cost = size * current_price
            available_usd = float(balance.get(quote_currency, {}).get('free', 0))
            if final_usd_cost > available_usd:
                logger.warning(f"⚠️ Adjusting size to available USD balance: ${available_usd:.2f}")
                size = available_usd / current_price
                
            if size < min_size:
                logger.error(f"❌ Final size {size:.4f} {base_currency} still below minimum")
                return 0.0
                
            logger.info(f"✅ Valid buy size: {size:.4f} {base_currency} (${final_usd_cost:.2f} USD)")
            return round(size, precision)
            
        except Exception as e:
            logger.error(f"❌ Position size error: {e}")
            return 0.0

    async def manage_stop_loss(self, symbol: str, position: Dict, current_price: float) -> None:
        """Manage stop loss and take profits for both spot and margin positions"""
        try:
            # Add this debug log at start
            logger.info(f"🏁 STARTING STOP LOSS MANAGEMENT FOR {symbol}")
            logger.debug(f"Position data: {position}")
            logger.debug(f"Current price: {current_price}")

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
                        'current_position_size': position_size,  # Add this to track remaining size
                        'entry_price': entry_price,
                        'is_spot': is_spot,
                        'position_side': position_side,  # Add this to track position side
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
            position_side = position_data.get('position_side', 'long')  # Get from position data
            
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
                orders = await self.exchange.fetch_closed_orders(symbol, limit=5)
                recent_buys = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed']
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
                        await self.execute_stop_loss(
                            symbol=symbol,
                            position_size=position_data['current_position_size'],  # Use tracked size
                            is_spot=is_spot,
                            reason="stop loss"
                        )
                        return  # Add return here to handle the stop execution
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
                logger.info(f"TP Size: {tp_size}")
                
                try:
                    # Calculate TP2 size (30% of original position)
                    tp_size = round(position_size * 0.30, 8)  # 30% for TP2
                    logger.info(f"TP2 Size: {tp_size}")
                    
                    # Execute TP2
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
                        if tp_size == position_data['current_position_size']:  # If we took all remaining
                            position_data['current_position_size'] = 0
                        else:
                            position_data['current_position_size'] = position_size * 0.40  # Track remaining 40%
                        logger.info(f"✅ TP2 executed successfully")
                    else:
                        logger.error(f"❌ Failed to execute TP2")
                        
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
            logger.error(f"Critical error in stop management: {e}")
            logger.error(traceback.format_exc())

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
                        
                        # Try partial TP first
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
                            # Try partial order first
                            order = await self.exchange.create_market_order(
                                symbol=symbol,
                                side='sell',
                                amount=size,
                                params=params
                            )
                            logger.info(f"Partial TP executed: {size} {asset}")
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
        """Monitor positions with accurate price tracking and stop loss execution"""
        try:
            logger.info("\n=== MONITORING POSITIONS ===")
            
            # Get verified positions first
            verified_positions = await self.verify_actual_positions()
            
            # Monitor each verified position
            for position in verified_positions:
                symbol = position['symbol']
                try:
                    # Get current price
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = float(ticker['last'])
                    
                    # First manage stop loss levels
                    await self.manage_stop_loss(
                        symbol=symbol,
                        position=position,
                        current_price=current_price
                    )
                    
                    logger.info(f"Monitored {symbol} at ${current_price}")
                    
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
                    continue
                    
            logger.info("=== FINISHED POSITION MONITORING ===\n")
            
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")

    async def start(self):
        while True:  # Outer loop for strategy restart
            try:
                # Initialize tracking dictionaries ONCE
                self.positions = {}
                self.last_grid_check = {}
                self.active_positions = {}
                
                # Initialize symbol mappings
                self.symbol_map = {
                    'PF_XBTUSD': 'BTC/USD',
                    'PF_XRPUSD': 'XRP/USD',
                    'PI_ETHUSD': 'ETH/USD',
                    'PF_DOGEUSD': 'DOGE/USD',
                    'PF_LDOUSD': 'LDO/USD',
                    'PF_ADAUSD': 'ADA/USD',
                    'PF_MATICUSD': 'MATIC/USD',
                    'PF_FILUSD': 'FIL/USD',
                    'PF_APEUSD': 'APE/USD',
                    'PF_GMXUSD': 'GMX/USD',
                    'PF_BATUSD': 'BAT/USD',
                    'PF_XLMUSD': 'XLM/USD',

                    'PF_EOSUSD': 'EOS/USD',
                    'PF_OPUSD': 'OP/USD',
                    'PF_AAVEUSD': 'AAVE/USD',
                    'PF_LINKUSD': 'LINK/USD',
                    'PF_XMRUSD': 'XMR/USD',
                    'PF_ATOMUSD': 'ATOM/USD',
                    'PF_DOTUSD': 'DOT/USD',

                    'PF_ALGOUSD': 'ALGO/USD',
                    'PF_TRXUSD': 'TRX/USD',
                    'PF_SOLUSD': 'SOL/USD',
                    'PF_AVAXUSD': 'AVAX/USD',
                    'PF_UNIUSD': 'UNI/USD',
                    'PF_SNXUSD': 'SNX/USD',

                    'PF_NEARUSD': 'NEAR/USD',
                    'PF_FTMUSD': 'FTM/USD',
                    'PF_ARBUSD': 'ARB/USD',
                    'PF_COMPUSD': 'COMP/USD',
                    'PF_YFIUSD': 'YFI/USD'
                }
                self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
                
                # Retry loop for exchange connection
                retry_count = 0
                while retry_count < 5:
                    try:
                        # Get initial balance and verify positions
                        balance = await self.exchange.fetch_balance()
                        
                        # Verify spot positions and initialize tracking
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
                                    if buy_orders:
                                        entry_price = float(buy_orders[0]['price'])
                                        # Update both tracking systems
                                        self.active_positions[symbol] = {
                                            'info': {
                                                'price': entry_price,
                                                'size': spot_balance,
                                                'side': 'buy',
                                                'is_spot': True
                                            }
                                        }
                                        
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
                                        logger.info(f"Initialized position tracking for {symbol} at ${entry_price}")
                                except Exception as e:
                                    logger.error(f"Error initializing {symbol}: {e}")
                                    continue
                        
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
                        
                        # Monitor positions first
                        logger.info("\n=== MONITORING ACTIVE POSITIONS ===")
                        
                        # Get both spot and futures positions
                        positions = await self.verify_actual_positions()
                        
                        # Update active positions - using .items() since positions is now a dictionary
                        for symbol, pos in positions.items():
                            if symbol not in self.active_positions:
                                # Get entry price from order history
                                try:
                                    orders = await self.exchange.fetch_closed_orders(symbol, limit=5)
                                    buy_orders = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed']
                                    if buy_orders:
                                        entry_price = float(buy_orders[0]['price'])
                                    else:
                                        entry_price = float(pos['info']['price'])
                                        
                                    self.active_positions[symbol] = {
                                        'info': {
                                            'price': entry_price,
                                            'size': float(pos['info']['size']),
                                            'side': 'buy',
                                            'is_spot': True
                                        }
                                    }
                                    logger.info(f"Found position: {symbol}")
                                    logger.info(f"Size: {pos['info']['size']}")
                                    logger.info(f"Entry: {entry_price}")
                                except Exception as e:
                                    logger.error(f"Error initializing position for {symbol}: {e}")
                            else:
                                # Position already tracked
                                logger.info(f"Monitoring position: {symbol}")
                                logger.info(f"Size: {self.active_positions[symbol]['info']['size']}")
                                logger.info(f"Entry: {self.active_positions[symbol]['info']['price']}")
                        
                        if not self.active_positions:
                            logger.info("No active positions to monitor")
                        logger.info("=== FINISHED POSITION MONITORING ===\n")
                        
                        # Manage stop losses for active positions
                        for symbol, position in positions.items():
                            if symbol in self.active_positions:
                                current_price = await self.get_current_price(symbol)
                                await self.manage_stop_loss(symbol, position, current_price)
                        
                        # Existing grid logic
                        for symbol in self.active_symbols:
                            try:
                                await self.execute_grid_orders(symbol)
                            except Exception as e:
                                logger.error(f"Error executing grid orders for {symbol}: {e}")
                                continue
                        
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
                            
                        await asyncio.sleep(60)  # Changed to 60-second interval
                        
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
        """Enhanced scanner that finds candidates for filtering"""
        try:
            # Get all available symbols from exchange
            all_markets = await self.exchange.fetch_markets()
            all_symbols = [m['symbol'] for m in all_markets if m['active']]
            
            # Scan for symbols meeting basic conditions
            scan_tasks = [
                self._scan_symbol(sym) 
                for sym in all_symbols[:50]  # Scan top 50 by default
            ]
            candidates = await asyncio.gather(*scan_tasks)
            valid_candidates = [sym for sym in candidates if sym]
            
            # Pass candidates through full filtering
            self.active_symbols = await self.filter_symbols(valid_candidates)
            
            logger.info(f"Updated trading symbols: {self.active_symbols}")

        except asyncio.CancelledError:
            logger.info("Background market scanner stopped")
        except Exception as e:
            logger.error(f"Error in background market scanner: {e}")

    async def _scan_symbol(self, symbol: str) -> Union[str, None]:
        """Market scanner pre-filter using analyzer conditions"""
        try:
            # Add USD pair check
            if not symbol.upper().endswith('/USD'):
                logger.debug(f"Skipping non-USD pair: {symbol}")
                return None
                
            # Get recent data for analysis
            df = await self.get_historical_data(symbol, '4h', 50)
            if df is None or len(df) < 20:
                return None
                
            # Check analyzer conditions
            if not self.market_analyzer.check_volatility(df):
                return None
                
            if not self.market_analyzer.check_trend_strength(df):
                return None
                
            if (datetime.now() - pd.to_datetime(df.index[-1])).seconds > 3600:
                logger.warning(f"Stale data for {symbol}")
                return None
                
            return symbol
                
        except Exception as e:
            logger.error(f"Scan failed for {symbol}: {e}")
            return None

    async def verify_stop_orders(self, symbol: str, position: dict) -> None:
        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
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
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
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
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
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
        """Calculate grid levels using multiple methods with confidence scoring"""
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
                    tf_df['rsi'] = talib.RSI(tf_df['close'], timeperiod=14)
                    

                    # MACD
                    macd_line, signal_line, macd_hist = talib.MACD(
                        tf_df['close'], 
                        fastperiod=self.macd_fast,  # Correct parameter name
                        slowperiod=self.macd_slow,   # Correct parameter name
                        signalperiod=self.macd_signal
                    )
                    tf_df['macd'] = macd_line
                    tf_df['macd_signal'] = signal_line
                    tf_df['macd_hist'] = macd_hist
                    
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
            
            # Guard against zero position size
            if avg_position_size <= 0:
                logger.error(f"Invalid position size for {symbol}")
                return []

            total_fees = self.calculate_total_fees(current_price, avg_position_size, symbol)
            min_profit_needed = total_fees * 2

            logger.info(f"\nGrid Profitability Analysis for {symbol}:")
            logger.info(f"Min Profit Needed: ${min_profit_needed:.4f}")
            logger.info(f"Current Grid Spacing: {self.grid_spacing*100:.2f}%")
            logger.info(f"Estimated Profit per Grid: ${(current_price * self.grid_spacing * avg_position_size):.4f}")

            # Volume profile analysis with S/R and Technical boost
            try:
                price_bins = pd.qcut(df['close'], q=10, duplicates='drop')
                volume_profile = df.groupby(price_bins, observed=True)['volume'].sum()
                high_volume_levels = volume_profile[volume_profile > volume_profile.mean()].index
                
                for level in high_volume_levels:
                    price = round(level.mid, 2)
                    base_conf = 0.3
                    sr_boost = self.get_sr_boost(price, support_levels, resistance_levels)
                    tech_boost = self.calculate_technical_boost(price, tech_scores)
                    levels_with_confidence[price] = levels_with_confidence.get(price, 0) + base_conf + sr_boost + tech_boost
            except Exception as e:
                logger.error(f"Error in volume profile calculation: {e}")

            # Moving Averages with S/R and Technical boost
            try:
                ma_periods = [20, 50, 100, 200]
                for period in ma_periods:
                    ma = df['close'].rolling(period).mean().iloc[-1]
                    if abs(ma - current_price) / current_price <= 0.10:
                        price = round(ma, 2)
                        base_conf = 0.2
                        sr_boost = self.get_sr_boost(price, support_levels, resistance_levels)
                        tech_boost = self.calculate_technical_boost(price, tech_scores)
                        levels_with_confidence[price] = levels_with_confidence.get(price, 0) + base_conf + sr_boost + tech_boost
            except Exception as e:
                logger.error(f"Error in MA calculation: {e}")

            # Market Structure with S/R boost
            try:
                window = 5
                for i in range(window, len(df) - window):
                    if (df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max() and 
                        df['volume'].iloc[i] > df['volume'].iloc[i-window:i+window+1].mean()):
                        price = round(df['high'].iloc[i], 2)
                        base_conf = 0.3
                        sr_boost = self.get_sr_boost(price, support_levels, resistance_levels)  # Fixed function call
                        levels_with_confidence[price] = levels_with_confidence.get(price, 0) + base_conf + sr_boost
                    
                    if (df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min() and 
                        df['volume'].iloc[i] > df['volume'].iloc[i-window:i+window+1].mean()):
                        price = round(df['low'].iloc[i], 2)
                        base_conf = 0.3
                        sr_boost = self.get_sr_boost(price, support_levels, resistance_levels)  # Fixed function call
                        levels_with_confidence[price] = levels_with_confidence.get(price, 0) + base_conf + sr_boost
            except Exception as e:
                logger.error(f"Error in market structure calculation: {e}")

            # Recent Price Action with S/R boost
            try:
                recent_df = df.tail(50)
                price_clusters = pd.qcut(recent_df['close'], q=5, duplicates='drop')
                recent_levels = price_clusters.value_counts().index
                
                for level in recent_levels:
                    price = round(level.mid, 2)
                    base_conf = 0.2
                    sr_boost = self.get_sr_boost(price, support_levels, resistance_levels)  # Fixed function call
                    levels_with_confidence[price] = levels_with_confidence.get(price, 0) + base_conf + sr_boost
            except Exception as e:
                logger.error(f"Error in recent price action calculation: {e}")

            # Filter and sort levels (keep your existing logic)
            min_confidence = 0.3
            valid_levels = {
                price: conf for price, conf in levels_with_confidence.items()
                if conf >= min_confidence and abs(price - current_price) / current_price <= 0.10
            }

            # Minimum spacing and filtering (keep your existing logic)
            min_distance = max(current_price * 0.02, min_profit_needed / (current_price * avg_position_size))
            filtered_levels = []
            
            for price in sorted(valid_levels.keys()):
                if not filtered_levels or abs(price - filtered_levels[-1]) >= min_distance:
                    filtered_levels.append(price)

            # Add current price if not too close
            if all(abs(current_price - level) >= min_distance for level in filtered_levels):
                filtered_levels.append(current_price)
                filtered_levels.sort()

            # Log results
            logger.info(f"\nGrid levels for {symbol}:")
            logger.info(f"Current price: {current_price}")
            for level in filtered_levels:
                confidence = valid_levels.get(level, 'current')
                logger.info(f"Level: {level}, Confidence: {confidence if confidence != 'current' else 'Current Price'}")

            logger.info(f"\nSupport and Resistance Analysis for {symbol}:")
            logger.info(f"Support levels: {[round(s, 4) for s in support_levels]}")
            logger.info(f"Resistance levels: {[round(r, 4) for r in resistance_levels]}")
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
                        'info': {
                            'price': entry_price,
                            'size': spot_balance,
                            'side': 'buy'
                        }
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
        """Fetch and categorize instruments with minimum notional validation"""
        try:
            # Get portfolio-based limits
            balance = await self.exchange.fetch_balance()
            portfolio_value = float(balance['total']['USD'])
            max_min_notional = portfolio_value * self.max_risk_per_trade
            
            markets = await self.exchange.fetch_markets()
            tickers = await self.exchange.fetch_tickers()
            
            categorized_symbols = {
                'tier1': [],  # $100-500
                'tier2': [],  # $500-2000
                'tier3': [],  # $2000-5000
                'tier4': []   # $5000+
            }

            for market in markets:
                try:
                    symbol = market['symbol']
                    
                    # Filter criteria
                    if not (symbol.endswith('/USD') and 
                           market['spot'] and 
                           market['active'] and
                           symbol not in self.STABLE_BLACKLIST):
                        continue
                    
                    # Get market details
                    min_amount = market['limits']['amount']['min']
                    price = tickers[symbol]['last'] if symbol in tickers else 0
                    if not price:
                        continue
                    
                    # Calculate minimum notional value
                    min_notional = min_amount * price
                    if min_notional > max_min_notional:
                        logger.info(f"Excluding {symbol} - Min notional ${min_notional:.2f} > Max ${max_min_notional:.2f}")
                        continue
                    
                    # Categorize by price tier
                    if price < 500:
                        tier = 'tier1'
                    elif price < 2000:
                        tier = 'tier2'
                    elif price < 5000:
                        tier = 'tier3'
                    else:
                        tier = 'tier4'
                    
                    categorized_symbols[tier].append(symbol)
                    logger.info(f"Approved {symbol} | Price: ${price:.2f} | Min Notional: ${min_notional:.2f}")

                except Exception as e:
                    logger.error(f"Error processing {market.get('id', 'unknown')}: {e}")
                    continue

            # Log final selection
            logger.info("\nFinal Tradeable Instruments:")
            for tier, symbols in categorized_symbols.items():
                logger.info(f"\n{tier.upper()} ({len(symbols)} symbols):")
                for sym in sorted(symbols):
                    if sym in tickers:
                        price = tickers[sym]['last']
                        min_amount = next(m['limits']['amount']['min'] for m in markets if m['symbol'] == sym)
                        logger.info(f"- {sym} | Price: ${price:.2f} | Min Size: {min_amount}")

            return categorized_symbols

        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return {k: [] for k in ['tier1', 'tier2', 'tier3', 'tier4']}

    async def analyze_multi_timeframe_trend(self, symbol: str, timeframe_results: dict = None) -> dict:
        """Analyze multi-timeframe trend using pre-calculated results if available"""
        if timeframe_results:
            # Use pre-calculated results
            trend_results = {}
            
            # Map timeframe analyses to expected format
            if '5m' in timeframe_results:
                trend_results['15m'] = self._format_trend_result(timeframe_results['5m'])
            if '1h' in timeframe_results:
                trend_results['1h'] = self._format_trend_result(timeframe_results['1h'])
            if 'daily' in timeframe_results:
                trend_results['1d'] = self._format_trend_result(timeframe_results['daily'])
            
            return trend_results
        
        # Fallback to original calculation if no pre-calculated results
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
                    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
                    

                    # Add MACD
                    macd_line, signal_line, macd_hist = talib.MACD(
                        df['close'], 
                        fastperiod=self.macd_fast, 
                        slowperiod=self.macd_slow,   
                        signalperiod=self.macd_signal
                    )
                    df['macd'] = macd_line
                    df['macd_signal'] = signal_line
                    df['macd_histogram'] = macd_hist
                    
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

    def _format_trend_result(self, analysis_result: dict) -> dict:
        """Format individual timeframe analysis into trend result format"""
        return {
            'trend': 'bullish' if analysis_result.get('ema_bullish', False) else 'bearish',
            'strength': analysis_result.get('signal_strength', 0) * 100,
            'rsi': analysis_result.get('rsi', 50),
            'macd_hist': analysis_result.get('macd_histogram', 0),
            'is_oversold': analysis_result.get('rsi', 50) < 30,
            'is_overbought': analysis_result.get('rsi', 50) > 70,
            'support_levels': analysis_result.get('support_levels', []),
            'resistance_levels': analysis_result.get('resistance_levels', []),
            'near_support': analysis_result.get('near_support', False),
            'near_resistance': analysis_result.get('near_resistance', False)
        }

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
        """Market-analyzer enhanced symbol filtering"""
        try:
            # Stage 0: Remove blacklisted symbols
            filtered = [s for s in symbols if s not in self.STABLE_BLACKLIST]
            
            # Stage 1: Liquidity filter (lines 3361-3371)
            liquidity_filtered = await self._quick_liquidity_filter(filtered)
            
            # Stage 2: Parallel market analysis (new)
            analysis_tasks = [
                self._analyze_symbol(sym)
                for sym in liquidity_filtered[:20]  # Analyze top 20
            ]
            results = await asyncio.gather(*analysis_tasks)
            
            # Stage 3: Scoring and sorting with dynamic threshold
            sorted_symbols = sorted(
                [(sym, score) for sym, score in results if score is not None],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top 10% or minimum 3 symbols
            top_count = max(3, int(len(sorted_symbols) * 0.1))
            return [sym for sym, _ in sorted_symbols[:top_count]]
        
        except Exception as e:
            logger.error(f"Filter error: {e}")
            return []

    async def _analyze_symbol(self, symbol: str) -> Tuple[str, float]:
        """Market analyzer-powered symbol validation"""
        try:
            # Get multi-timeframe data (lines 3390-3393)
            df = await self.get_historical_data(symbol, '4h', limit=100)
            if df is None or len(df) < 50:
                return symbol, 0.0

            # Get analyzer scores (lines 460-472, 513-515)
            price_data = {
                'timeframe': '4h',
                'price_data': {
                    'open': df['open'].tolist(),
                    'high': df['high'].tolist(),
                    'low': df['low'].tolist(),
                    'close': df['close'].tolist()
                }
            }
            
            sentiment_score = await self.market_analyzer.get_market_sentiment(price_data, symbol)
            tech_signals = self.market_analyzer.calculate_signals(df)
            
            # Composite score (lines 537-543)
            confidence = (sentiment_score + tech_signals['signal_strength']) * 5  # Scale 0-10
            logger.info(f"Symbol {symbol} score: Sentiment {sentiment_score:.2f}, Tech {tech_signals['signal_strength']:.2f} → {confidence:.2f}")
            
            if self.is_weekend():
                confidence *= 1.2  # 20% weekend volatility bonus
                logger.info(f"Weekend adjustment → {confidence:.2f}")
            
            return symbol, confidence
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return symbol, 0.0

    async def _quick_liquidity_filter(self, symbols):
        """Fast pre-filter based on volume and balance"""
        balance = await self.get_account_balance()
        max_size = balance * 0.2
        
        tickers = await self.exchange.fetch_tickers()
        return [
            sym for sym in symbols
            if tickers.get(sym, {}).get('quoteVolume', 0) > 1_000_000
            and tickers[sym]['last'] * 3 <= max_size
        ]

    async def validate_entry_conditions(self, symbol: str, df: pd.DataFrame) -> bool:
        """Optimized entry validation with parallel technical analysis"""
        try:
            # Validate input DataFrame
            if df is None or len(df) < self.atr_periods:
                logger.warning(f"Insufficient data for {symbol}")
                return False

            # Calculate position size first
            size = await self.calculate_position_size(symbol)
            if size <= 0:
                logger.warning(f"Skipping {symbol} - invalid size")
                return False

            # Batch load multi-timeframe data with validation
            timeframes = ['1d', '4h', '1h', '5m']
            dfs = await asyncio.gather(
                *[self.get_historical_data(symbol, tf) for tf in timeframes],
                return_exceptions=True
            )
            
            # Check for exceptions in timeframe data
            if any(isinstance(d, Exception) for d in dfs):
                logger.error(f"Error fetching timeframe data for {symbol}")
                return False
                
            daily_df, four_hour_df, hourly_df, five_min_df = dfs
            
            # Validate all timeframes have sufficient data and required columns
            if any(d is None or len(d) < self.atr_periods or not {'open','high','low','close'}.issubset(d.columns) 
                   for d in [daily_df, four_hour_df, hourly_df, five_min_df]):
                logger.warning(f"Insufficient multi-timeframe data for {symbol}")
                return False

            # Parallel technical analysis with error handling
            try:
                analysis_tasks = [
                    self._analyze_daily(daily_df, symbol),
                    self._analyze_4h(four_hour_df, symbol),
                    self._analyze_1h(hourly_df, symbol),
                    self._analyze_5m(five_min_df, symbol)
                ]
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

                # Check for exceptions in results
                if any(isinstance(r, Exception) for r in results):
                    logger.error(f"Analysis error in timeframe analysis")
                    return False
                
                daily_ok, fourh_ok, hour_ok, five_min_ok = results
                
            except Exception as e:
                logger.error(f"Technical analysis error: {e}")
                return False

            # Combined validation
            if not all([daily_ok, fourh_ok, hour_ok, five_min_ok]):
                return False

            # Batch sentiment check with validation
            if len(daily_df) > 0:  # Make sure we have data
                # Validate OHLC columns exist in daily data
                if not all(col in daily_df.columns for col in ['open', 'high', 'low', 'close']):
                    logger.error(f"Missing OHLC columns in daily data for {symbol}")
                    return False
                
                sentiment_params = {
                    'timeframe': '1d',
                    'price_data': {
                        'open': daily_df['open'].astype(float).tolist(),
                        'high': daily_df['high'].astype(float).tolist(),
                        'low': daily_df['low'].astype(float).tolist(),
                        'close': daily_df['close'].astype(float).tolist()
                    }
                }
                try:
                    sentiment_score = await self.market_analyzer.get_market_sentiment(sentiment_params, symbol)
                    if sentiment_score < 0.6:
                        logger.info(f"❌ Insufficient sentiment score: {sentiment_score:.2f}")
                        return False
                except Exception as e:
                    logger.error(f"Sentiment analysis error: {e}")
                    return False
            else:
                logger.warning("Insufficient data for sentiment analysis")
                return False

            # New volatility check ===============
            try:
                current_price = float(df['close'].iloc[-1])
                volatility = await self.get_asset_volatility(symbol)
                
                if volatility >= 0.6:  # 60% volatility threshold
                    logger.info(f"❌ Volatility too high: {volatility*100:.2f}%")
                    return False
                logger.info(f"✅ Acceptable volatility: {volatility*100:.2f}%")
            except Exception as e:
                logger.error(f"Volatility check failed: {e}")
                return False

            # Add bear market bounce check here 
            market_state = "bull" if daily_df['close'].iloc[-1] > daily_df['close'].iloc[-200] else "bear"
            if market_state == "bear":
                if not await self.detect_bear_bounce(five_min_df):
                    logger.info("❌ No bear market bounce detected")
                    return False

            # Final price structure check with validation
            try:
                if len(df['close']) == 0:
                    logger.error("Empty price data")
                    return False
                    
                current_price = float(df['close'].iloc[-1])
                resistance_check = await self._check_resistance_levels(symbol, current_price)
                
                # Position validation - only if position exists
                if symbol in self.positions:
                    position = self.positions[symbol]
                    if isinstance(position, dict):
                        entry_price = position.get('entry_price')
                        if entry_price is None:
                            logger.warning(f"No entry price found for existing position in {symbol}")
                    else:
                        logger.warning(f"Invalid position data format for {symbol}")
                
                return resistance_check
                
            except Exception as e:
                logger.error(f"Price structure check error: {e}")
                return False

        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return False

    async def _analyze_daily(self, df: pd.DataFrame, symbol: str) -> bool:
        """Enhanced daily analysis with reversal confirmation"""
        try:
            # Existing swing/EMA analysis
            support_levels, resistance_levels = await self.calculate_support_resistance(df, symbol)
            
            # Get current price from the existing pattern
            current_price = float(df['close'].iloc[-1])
            
            # Add 200 EMA trend check
            ema200 = talib.EMA(df['close'], 200).iloc[-1]
            price_above_ema200 = current_price > ema200
            
            # Determine market state (matches line 3507-3515)
            market_state = "bull" if price_above_ema200 else "bear"
            
            logger.info(f"200 EMA: {ema200:.4f}")
            logger.info(f"Market State: {market_state.upper()} | Price vs 200 EMA: {'✅ Above' if price_above_ema200 else '❌ Below'}")
            
            # New reversal indicators
            rsi = talib.RSI(df['close'], 14).iloc[-1]
            rsi_prev = talib.RSI(df['close'], 14).iloc[-2]
            macd_line, signal_line, _ = talib.MACD(df['close'])
            

            # Dynamic criteria based on market state
            rsi_threshold = 30 if market_state == "bull" else 25
            volume_multiplier = 1.25 if market_state == "bull" else 1.1
            
            # Bullish reversal criteria
            rsi_bullish = (rsi > rsi_threshold) and (rsi_prev <= rsi_threshold)
            macd_bullish = (macd_line.iloc[-1] > signal_line.iloc[-1]) and \
                          (macd_line.iloc[-2] <= signal_line.iloc[-2])
            
            logger.info(f"Daily Reversal Checks:")
            logger.info(f"RSI Cross >{rsi_threshold}: {'✅' if rsi_bullish else '❌'} ({rsi_prev:.1f} → {rsi:.1f})")
            logger.info(f"MACD Bull Cross: {'✅' if macd_bullish else '❌'}")

            # ATR validation with market-based ranges
            atr = talib.ATR(df['high'], df['low'], df['close'], 14).iloc[-1]
            atr_pct = (atr / current_price) * 100
            atr_min, atr_max = (2.0, 4.0) if market_state == "bear" else (1.5, 5.0)
            

            logger.info(f"Daily ATR: {atr_pct:.1f}%")
            
            if not (atr_min <= atr_pct <= atr_max):
                logger.info(f"❌ Daily ATR out of {market_state} range ({atr_min}-{atr_max}%): {atr_pct:.1f}%")
                return False

            # Volume validation with dynamic multiplier
            volume = df['volume'].iloc[-1]
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            strong_volume = volume > vol_ma * volume_multiplier
            
            logger.info(f"Volume Check: {volume:,.2f} vs MA {vol_ma:,.2f}")
            logger.info(f"Volume Strength: {'✅' if strong_volume else '❌'} (Req: {volume_multiplier}x)")
            
            # Combined validation with market-specific rules
            return (
                (price_above_ema200 if market_state == "bull" else True) and
                self._ema_bullish_cross(df) and
                (rsi_bullish or macd_bullish) and
                strong_volume and
                (await self._price_position_checks(current_price, support_levels, resistance_levels))
            )
            
        except Exception as e:
            logger.error(f"Daily analysis error: {e}")
            return False

    def _ema_bullish_cross(self, df: pd.DataFrame) -> bool:
        """Existing EMA cross check from line 3447-3466"""
        df['ema9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema21'] = talib.EMA(df['close'], timeperiod=21)
        current_cross = df['ema9'].iloc[-1] > df['ema21'].iloc[-1]
        previous_cross = df['ema9'].iloc[-2] <= df['ema21'].iloc[-2]
        return current_cross and previous_cross


    async def _analyze_4h(self, df: pd.DataFrame, symbol: str) -> bool:
        """4-hour trend analysis - Relaxed Version"""
        try:
            # Simplified EMA Structure
            df['ema9'] = talib.EMA(df['close'], 9)
            
            # Price Relationships (keep basic trend check)
            price_above_ema9 = df['close'].iloc[-1] > df['ema9'].iloc[-1]
            
            # Momentum Indicators (keep core requirements)
            macd_line, signal_line, _ = talib.MACD(df['close'])
            histogram = macd_line - signal_line

            macd_bullish = (
                (macd_line.iloc[-1] > signal_line.iloc[-1]) and
                (histogram.iloc[-1] > histogram.iloc[-2])
            )
            
            rsi = talib.RSI(df['close'], 14).iloc[-1]
            rsi_above_45 = rsi > 45  # Relaxed from 50
            
            # Volatility Check (keep existing)
            current_price = df['close'].iloc[-1]
            atr = talib.ATR(df['high'], df['low'], df['close'], 14).iloc[-1]
            valid_atr = 1.0 <= (atr/current_price)*100 <= 3.5
            
            # Volume Validation (relaxed multiplier)
            volume = df['volume'].iloc[-1]
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            strong_volume = volume > vol_ma * 1.1  # Reduced from 1.2
            
            # Support Check (wider threshold)
            support_levels, _ = await self.calculate_support_resistance(df, symbol)
            near_support = any(abs(current_price - s)/s < 0.015 for s in support_levels[:3])
            
            logger.info(f"\n4H Analysis for {symbol}:")
            logger.info(f"Price > EMA9: {'✅' if price_above_ema9 else '❌'}") 
            logger.info(f"MACD Bullish: {'✅' if macd_bullish else '❌'} | RSI: {rsi:.1f}")
            logger.info(f"ATR: {(atr/current_price)*100:.1f}% | Volume: {'✅' if strong_volume else '❌'}")
            logger.info(f"Near Support: {'✅' if near_support else '❌'}")

            return all([
                price_above_ema9,
                macd_bullish,
                rsi_above_45,
                valid_atr,
                strong_volume,
                near_support
            ])
            
        except Exception as e:
            logger.error(f"4H analysis error: {e}")
            return False

    async def _analyze_1h(self, df: pd.DataFrame, symbol: str) -> bool:
        """1-hour trend confirmation for 15min entries - Relaxed Version"""
        try:
            # Keep EMA calculations for reference but remove cross requirement
            df['ema9'] = talib.EMA(df['close'], 9)
            df['ema21'] = talib.EMA(df['close'], 21)
            
            # Simplified price relationship check
            price_above_ema9 = df['close'].iloc[-1] > df['ema9'].iloc[-1]
            
            # Momentum Check (keep existing)
            macd_line, signal_line, _ = talib.MACD(df['close'])
            histogram = macd_line - signal_line
            macd_bullish = (
                (macd_line.iloc[-1] > signal_line.iloc[-1]) and
                (histogram.iloc[-1] > histogram.iloc[-2]) and
                (macd_line.iloc[-1] > 0)
            )
            
            # Keep other existing checks
            current_price = df['close'].iloc[-1]
            atr = talib.ATR(df['high'], df['low'], df['close'], 14).iloc[-1]
            valid_atr = 1.0 <= (atr/current_price)*100 <= 3.0
            
            volume = df['volume'].iloc[-1]
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            strong_volume = volume > vol_ma * 1.15
            
            support_levels, _ = await self.calculate_support_resistance(df, symbol)
            near_support = any(abs(current_price - s)/s < 0.008 for s in support_levels[:3])
            
            # Updated logging
            logger.info(f"\n1H Analysis for {symbol}:")
            logger.info(f"Price > EMA9: {'✅' if price_above_ema9 else '❌'}")
            logger.info(f"MACD Bullish: {'✅' if macd_bullish else '❌'} | ATR: {(atr/current_price)*100:.1f}%")
            logger.info(f"Volume: {'✅' if strong_volume else '❌'} | Near Support: {'✅' if near_support else '❌'}")

            # Simplified return conditions
            return all([
                price_above_ema9,  # Basic trend alignment
                macd_bullish,
                valid_atr,
                strong_volume,
                near_support
            ])
            
        except Exception as e:
            logger.error(f"1H analysis error: {e}")
            return False

    async def _analyze_5m(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, dict]:
        try:
            # EMA Structure
            df['ema9'] = talib.EMA(df['close'], 9)
            df['ema21'] = talib.EMA(df['close'], 21)
            
            # More lenient trend requirements - either crossing or trending up
            current_above = df['ema9'].iloc[-1] > df['ema21'].iloc[-1]
            prev_below = df['ema9'].iloc[-2] <= df['ema21'].iloc[-2]
            ema_trending_up = df['ema9'].iloc[-1] > df['ema9'].iloc[-2]
            ema_bullish = (current_above and prev_below) or ema_trending_up
            
            # Relaxed MACD conditions
            macd_line, signal_line, _ = talib.MACD(
                df['close'],
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            histogram = macd_line - signal_line
            macd_bullish = (
                (macd_line.iloc[-1] > signal_line.iloc[-1]) or  # Either crossing
                (histogram.iloc[-1] > histogram.iloc[-2])        # Or gaining momentum
            )
            
            # Relaxed RSI conditions
            rsi = talib.RSI(df['close'], 14).iloc[-1]
            rsi_prev = talib.RSI(df['close'], 14).iloc[-2]
            rsi_bullish = (rsi > rsi_prev) and (rsi < 70)  # Just trending up and not overbought
            
            # Volume validation with lower threshold
            current_volume = df['volume'].iloc[-1]
            vol_ma = df['volume'].rolling(20, min_periods=1).mean().iloc[-1]
            volume_spike = current_volume > vol_ma * 1.1  # Reduced from 1.3
            
            # Wider ATR range
            current_price = df['close'].iloc[-1]
            atr = talib.ATR(df['high'], df['low'], df['close'], 14).iloc[-1]
            valid_atr = 0.01 <= (atr/current_price) <= 0.04  # Wider range 1-4%
            
            # Support check with wider threshold
            _, fifteen_m_supports = await self.calculate_support_resistance(df, symbol)
            near_support = any(abs(current_price - s)/s < 0.008 for s in fifteen_m_supports[:3])
            
            # More lenient candle pattern
            last_candle = df.iloc[-1]
            bullish_candle = last_candle['close'] > last_candle['open']  # Just needs to be green
            
            # Logging
            logger.info(f"\n5M Analysis for {symbol}:")
            logger.info(f"EMA Cross/Trend: {'✅' if ema_bullish else '❌'} | MACD: {'✅' if macd_bullish else '❌'}")
            logger.info(f"Volume: {current_volume/vol_ma:.1f}x MA | ATR: {(atr/current_price)*100:.1f}%")
            logger.info(f"Near Support: {'✅' if near_support else '❌'} | Bullish Candle: {'✅' if bullish_candle else '❌'}")
            logger.info(f"RSI Trend Up: {'✅' if rsi_bullish else '❌'} ({rsi_prev:.1f} → {rsi:.1f})")

            # Need fewer conditions to be true
            required_conditions = [
                ema_bullish or macd_bullish,  # Either EMA or MACD is bullish
                rsi_bullish,                  # RSI trending up
                volume_spike,                 # Decent volume
                valid_atr,                    # Reasonable volatility
                near_support or bullish_candle # Either near support or bullish candle
            ]
            
            analysis_result = {
                'ema_bullish': ema_bullish,
                'macd_bullish': macd_bullish,
                'rsi': rsi,
                'macd_histogram': histogram.iloc[-1],
                'signal_strength': sum(required_conditions) / len(required_conditions),
                'support_levels': fifteen_m_supports,
                'near_support': near_support,
                'volume_spike': volume_spike,
                'valid_atr': valid_atr
            }
            
            return sum(required_conditions) >= 4, analysis_result
            
        except Exception as e:
            logger.error(f"5M analysis error: {e}")
            return False, {}

    async def _validate_symbol(self, symbol):
        """Parallel technical validation for a single symbol"""
        try:
            # Add multi-timeframe data loading
            timeframes = ['1d', '4h', '1h', '5m']
            daily_df, four_hour_df, hourly_df, five_min_df = await asyncio.gather(
                *[self.get_historical_data(symbol, tf) for tf in timeframes]
            )
            
            # Original validation logic continues...
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                return False

            # Add proper sentiment analysis with data validation
            try:
                # First validate the DataFrame structure
                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in daily_df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in daily_df.columns]
                    logger.error(f"Missing columns in daily data: {missing}")
                    return False

                # Then convert values
                sentiment_params = {
                    'timeframe': '1d',
                    'price_data': {
                        'open': [float(x) for x in daily_df['open'].tolist()],
                        'high': [float(x) for x in daily_df['high'].tolist()],
                        'low': [float(x) for x in daily_df['low'].tolist()],
                        'close': [float(x) for x in daily_df['close'].tolist()]
                    }
                }
                sentiment_score = await self.market_analyzer.get_market_sentiment(
                    data=sentiment_params, 
                    symbol=symbol
                )
                logger.info(f"Sentiment Score: {sentiment_score:.2f}")
                
                if sentiment_score < 0.6:
                    logger.info(f"❌ Market sentiment not bullish enough: {sentiment_score:.2f}")
                    return False
                    
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                return False

            # Continue with existing validation logic...
            current_price = float(df['close'].iloc[-1])
            volume = float(df['volume'].iloc[-1])
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            
            logger.info(f"\n{'='*50}")
            logger.info(f"VALIDATING ENTRY - {symbol}")
            logger.info(f"Price: ${current_price:.4f}")
            
            # Calculate base technical indicators
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]

            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            

            # Daily timeframe conditions
            daily_ema9 = talib.EMA(daily_df['close'], timeperiod=9)
            daily_ema21 = talib.EMA(daily_df['close'], timeperiod=21)
            
            # Define daily EMA cross
            daily_ema_cross = (
                daily_ema9.iloc[-1] > daily_ema21.iloc[-1] and
                daily_ema9.iloc[-2] <= daily_ema21.iloc[-2]
            )

            # Then use the condition
            if not daily_ema_cross:
                logger.info("❌ No daily 9/21 EMA bullish cross")
                return False
            
            daily_rsi = talib.RSI(daily_df['close'], timeperiod=14).iloc[-1]
            daily_rsi_prev = talib.RSI(daily_df['close'], timeperiod=14).iloc[-2]
            daily_macd = talib.trend.MACD(daily_df['close'])

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
                four_hour_rsi = talib.RSI(four_hour_df['close'], timeperiod=14).iloc[-1]
                hourly_rsi = talib.RSI(hourly_df['close'], timeperiod=14).iloc[-1]
                

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
                
                if (atr/current_price) > 0.04:  # 4% threshold
                    logger.info(f"❌ ATR too high: {(atr/current_price)*100:.1f}%")
                    return False
                
                # New addition: Minimum ATR requirement (1.5-3% daily range)
                elif (atr/current_price) < 0.015:  # 1.5% threshold
                    logger.info(f"❌ ATR too low: {(atr/current_price)*100:.1f}% - Stagnant price action")
                    return False
                
                # 4. Check 5min EMAs and MACD
                ema9 = talib.EMA(five_min_df['close'], timeperiod=9).iloc[-1]
                ema21 = talib.EMA(five_min_df['close'], timeperiod=21).iloc[-1]
                ema200 = talib.EMA(five_min_df['close'], timeperiod=200).iloc[-1]
                

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
                ema9 = talib.EMA(df['close'], timeperiod=9).iloc[-1]
                ema21 = talib.EMA(df['close'], timeperiod=21).iloc[-1]
                ema200 = talib.EMA(df['close'], timeperiod=200).iloc[-1]
                

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
                macd = talib.MACD(df['close'])
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
                        orders = await self.exchange.fetch_closed_orders(symbol, limit=3)
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
        """Validate grid order size with exchange limits and available balance"""
        try:
            market = self.exchange.market(symbol)
            min_size = market['limits']['amount']['min']
            precision = market['precision']['amount']
            
            # Handle futures contracts
            if market['futures']:
                return round(max(size, min_size), precision)
                
            # Spot market validation
            base_currency = symbol.split('/')[0]
            balance = await self.exchange.fetch_balance()
            available = balance.get(base_currency, {}).get('free', 0)
            
            # Check against minimum size
            if size < min_size:
                logger.warning(f"Grid size {size} < min {min_size} for {symbol}")
                adjusted = round(min_size * 1.01, precision)
                logger.info(f"Adjusted to {adjusted}")
                return adjusted
                
            # Check available balance
            if available < size:
                logger.warning(f"Insufficient {base_currency} balance: {available} < {size}")
                return 0.0
            
            return round(size, precision)
            
        except Exception as e:
            logger.error(f"Size validation error: {e}")
            return 0.0

    async def verify_actual_positions(self) -> Dict[str, Dict]:
        """Verify actual positions vs open orders with precise FIFO tracking"""
        try:
            balance = await self.exchange.fetch_balance()
            markets = await self.exchange.fetch_markets()
            
            # Create symbol mapping
            symbol_map = {m['base'].replace('XBT','BTC').replace('XX','').replace('Z','').replace('.S',''): m['symbol'] 
                         for m in markets if m['quote'] == 'USD' and '/USD' in m['symbol']}

            actual_positions = {}  # Changed to dictionary
            for currency, amount in balance.get('total', {}).items():
                clean_currency = currency.replace('XBT','BTC').replace('XX','').replace('Z','').replace('.S','')
                
                if clean_currency not in symbol_map:
                    continue

                total_amount = float(amount)
                if total_amount > 0.000001:
                    try:
                        proper_symbol = symbol_map[clean_currency]
                        
                        # Get ALL trades for this symbol
                        since = self.exchange.milliseconds() - (86400 * 1000 * 30)
                        trades = await self.exchange.fetch_my_trades(proper_symbol, since=since)
                        
                        # FIFO processing with sell reconciliation
                        fifo_queue = []
                        for trade in sorted(trades, key=lambda x: x['timestamp']):
                            trade_size = float(trade['amount'])
                            
                            if trade['side'] == 'buy':
                                fifo_queue.append((trade_size, float(trade['price'])))
                            elif trade['side'] == 'sell':
                                remaining_sell = trade_size
                                while remaining_sell > 0 and fifo_queue:
                                    oldest_size, oldest_price = fifo_queue[0]
                                    if oldest_size > remaining_sell:
                                        fifo_queue[0] = (oldest_size - remaining_sell, oldest_price)
                                        remaining_sell = 0
                                    else:
                                        remaining_sell -= oldest_size
                                        fifo_queue.pop(0)
                        
                        # Calculate remaining position cost basis
                        total_cost = 0
                        accumulated = 0
                        for size, price in fifo_queue:
                            if accumulated + size > total_amount:
                                partial = total_amount - accumulated
                                total_cost += partial * price
                                accumulated = total_amount
                                break
                            else:
                                total_cost += size * price
                                accumulated += size
                        
                        entry_price = total_cost / accumulated if accumulated > 0 else await self.get_current_price(proper_symbol)
                        
                        position = {
                            'symbol': proper_symbol,
                            'info': {
                                'symbol': proper_symbol,
                                'size': total_amount,
                                'price': entry_price,
                                'side': 'long',
                                'is_spot': True,
                                'start_time': time.time()
                            }
                        }
                        
                        # Add debug logging here
                        logger.debug(f"Position type: {type(position)}")
                        logger.debug(f"Position data: {position}")
                        
                        # Update both tracking systems with complete fields
                        self.active_positions[proper_symbol] = position
                        self.positions[proper_symbol] = {
                            'entry_price': entry_price,
                            'position_size': total_amount,
                            'side': 'buy',
                            'start_time': time.time(),
                            'trailing_stop': None,
                            'tp1_triggered': False,
                            'tp2_triggered': False,
                            'breakeven_triggered': False,
                            'trailing_active': False,
                            'stop_buffer': 0.002,
                            'original_size': total_amount,
                            'is_spot': True,
                            'spot_balance': total_amount
                        }
                        
                        actual_positions[proper_symbol] = position  # Changed to dictionary assignment
                        logger.info(f"Verified position: {proper_symbol} - Entry: {entry_price:.6f}")
                        
                    except Exception as e:
                        logger.error(f"Position error {clean_currency}: {str(e)[:100]}")
                        continue

            # Add validation before returning
            if not actual_positions:
                logger.info("No active positions found")
                return {}  # Return empty dict instead of empty list
                
            # Add detailed debug logging here
            logger.debug(f"Final positions type: {type(actual_positions)}")
            for sym, pos in actual_positions.items():
                logger.debug(f"Position for {sym} - Type: {type(pos)}")
                logger.debug(f"Position data: {pos}")
            
            logger.debug(f"Verified positions: {actual_positions}")
            return actual_positions

        except Exception as e:
            logger.error(f"Position verification failed: {e}")
            return {}  # Return empty dict on error

    async def _check_resistance_levels(self, symbol, current_price):
        """Check if current price is near any resistance level"""
        try:
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                return False
            
            # Ensure DataFrame has required columns
            required_columns = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"DataFrame for {symbol} missing required columns: {df.columns}")
                return False

            support_levels, resistance_levels = await self.calculate_support_resistance(df, symbol)
            return any(abs(current_price - r) / r < 0.03 for r in resistance_levels)
        except Exception as e:
            logger.error(f"Error checking resistance levels for {symbol}: {e}")
            return False

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for Kraken API"""
        # Kraken requires symbols like 'XLM/USD' instead of 'XLMUSD'
        if '/' not in symbol:
            # Handle different symbol formats
            base = symbol[:3] if len(symbol) == 6 else symbol[:4]
            quote = symbol[3:] if len(symbol) == 6 else symbol[4:]
            return f"{base}/{quote}"
        return symbol.upper().replace(' ', '').replace('-', '')

    async def get_average_entry_price(self, symbol: str, position_size: float, mid_price: float) -> float:
        """Get exact entry price using FIFO accounting"""
        try:
            # Check if we already have accurate tracking
            if symbol in self.positions:
                tracked_price = self.positions[symbol].get('entry_price')
                if tracked_price and tracked_price > 0:
                    return tracked_price

            # Calculate from trade history with FIFO matching
            since = self.exchange.milliseconds() - (86400 * 1000 * 7)  # 1 week
            all_trades = await self.exchange.fetch_my_trades(symbol, since=since)
            
            if not all_trades:
                return mid_price

            # Process trades in chronological order with FIFO
            fifo_queue = []
            
            for trade in sorted(all_trades, key=lambda x: x['timestamp']):
                trade_size = float(trade['amount'])
                trade_price = float(trade['price'])
                
                if trade['side'] == 'buy':
                    fifo_queue.append((trade_size, trade_price))
                elif trade['side'] == 'sell':
                    # Remove from oldest buys first
                    while trade_size > 0 and fifo_queue:
                        oldest_size, oldest_price = fifo_queue[0]
                        if oldest_size > trade_size:
                            fifo_queue[0] = (oldest_size - trade_size, oldest_price)
                            trade_size = 0
                        else:
                            trade_size -= oldest_size
                            fifo_queue.pop(0)

            # Calculate remaining position cost basis
            total_cost = 0
            accumulated = 0
            
            for size, price in fifo_queue:
                if accumulated + size > position_size:
                    partial = position_size - accumulated
                    total_cost += partial * price
                    accumulated = position_size
                    break
                else:
                    total_cost += size * price
                    accumulated += size

            if accumulated > 0:
                exact_price = total_cost / accumulated
                logger.info(f"Exact FIFO entry price: {exact_price:.6f} for {accumulated} {symbol}")
                return exact_price

            return mid_price

        except Exception as e:
            logger.error(f"FIFO price error {symbol}: {e}")
            return mid_price

    async def _price_position_checks(self, current_price: float, 
                                    support_levels: List[float], 
                                    resistance_levels: List[float]) -> bool:
        """Validate price position relative to support/resistance levels"""
        try:
            # Resistance check with buffer
            above_resistance = False
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                resistance_buffer = nearest_resistance * 0.02  # 2% buffer
                above_resistance = current_price > (nearest_resistance + resistance_buffer)
                logger.info(f"Resistance Check: {current_price:.4f} vs {nearest_resistance:.4f} (+2% buffer)")

            # Support zone check
            near_support = False
            if support_levels:
                # Check against first 3 support levels with 1.5% tolerance
                relevant_supports = sorted(support_levels)[:3]
                near_support = any(abs(current_price - s)/s < 0.015 for s in relevant_supports)
                logger.info(f"Support Check: Near {'✅' if near_support else '❌'}")

            logger.info(f"Price Position: {'✅ Above Resistance' if above_resistance else '⚠️ Neutral' if near_support else '❌ Below Support'}")
            
            return above_resistance or near_support
            
        except Exception as e:
            logger.error(f"Price position check error: {e}")
            return False

    async def detect_bear_bounce(self, df: pd.DataFrame) -> bool:
        """Identify potential counter-trend rallies in bear markets"""
        try:
            # Require 3+ red candles followed by green
            last_3_closes = df['close'].iloc[-3:].values
            direction_changes = np.diff(np.sign(np.diff(last_3_closes)))
            bounce_signal = (direction_changes[-1] > 0) and (df['volume'].iloc[-1] > df['volume'].iloc[-2]*1.5)
            
            # Add RSI confirmation
            rsi = ta.RSI(df['close'], 14).iloc[-1]
            return bounce_signal and (rsi < 35)
        except Exception as e:
            logger.error(f"Bounce detection error: {e}")
            return False
    
    def is_weekend(self):
        """Check if current time is weekend in Kraken's primary markets (UTC-7 to UTC+3)"""
        try:
            # Get current time in UTC
            utc_now = datetime.now(timezone.utc)
            
            # Check major crypto market timezones
            market_tz = [
                pytz.timezone('America/Los_Angeles'),  # UTC-7/-8
                pytz.timezone('Europe/London'),        # UTC+0/+1
                pytz.timezone('Asia/Hong_Kong')        # UTC+8
            ]
            
            # Check if any major market is in weekend
            for tz in market_tz:
                local_time = utc_now.astimezone(tz)
                if local_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Weekend check error: {e}")
            return False
    
    async def refresh_grid_levels(self, symbol: str) -> None:
        """Recalculate grid levels after trades"""
        try:
            logger.info(f"🔄 Refreshing grid levels for {symbol}")
            
            # Get updated market data
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                logger.warning(f"❌ No data for {symbol} refresh")
                return
                
            # Recalculate support/resistance
            support_levels, resistance_levels = await self.calculate_support_resistance(df, symbol)
            
            # Update grid prices
            current_price = df['close'].iloc[-1]
            grid_prices = await self.calculate_hybrid_grids(
                symbol, df, support_levels, resistance_levels
            )
            logger.info(f"Current price: {current_price} | Grid levels: {len(grid_prices)}")
            
        except Exception as e:
            logger.error(f"Grid refresh error: {e}")

    async def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate recent volatility using multiple methods"""
        try:
            # Calculate different volatility metrics
            
            # 1. Traditional volatility (standard deviation of returns)
            returns = df['close'].pct_change()
            trad_vol = returns.std() * np.sqrt(252) * 100  # Annualized and converted to percentage
            
            # 2. True Range based volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window).mean()
            atr_vol = (atr.iloc[-1] / df['close'].iloc[-1]) * 100  # Convert to percentage
            
            # 3. Parkinson volatility (uses high-low range)
            hl_vol = np.sqrt(1/(4*np.log(2)) * 
                            ((np.log(df['high']/df['low'])**2)
                             .rolling(window)
                             .mean())) * np.sqrt(252) * 100
            
            # Combine volatility metrics with weights
            combined_vol = (
                0.4 * trad_vol +      # Traditional volatility (40% weight)
                0.4 * atr_vol +       # ATR-based volatility (40% weight)
                0.2 * hl_vol.iloc[-1] # Parkinson volatility (20% weight)
            )
            
            logger.info(f"\nVolatility Analysis:")
            logger.info(f"Traditional Vol: {trad_vol:.2f}%")
            logger.info(f"ATR-based Vol: {atr_vol:.2f}%")
            logger.info(f"Parkinson Vol: {hl_vol.iloc[-1]:.2f}%")
            logger.info(f"Combined Vol: {combined_vol:.2f}%")
            
            return combined_vol

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 2.0  # Return a default moderate volatility

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        try:
            atr = talib.ATR(high, low, close, period)
            if pd.isna(atr.iloc[-1]) or atr.iloc[-1] <= 0:
                # Fallback to recent price range
                recent_range = high.rolling(period).max() - low.rolling(period).min()
                return recent_range.iloc[-1] / close.iloc[-1]
            return atr.iloc[-1]
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 0.01  # 1% default

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
        strategy.is_spot = True  # Force spot mode
        
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
