"""
🤖 QUOTEX AUTOMATION TRADING BOT 🤖
===================================

Advanced Binary Options Trading Bot with Intelligent Analysis
- Automatic login and session management
- Asset discovery and selection (>85% payout)
- Intelligent price action and candle psychology analysis
- Automated trading with risk management
- Real-time monitoring and performance tracking
- Short expiry trading (60-120 seconds)

Author: AI Trading Systems
Version: 2.0.0
License: MIT
"""

import os
import sys
import time
import json
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import Quotex API and Brain Model for neural network predictions
from quotexapi.stable_api import Quotex
from quotexapi.config import email, password
from brain_model import get_brain_prediction

# Configure logging with UTF-8 encoding for Windows compatibility
import sys
import io

# Set UTF-8 encoding for stdout to handle emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingStatistics:
    """📊 Trading performance statistics tracker"""

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics"""
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.start_balance = 0.0
        self.current_balance = 0.0
        self.start_time = datetime.now()
        self.trade_history = []
        self.asset_performance = {}

    def add_trade(self, asset: str, direction: str, amount: float,
                  result: str, profit: float, confidence: float):
        """Add a completed trade to statistics"""
        trade = {
            'timestamp': datetime.now(),
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'result': result,
            'profit': profit,
            'confidence': confidence
        }

        self.trade_history.append(trade)
        self.total_trades += 1

        if result == 'win':
            self.winning_trades += 1
            self.total_profit += profit
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit)

        # Track asset performance
        if asset not in self.asset_performance:
            self.asset_performance[asset] = {'wins': 0, 'losses': 0, 'profit': 0.0}

        if result == 'win':
            self.asset_performance[asset]['wins'] += 1
        else:
            self.asset_performance[asset]['losses'] += 1
        self.asset_performance[asset]['profit'] += profit

    def get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def get_net_profit(self) -> float:
        """Calculate net profit/loss"""
        return self.total_profit - self.total_loss

    def get_roi(self) -> float:
        """Calculate return on investment percentage"""
        if self.start_balance == 0:
            return 0.0
        return ((self.current_balance - self.start_balance) / self.start_balance) * 100

    def print_summary(self):
        """Print comprehensive trading summary"""
        runtime = datetime.now() - self.start_time

        logger.info("=" * 60)
        logger.info("📊 TRADING PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️  Runtime: {runtime}")
        logger.info(f"💰 Start Balance: ${self.start_balance:.2f}")
        logger.info(f"💰 Current Balance: ${self.current_balance:.2f}")
        logger.info(f"📈 Net Profit: ${self.get_net_profit():.2f}")
        logger.info(f"📊 ROI: {self.get_roi():.2f}%")
        logger.info(f"🎯 Total Trades: {self.total_trades}")
        logger.info(f"✅ Winning Trades: {self.winning_trades}")
        logger.info(f"❌ Losing Trades: {self.losing_trades}")
        logger.info(f"🏆 Win Rate: {self.get_win_rate():.1f}%")

        if self.asset_performance:
            logger.info("\n📊 ASSET PERFORMANCE:")
            for asset, perf in self.asset_performance.items():
                total = perf['wins'] + perf['losses']
                win_rate = (perf['wins'] / total * 100) if total > 0 else 0
                logger.info(f"   {asset}: {perf['wins']}W/{perf['losses']}L "
                          f"({win_rate:.1f}%) Profit: ${perf['profit']:.2f}")


class QuotexTradingBot:
    """🤖 Advanced Quotex Trading Bot with Intelligent Analysis"""

    def __init__(self,
                 confidence_threshold: float = 0.50,
                 min_payout: float = 85.0,
                 trade_amount: float = 10.0,
                 max_trades_per_hour: int = 20,
                 demo_mode: bool = True,
                 use_martingale: bool = True,
                 martingale_multiplier: float = 2.2,
                 max_martingale_steps: int = 5,
                 enable_advanced_analysis: bool = False,
                 enhanced_expiry_time: int = 60,
                 enhanced_mode: bool = False,
                 analysis_mode: str = "Standard",
                 **kwargs):
        """
        Initialize the enhanced trading bot

        Args:
            confidence_threshold: Minimum confidence for trading (0.30 = 30%)
            min_payout: Minimum payout percentage required (85.0 = 85%)
            trade_amount: Base amount to trade per position
            max_trades_per_hour: Maximum trades per hour limit
            demo_mode: Use demo account (True) or live account (False)
            use_martingale: Enable Martingale strategy after losses
            martingale_multiplier: Multiplier for trade amount after loss (2.2 = 2.2x)
            max_martingale_steps: Maximum consecutive Martingale steps allowed
            enable_advanced_analysis: Enable advanced analysis (slower but more accurate)
            enhanced_expiry_time: Enhanced expiry time (60 or 90 seconds)
            enhanced_mode: Enable enhanced features and analysis
            analysis_mode: Analysis mode (Standard, Conservative, Aggressive)
        """
        self.confidence_threshold = confidence_threshold
        self.min_payout = min_payout
        self.base_trade_amount = trade_amount  # Store original amount
        self.trade_amount = trade_amount       # Current trade amount (can change with Martingale)
        self.max_trades_per_hour = max_trades_per_hour
        self.demo_mode = demo_mode

        # 🚀 ENHANCED FEATURES
        self.enhanced_expiry_time = enhanced_expiry_time
        self.enhanced_mode = enhanced_mode
        self.analysis_mode = analysis_mode

        # Martingale settings
        self.use_martingale = use_martingale
        self.martingale_multiplier = martingale_multiplier
        self.max_martingale_steps = max_martingale_steps
        self.current_martingale_step = 0
        self.consecutive_losses = 0
        self.pending_martingale = False  # Flag for pending Martingale trade

        # Initialize components
        self.client = None

        # 🧠 BRAIN MODEL SYSTEM: Simple Neural Network Predictions
        # No complex analysis - just the trained brain model

        # 🧠 SIMPLE PERFORMANCE TRACKING
        self.performance_tracker = {
            'recent_trades': [],
            'win_rate_threshold': 0.70,          # Require 70%+ win rate
            'confidence_adjustment': 0.0,        # Dynamic confidence adjustment
            'model_performance': {
                'brain_model': {'wins': 0, 'total': 0}
            }
        }

        # 🔗 CONNECTION STABILITY CONFIGURATION
        self.connection_config = {
            'max_reconnect_attempts': 5,
            'reconnect_delay': 3,  # seconds
            'connection_timeout': 10,  # seconds
            'heartbeat_interval': 30,  # seconds
            'connection_check_interval': 60,  # seconds
            'auto_reconnect_enabled': True,
            'connection_health_threshold': 3,  # failed checks before reconnect
        }

        # Connection monitoring
        self.connection_health = {
            'last_successful_connection': None,
            'failed_connection_attempts': 0,
            'last_heartbeat': None,
            'connection_stable': False,
            'reconnect_in_progress': False,
        }

        # 🔥 ADVANCED FILTERS CONFIGURATION
        self.advanced_filters = {
            'session_filter_enabled': True,  # Enable session-based filtering
            'losing_streak_protection': True,  # Enable losing streak protection
            'confidence_boost_enabled': True,  # Enable confidence boosting
            'volatile_hours': [8, 9, 13, 14, 15, 16],  # Major session opens/closes
            'min_confidence_volatile': 0.80,  # Minimum confidence during volatile hours
            'min_confidence_losing_streak': 0.75,  # Minimum confidence during losing streaks
        }

        self.stats = TradingStatistics()
        self.running = False
        self.selected_asset = None
        self.last_trade_time = None
        self.trades_this_hour = 0
        self.hour_start = datetime.now().replace(minute=0, second=0, microsecond=0)



        logger.info("🤖 BRAIN MODEL Quotex Trading Bot Initialized")
        logger.info(f"   🎯 Confidence Threshold: {confidence_threshold:.1%}")
        logger.info(f"   💰 Min Payout Required: {min_payout:.1f}%")
        logger.info(f"   💵 Base Trade Amount: ${trade_amount:.2f}")
        logger.info(f"   📊 Account Mode: {'DEMO' if demo_mode else 'LIVE'}")
        logger.info(f"   🚀 Enhanced Mode: {'ENABLED' if enhanced_mode else 'DISABLED'}")
        logger.info(f"   ⏱️ Enhanced Expiry: {enhanced_expiry_time}s")
        logger.info(f"   📈 Analysis Mode: {analysis_mode}")
        logger.info(f"   🧠 Brain Model: Neural Network with Price Action Training")
        if use_martingale:
            logger.info(f"   🎲 Martingale: ENABLED (x{martingale_multiplier} after loss, max {max_martingale_steps} steps)")
            logger.info(f"   🎯 Auto-Execute: Martingale trades will be placed automatically after losses")
        else:
            logger.info(f"   🎲 Martingale: DISABLED")

    async def connect_and_login(self) -> bool:
        """🔌 Connect to Quotex API and handle login"""
        try:
            logger.info("Connecting to Quotex API...")

            # Initialize Quotex client with better settings for Cloudflare bypass
            self.client = Quotex(
                email=email,
                password=password,
                lang="en",  # English for better compatibility
                user_data_dir="browser"
            )

            # Enable debug mode to see connection details
            self.client.debug_ws_enable = False

            # Set account mode
            if self.demo_mode:
                self.client.set_account_mode("PRACTICE")
            else:
                self.client.set_account_mode("REAL")

            # Connect with enhanced retry logic
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Connection attempt {attempt + 1}/{max_attempts}...")

                    # Add delay between attempts to avoid rate limiting
                    if attempt > 0:
                        wait_time = min(30, 5 * attempt)  # Progressive backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)

                    check, reason = await self.client.connect()
                    if check:
                        logger.info("✅ Successfully connected to Quotex API")

                        # Update connection health
                        self.connection_health['last_successful_connection'] = datetime.now()
                        self.connection_health['failed_connection_attempts'] = 0
                        self.connection_health['connection_stable'] = True
                        self.connection_health['reconnect_in_progress'] = False

                        # Set account mode after connection
                        account_mode = "DEMO" if self.demo_mode else "LIVE"
                        self.client.change_account("PRACTICE" if self.demo_mode else "REAL")
                        logger.info(f"Switched to {account_mode} account")

                        # Get initial balance
                        balance = await self.client.get_balance()
                        self.stats.start_balance = balance
                        self.stats.current_balance = balance
                        logger.info(f"Current Balance: ${balance:.2f}")

                        # Start connection monitoring
                        asyncio.create_task(self._monitor_connection())

                        return True
                    else:
                        error_msg = str(reason)
                        if "403" in error_msg or "Forbidden" in error_msg:
                            logger.error("Connection blocked by Cloudflare protection")
                            logger.info("This may be due to:")
                            logger.info("1. Too many connection attempts")
                            logger.info("2. IP-based rate limiting")
                            logger.info("3. Bot detection measures")
                            logger.info("Try again later or use a VPN")
                        else:
                            logger.error(f"Connection failed: {reason}")

                        if attempt < max_attempts - 1:
                            logger.info(f"Retrying connection... ({attempt + 1}/{max_attempts})")

                except Exception as e:
                    error_msg = str(e)
                    if "403" in error_msg or "Forbidden" in error_msg:
                        logger.error("Connection blocked by Cloudflare protection")
                        logger.info("Suggestion: Wait 10-15 minutes before trying again")
                    else:
                        logger.error(f"Connection error: {error_msg}")

                    if attempt < max_attempts - 1:
                        logger.info(f"Will retry in a moment... ({attempt + 1}/{max_attempts})")

            logger.error("Failed to connect to Quotex after all attempts")
            logger.info("Possible solutions:")
            logger.info("1. Check your internet connection")
            logger.info("2. Verify your login credentials in settings/config.ini")
            logger.info("3. Try again later (Cloudflare may be blocking)")
            logger.info("4. Use a VPN if available")
            return False

        except Exception as e:
            logger.error(f"Critical connection error: {str(e)}")
            self.connection_health['connection_stable'] = False
            return False

    async def _monitor_connection(self):
        """🔗 Monitor websocket connection health and handle reconnections"""
        try:
            logger.info("🔗 Starting connection monitoring...")

            while self.running:
                try:
                    # Check connection health
                    if hasattr(self.client, 'check_connect'):
                        is_connected = self.client.check_connect
                    else:
                        # Fallback: try to get balance as a connection test
                        try:
                            await asyncio.wait_for(self.client.get_balance(), timeout=5.0)
                            is_connected = True
                        except:
                            is_connected = False

                    current_time = datetime.now()

                    if is_connected:
                        # Connection is healthy
                        self.connection_health['last_heartbeat'] = current_time
                        self.connection_health['connection_stable'] = True
                        self.connection_health['failed_connection_attempts'] = 0

                        # Reset reconnect flag if it was set
                        if self.connection_health['reconnect_in_progress']:
                            logger.info("🔗 Connection restored successfully")
                            self.connection_health['reconnect_in_progress'] = False
                    else:
                        # Connection is unhealthy
                        self.connection_health['failed_connection_attempts'] += 1
                        self.connection_health['connection_stable'] = False

                        logger.warning(f"🔗 Connection check failed ({self.connection_health['failed_connection_attempts']}/{self.connection_config['connection_health_threshold']})")

                        # Trigger reconnection if threshold reached
                        if (self.connection_health['failed_connection_attempts'] >=
                            self.connection_config['connection_health_threshold'] and
                            not self.connection_health['reconnect_in_progress']):

                            logger.warning("🔗 Connection health threshold reached - initiating reconnection")
                            asyncio.create_task(self._handle_reconnection())

                    # Wait before next check
                    await asyncio.sleep(self.connection_config['connection_check_interval'])

                except Exception as e:
                    logger.error(f"🔗 Connection monitoring error: {str(e)}")
                    await asyncio.sleep(30)  # Wait longer on error

        except Exception as e:
            logger.error(f"🔗 Connection monitor crashed: {str(e)}")

    async def _handle_reconnection(self):
        """🔄 Handle automatic reconnection with exponential backoff"""
        if self.connection_health['reconnect_in_progress']:
            return  # Already reconnecting

        self.connection_health['reconnect_in_progress'] = True

        try:
            logger.info("🔄 Starting automatic reconnection process...")

            max_attempts = self.connection_config['max_reconnect_attempts']
            base_delay = self.connection_config['reconnect_delay']

            for attempt in range(max_attempts):
                try:
                    logger.info(f"🔄 Reconnection attempt {attempt + 1}/{max_attempts}")

                    # Progressive delay with exponential backoff
                    if attempt > 0:
                        delay = min(base_delay * (2 ** attempt), 60)  # Cap at 60 seconds
                        logger.info(f"⏳ Waiting {delay} seconds before reconnection...")
                        await asyncio.sleep(delay)

                    # Close existing connection
                    try:
                        if hasattr(self.client, 'close'):
                            self.client.close()
                        await asyncio.sleep(2)  # Give time for cleanup
                    except:
                        pass

                    # Attempt reconnection
                    check, reason = await asyncio.wait_for(
                        self.client.connect(),
                        timeout=self.connection_config['connection_timeout']
                    )

                    if check:
                        logger.info("✅ Reconnection successful!")

                        # Update connection health
                        self.connection_health['last_successful_connection'] = datetime.now()
                        self.connection_health['failed_connection_attempts'] = 0
                        self.connection_health['connection_stable'] = True
                        self.connection_health['reconnect_in_progress'] = False

                        # Restore account mode
                        account_mode = "PRACTICE" if self.demo_mode else "REAL"
                        self.client.change_account(account_mode)

                        return True
                    else:
                        logger.warning(f"🔄 Reconnection attempt {attempt + 1} failed: {reason}")

                except asyncio.TimeoutError:
                    logger.warning(f"🔄 Reconnection attempt {attempt + 1} timed out")
                except Exception as e:
                    logger.warning(f"🔄 Reconnection attempt {attempt + 1} error: {str(e)}")

            # All attempts failed
            logger.error("❌ All reconnection attempts failed")
            self.connection_health['reconnect_in_progress'] = False
            self.connection_health['connection_stable'] = False

            return False

        except Exception as e:
            logger.error(f"❌ Reconnection handler error: {str(e)}")
            self.connection_health['reconnect_in_progress'] = False
            return False

    def is_connection_stable(self) -> bool:
        """Check if connection is currently stable"""
        return (self.connection_health['connection_stable'] and
                not self.connection_health['reconnect_in_progress'])

    async def discover_tradeable_assets(self) -> List[Tuple[str, float]]:
        """🔍 Discover all tradeable assets with high payouts and data availability"""
        try:
            logger.info("🔍 Discovering all tradeable assets...")

            # Get all available assets
            instruments = await self.client.get_instruments()
            if not instruments:
                logger.error("❌ No instruments available")
                return []

            # Get payment data for all assets
            payment_data = self.client.get_payment()

            # Assets that commonly have data issues - we'll deprioritize these
            problematic_assets = {
                'PEPUSD_otc', 'BNBUSD_otc', 'USDINR_otc', 'USDBDT_otc', 'USDMXN_otc',
                'FLOUSD_otc', 'SHIUSD_otc', 'USDPKR_otc', 'USDDZD_otc', 'USDNGN_otc',
                'NZDCAD_otc', 'CADCHF_otc', 'UKBrent_otc', 'PFE_otc', 'XAUUSD_otc',
                'EURNZD_otc', 'XAGUSD_otc', 'MCD_otc', 'AXP_otc', 'INTC_otc'
            }

            # Preferred assets that usually have good data
            preferred_assets = {
                'EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc', 'AUDUSD_otc', 'USDCAD_otc',
                'EURJPY_otc', 'GBPJPY_otc', 'EURGBP_otc', 'AUDCAD_otc', 'AUDCHF_otc',
                'USDCHF_otc', 'NZDUSD_otc', 'EURAUD_otc', 'GBPAUD_otc', 'USDPHP_otc',
                'MSFT_otc', 'AAPL_otc', 'GOOGL_otc', 'TSLA_otc'
            }

            preferred_list = []
            regular_list = []

            for instrument in instruments:
                _, asset_name, asset_display, *rest = instrument

                # Check if asset is open
                if len(rest) >= 12 and rest[11]:  # Asset is open
                    # Get payout information
                    if asset_display in payment_data:
                        payout_info = payment_data[asset_display]
                        payout_1m = payout_info.get('profit', {}).get('1M', 0)

                        if payout_1m >= self.min_payout:
                            asset_tuple = (asset_name, payout_1m)

                            # Skip problematic assets unless they're the only option
                            if asset_name in problematic_assets:
                                continue
                            elif asset_name in preferred_assets:
                                preferred_list.append(asset_tuple)
                                logger.info(f"📊 {asset_name}: Payout {payout_1m:.1f}% - ✅ Open (Preferred)")
                            else:
                                regular_list.append(asset_tuple)
                                logger.info(f"📊 {asset_name}: Payout {payout_1m:.1f}% - ✅ Open")

            # Sort by payout (highest first) within each category
            preferred_list.sort(key=lambda x: x[1], reverse=True)
            regular_list.sort(key=lambda x: x[1], reverse=True)

            # Combine lists with preferred assets first
            tradeable_assets = preferred_list + regular_list

            if tradeable_assets:
                logger.info(f"🎯 Found {len(tradeable_assets)} tradeable assets ({len(preferred_list)} preferred, {len(regular_list)} regular)")
                return tradeable_assets
            else:
                logger.warning("⚠️ No assets found meeting minimum payout requirements")
                return []

        except Exception as e:
            logger.error(f"❌ Error discovering assets: {str(e)}")
            # Return fallback assets if discovery fails
            logger.info("🔄 Using fallback asset list...")
            return [
                ("EURUSD", 80.0), ("GBPUSD", 80.0), ("USDJPY", 80.0), ("AUDUSD", 80.0),
                ("USDCAD", 80.0), ("USDCHF", 80.0), ("NZDUSD", 80.0), ("EURGBP", 80.0),
                ("BTCUSD", 85.0), ("ETHUSD", 85.0), ("XRPUSD", 85.0)
            ]

    async def fetch_candle_data(self, asset: str, count: int = 50, verbose: bool = True) -> Optional[pd.DataFrame]:
        """📊 Fetch historical candle data for analysis with improved error handling"""
        try:
            # Reduced logging for speed
            if verbose and count > 100:  # Only log for large requests
                logger.info(f"📊 Fetching {count} candles for {asset}...")

            # Skip asset availability check for now as it's causing issues
            # We'll rely on the candle fetching methods to determine availability

            # Try multiple methods to fetch candles
            candles = None

            # Method 1: Standard get_candles with timeout
            try:
                end_time = time.time()
                offset = count * 60  # 60 seconds per candle
                period = 60  # 1-minute candles

                # Add timeout to prevent hanging
                candles = await asyncio.wait_for(
                    self.client.get_candles(asset, end_time, offset, period),
                    timeout=5.0
                )

                if candles and len(candles) > 0:
                    if verbose:
                        logger.debug(f"Method 1 success for {asset}: {len(candles)} candles")
                else:
                    candles = None

            except asyncio.TimeoutError:
                if verbose:
                    logger.debug(f"Method 1 timeout for {asset}")
                candles = None
            except Exception as e:
                if verbose:
                    logger.debug(f"Method 1 failed for {asset}: {str(e)}")
                candles = None

            # Method 2: Try get_candle_v2 if first method fails
            if not candles:
                try:
                    candles = await asyncio.wait_for(
                        self.client.get_candle_v2(asset, 60),
                        timeout=5.0
                    )

                    if candles and len(candles) > 0:
                        if verbose:
                            logger.debug(f"Method 2 success for {asset}: {len(candles)} candles")
                    else:
                        candles = None

                except asyncio.TimeoutError:
                    if verbose:
                        logger.debug(f"Method 2 timeout for {asset}")
                    candles = None
                except Exception as e:
                    if verbose:
                        logger.debug(f"Method 2 failed for {asset}: {str(e)}")
                    candles = None

            # Method 3: Try with different time parameters
            if not candles:
                try:
                    end_time = time.time() - 300  # 5 minutes ago
                    offset = min(count * 60, 3600)  # Cap at 1 hour
                    period = 60

                    candles = await asyncio.wait_for(
                        self.client.get_candles(asset, end_time, offset, period),
                        timeout=5.0
                    )

                    if candles and len(candles) > 0:
                        if verbose:
                            logger.debug(f"Method 3 success for {asset}: {len(candles)} candles")
                    else:
                        candles = None

                except asyncio.TimeoutError:
                    if verbose:
                        logger.debug(f"Method 3 timeout for {asset}")
                    candles = None
                except Exception as e:
                    if verbose:
                        logger.debug(f"Method 3 failed for {asset}: {str(e)}")
                    candles = None

            # Method 4: Try with asset without _otc suffix (KEEP THIS - IT WORKS!)
            if not candles and "_otc" in asset:
                try:
                    clean_asset = asset.replace("_otc", "")
                    candles = await asyncio.wait_for(
                        self.client.get_candles(clean_asset, time.time(), count * 60, 60),
                        timeout=5.0
                    )
                    if candles and len(candles) > 0:
                        if verbose:
                            logger.debug(f"Method 4 success for {clean_asset}: {len(candles)} candles")
                    else:
                        candles = None
                except asyncio.TimeoutError:
                    if verbose:
                        logger.debug(f"Method 4 timeout for {clean_asset}")
                    candles = None
                except Exception as e:
                    if verbose:
                        logger.debug(f"Method 4 failed for {clean_asset}: {str(e)}")
                    candles = None

            if not candles:
                if verbose:
                    # Check if connection is still alive and handle reconnection
                    try:
                        if not self.is_connection_stable():
                            logger.warning(f"❌ {asset}: Connection unstable - data fetch failed")

                            # Wait for reconnection if in progress
                            if self.connection_health['reconnect_in_progress']:
                                logger.info(f"⏳ {asset}: Waiting for reconnection to complete...")
                                for _ in range(10):  # Wait up to 10 seconds
                                    await asyncio.sleep(1)
                                    if self.is_connection_stable():
                                        logger.info(f"✅ {asset}: Connection restored, retrying data fetch...")
                                        # Could retry fetching here, but for now just continue
                                        break
                            else:
                                # Trigger reconnection if not already in progress
                                logger.info(f"🔄 {asset}: Triggering connection check...")
                                asyncio.create_task(self._handle_reconnection())
                        else:
                            logger.info(f"❌ {asset}: No candle data available from any method (connection stable)")
                    except Exception as e:
                        logger.info(f"❌ {asset}: No candle data available - {str(e)}")
                return None

            # Convert to DataFrame with improved data handling
            df_data = []
            for candle in candles:
                try:
                    # Handle different candle data formats
                    if isinstance(candle, dict):
                        # Format 1: Direct dictionary with OHLC keys
                        if all(k in candle for k in ['open', 'high', 'low', 'close']):
                            df_data.append({
                                'open': float(candle['open']),
                                'high': float(candle['high']),
                                'low': float(candle['low']),
                                'close': float(candle['close']),
                                'time': candle.get('time', time.time())
                            })
                        # Format 2: Dictionary with different key names
                        elif all(k in candle for k in ['o', 'h', 'l', 'c']):
                            df_data.append({
                                'open': float(candle['o']),
                                'high': float(candle['h']),
                                'low': float(candle['l']),
                                'close': float(candle['c']),
                                'time': candle.get('t', candle.get('time', time.time()))
                            })
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 4:
                        # Format 3: Array format [time, open, close, high, low] or similar
                        if len(candle) >= 5:
                            df_data.append({
                                'open': float(candle[1]),
                                'high': float(candle[3]),
                                'low': float(candle[4]),
                                'close': float(candle[2]),
                                'time': candle[0]
                            })
                        else:
                            # Assume [open, high, low, close]
                            df_data.append({
                                'open': float(candle[0]),
                                'high': float(candle[1]),
                                'low': float(candle[2]),
                                'close': float(candle[3]),
                                'time': time.time()
                            })
                except (ValueError, TypeError, IndexError) as e:
                    if verbose:
                        logger.debug(f"Skipping invalid candle data: {candle} - {str(e)}")
                    continue

            if len(df_data) < 15:  # Reduced from 30 for faster analysis
                if verbose:
                    logger.info(f"❌ {asset}: Insufficient candle data: {len(df_data)} candles (need at least 15)")
                return None

            # Create DataFrame and validate data
            df = pd.DataFrame(df_data)

            # Basic data validation
            if df.empty or df.isnull().all().any():
                if verbose:
                    logger.info(f"❌ {asset}: Invalid candle data (contains null values)")
                return None

            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any rows with NaN values
            df = df.dropna()

            if len(df) < 15:  # Reduced from 30 for faster analysis
                if verbose:
                    logger.info(f"❌ {asset}: Insufficient valid candle data after cleaning: {len(df)} candles")
                return None

            # Sort by time if time column exists
            if 'time' in df.columns:
                df = df.sort_values('time').reset_index(drop=True)

            # Reduced success logging for speed
            if verbose and len(df) < 50:  # Only log if we got fewer candles than expected
                logger.info(f"⚠️ {asset}: Only fetched {len(df)} candles")

            # Minimal delay to prevent API overload
            await asyncio.sleep(0.05)
            return df

        except Exception as e:
            if verbose:
                logger.info(f"❌ {asset}: Error fetching candle data - {str(e)}")
            return None

    async def analyze_and_predict(self, data: pd.DataFrame) -> Tuple[str, float, int]:
        """🧠 BRAIN MODEL PREDICTION: Simple neural network analysis for fast and accurate trading"""
        try:
            # 🧠 BRAIN MODEL: Use trained neural network for prediction
            brain_result = get_brain_prediction(data)

            signal = brain_result.get('signal', 'NEUTRAL')
            confidence = brain_result.get('confidence', 0.0)
            optimal_expiry = brain_result.get('details', {}).get('optimal_expiry', 60)

            # 📊 DETAILED LOGGING FOR TRANSPARENCY
            if signal != "NEUTRAL":
                logger.info(f"🧠 BRAIN MODEL PREDICTION:")
                logger.info(f"   🎯 Signal: {signal} ({confidence:.1%})")
                logger.info(f"   ⏱️ Optimal Expiry: {optimal_expiry}s")
                logger.info(f"   🔬 Model Type: {brain_result.get('details', {}).get('model_type', 'brain_neural_network')}")

                # Show additional details if available
                details = brain_result.get('details', {})
                if 'data_points_used' in details:
                    logger.info(f"   📊 Data Points: {details['data_points_used']}")
                if 'sequence_length' in details:
                    logger.info(f"   📏 Sequence Length: {details['sequence_length']}")

            return signal, confidence, optimal_expiry

        except Exception as e:
            logger.error(f"❌ Brain model analysis failed: {str(e)}")
            return "NEUTRAL", 0.0, 60



    def should_trade(self, signal: str, confidence: float) -> bool:
        """🎯 ADVANCED RISK MANAGEMENT - Determine if we should execute a trade"""

        # 🔥 ENHANCED CONFIDENCE FILTERING
        # Dynamic confidence threshold based on recent performance
        dynamic_threshold = self._calculate_dynamic_threshold()

        if confidence < dynamic_threshold:
            logger.info(f"⚠️ Confidence {confidence:.1%} below dynamic threshold {dynamic_threshold:.1%}")
            return False

        # Check if signal is actionable
        if signal == "NEUTRAL":
            logger.info("⚠️ Neutral signal - no trade")
            return False

        # 🎯 ADVANCED ACCURACY FILTERS
        if not self._passes_advanced_filters(confidence):
            return False

        # Check trade frequency limits
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)

        if current_hour > self.hour_start:
            # New hour, reset counter
            self.hour_start = current_hour
            self.trades_this_hour = 0

        if self.trades_this_hour >= self.max_trades_per_hour:
            logger.info(f"⚠️ Max trades per hour reached ({self.max_trades_per_hour})")
            return False

        # Check minimum time between trades (30 seconds)
        if self.last_trade_time:
            time_since_last = (current_time - self.last_trade_time).total_seconds()
            if time_since_last < 30:
                logger.info(f"⚠️ Too soon since last trade ({time_since_last:.1f}s)")
                return False

        # Check balance
        if self.stats.current_balance < self.trade_amount:
            logger.info(f"⚠️ Insufficient balance: ${self.stats.current_balance:.2f} < ${self.trade_amount:.2f}")
            return False

        return True

    def _calculate_dynamic_threshold(self) -> float:
        """🧠 Calculate dynamic confidence threshold based on recent performance"""
        try:
            # Base threshold
            base_threshold = self.confidence_threshold

            # Get recent performance
            recent_trades = self.performance_tracker['recent_trades'][-10:]  # Last 10 trades

            if len(recent_trades) < 3:
                return max(0.55, base_threshold)  # Start at 55% for new sessions (reduced for more opportunities)

            # Calculate recent win rate
            recent_wins = sum(1 for trade in recent_trades if trade.get('result') == 'win')
            recent_win_rate = recent_wins / len(recent_trades)

            # Adjust threshold based on performance (more balanced approach)
            if recent_win_rate >= 0.75:  # 75%+ win rate
                adjusted_threshold = base_threshold * 0.85  # Lower threshold (more trades)
            elif recent_win_rate >= 0.65:  # 65%+ win rate
                adjusted_threshold = base_threshold * 0.90  # Slightly lower threshold
            elif recent_win_rate >= 0.55:  # 55%+ win rate
                adjusted_threshold = base_threshold  # Keep same threshold
            elif recent_win_rate >= 0.45:  # 45%+ win rate
                adjusted_threshold = base_threshold * 1.05  # Slightly raise threshold
            else:  # Below 45% win rate
                adjusted_threshold = base_threshold * 1.15  # Moderately raise threshold

            # Ensure balanced threshold range for better opportunities
            final_threshold = max(0.45, min(0.80, adjusted_threshold))

            if final_threshold != base_threshold:
                logger.info(f"🧠 Dynamic threshold: {final_threshold:.1%} (was {base_threshold:.1%}) - Win rate: {recent_win_rate:.1%}")

            return final_threshold

        except Exception as e:
            logger.error(f"❌ Error calculating dynamic threshold: {e}")
            return max(0.50, self.confidence_threshold)

    def _passes_advanced_filters(self, confidence: float) -> bool:
        """🔥 Advanced accuracy filters to prevent losses"""
        try:
            # 1. Enhanced confidence requirement during losing streaks (more balanced)
            if self.advanced_filters.get('losing_streak_protection', True):
                recent_trades = self.performance_tracker['recent_trades'][-5:]  # Last 5 trades
                if len(recent_trades) >= 3:
                    recent_losses = sum(1 for trade in recent_trades if trade.get('result') == 'loss')
                    min_confidence_streak = self.advanced_filters.get('min_confidence_losing_streak', 0.75)
                    if recent_losses >= 4:  # 4+ losses in last 5 trades (more lenient)
                        if confidence < min_confidence_streak:  # Require 75%+ confidence (configurable)
                            logger.info(f"⚠️ Losing streak detected - requiring {min_confidence_streak:.1%}+ confidence (got {confidence:.1%})")
                            return False

            # 2. Session-based filtering (avoid volatile periods)
            if self.advanced_filters.get('session_filter_enabled', True):
                current_hour = datetime.now().hour
                volatile_hours = self.advanced_filters.get('volatile_hours', [8, 9, 13, 14, 15, 16])
                min_confidence_volatile = self.advanced_filters.get('min_confidence_volatile', 0.80)

                if current_hour in volatile_hours and confidence < min_confidence_volatile:
                    logger.info(f"⚠️ Volatile session hour {current_hour} - requiring {min_confidence_volatile:.1%}+ confidence")
                    return False

            # 3. Balanced confidence filter for better opportunities
            if confidence < 0.50:  # Reduced from 65% for more opportunities
                logger.info(f"⚠️ Confidence {confidence:.1%} below 50% safety threshold")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error in advanced filters: {e}")
            return confidence >= 0.55  # Fallback to 55% minimum (more balanced)

    def calculate_martingale_amount(self) -> float:
        """💰 Calculate current trade amount based on Martingale strategy"""
        if not self.use_martingale or self.current_martingale_step == 0:
            return self.base_trade_amount

        # Calculate Martingale amount: base_amount * (multiplier ^ step)
        martingale_amount = self.base_trade_amount * (self.martingale_multiplier ** self.current_martingale_step)

        logger.info(f"🎲 Martingale Step {self.current_martingale_step}: ${martingale_amount:.2f} "
                   f"(x{self.martingale_multiplier ** self.current_martingale_step:.1f})")

        return martingale_amount

    def handle_trade_result(self, is_win: bool, profit: float, opportunity: Dict = None):
        """🎯 ENHANCED TRADE RESULT HANDLING - Track performance and update Martingale state"""

        # 📊 TRACK PERFORMANCE FOR ADAPTIVE LEARNING
        trade_record = {
            'timestamp': datetime.now(),
            'result': 'win' if is_win else 'loss',
            'profit': profit,
            'opportunity': opportunity
        }

        # Add to recent trades (keep last 20)
        self.performance_tracker['recent_trades'].append(trade_record)
        if len(self.performance_tracker['recent_trades']) > 20:
            self.performance_tracker['recent_trades'].pop(0)

        # Update model performance tracking
        self.performance_tracker['model_performance']['brain_model']['total'] += 1
        if is_win:
            self.performance_tracker['model_performance']['brain_model']['wins'] += 1

        if is_win:
            # Reset Martingale on win
            if self.current_martingale_step > 0:
                logger.info(f"🎉 WIN! Resetting Martingale (was at step {self.current_martingale_step})")
            self.current_martingale_step = 0
            self.consecutive_losses = 0
            self.trade_amount = self.base_trade_amount
            self.pending_martingale = False

            # 🎯 LOG PERFORMANCE UPDATE
            recent_win_rate = self._calculate_recent_win_rate()
            logger.info(f"📊 Recent Win Rate: {recent_win_rate:.1%} (Last {len(self.performance_tracker['recent_trades'])} trades)")

        else:
            # Increase Martingale on loss
            self.consecutive_losses += 1

            # 🚨 ENHANCED LOSS HANDLING
            recent_win_rate = self._calculate_recent_win_rate()
            if recent_win_rate < 0.50 and len(self.performance_tracker['recent_trades']) >= 5:
                logger.warning(f"🚨 LOW WIN RATE ALERT: {recent_win_rate:.1%} - Increasing accuracy requirements!")

            if self.use_martingale and self.current_martingale_step < self.max_martingale_steps:
                self.current_martingale_step += 1
                self.trade_amount = self.calculate_martingale_amount()
                logger.info(f"💔 LOSS #{self.consecutive_losses}! Activating Martingale step {self.current_martingale_step}")
                logger.info(f"🎲 Next trade amount: ${self.trade_amount:.2f}")

                # Set flag to execute Martingale trade (will use next best signal)
                self.pending_martingale = True
                logger.info(f"🎯 Martingale trade queued: Will use BEST SIGNAL from next cycle")
            else:
                if self.current_martingale_step >= self.max_martingale_steps:
                    logger.warning(f"⚠️ Maximum Martingale steps ({self.max_martingale_steps}) reached! Resetting to base amount.")
                    self.current_martingale_step = 0
                    self.consecutive_losses = 0
                    self.trade_amount = self.base_trade_amount
                    self.pending_martingale = False

    def _calculate_recent_win_rate(self) -> float:
        """Calculate recent win rate for performance tracking"""
        recent_trades = self.performance_tracker['recent_trades']
        if not recent_trades:
            return 0.0

        wins = sum(1 for trade in recent_trades if trade['result'] == 'win')
        return wins / len(recent_trades)

    async def execute_martingale_trade_with_signal(self, opportunity: Dict) -> Optional[Dict]:
        """🎲 Execute a Martingale trade using the current best signal"""
        if not self.pending_martingale:
            return None

        asset = opportunity['asset']
        signal = opportunity['signal']
        confidence = opportunity['confidence']
        optimal_expiry = opportunity['optimal_expiry']
        payout = opportunity['payout']

        logger.info("🎲 EXECUTING MARTINGALE TRADE WITH CURRENT BEST SIGNAL 🎲")
        logger.info(f"   📊 Asset: {asset}")
        logger.info(f"   🎯 Signal: {signal}")
        logger.info(f"   💰 Amount: ${self.trade_amount:.2f} (Step {self.current_martingale_step})")
        logger.info(f"   🕐 Expiry: {optimal_expiry}s")
        logger.info(f"   🎯 Confidence: {confidence:.1%}")
        logger.info(f"   💎 Payout: {payout:.1f}%")

        # Set the asset for trading
        self.selected_asset = asset

        # Execute the Martingale trade with current signal and opportunity data for AI learning
        trade_result = await self.execute_trade(signal, confidence, optimal_expiry, opportunity)

        if trade_result:
            self.pending_martingale = False
            logger.info("✅ Martingale trade executed successfully with current best signal!")
            return trade_result
        else:
            logger.error("❌ Martingale trade execution failed!")
            self.pending_martingale = False
            return None

    async def execute_martingale_trade(self) -> Optional[Dict]:
        """🎲 Execute a Martingale trade using the last trade parameters (DEPRECATED - use execute_martingale_trade_with_signal)"""
        if not self.pending_martingale or not self.last_trade_signal or not self.last_trade_asset:
            return None

        logger.info("🎲 EXECUTING MARTINGALE TRADE (OLD METHOD) 🎲")
        logger.info(f"   📊 Asset: {self.last_trade_asset}")
        logger.info(f"   🎯 Signal: {self.last_trade_signal}")
        logger.info(f"   💰 Amount: ${self.trade_amount:.2f} (Step {self.current_martingale_step})")
        logger.info(f"   🕐 Expiry: {self.last_trade_expiry}s")

        # Set the asset for trading
        self.selected_asset = self.last_trade_asset

        # Execute the Martingale trade
        trade_result = await self.execute_trade(
            self.last_trade_signal,
            0.8,  # Use high confidence for Martingale trades
            self.last_trade_expiry
        )

        if trade_result:
            self.pending_martingale = False
            logger.info("✅ Martingale trade executed successfully!")
            return trade_result
        else:
            logger.error("❌ Martingale trade execution failed!")
            self.pending_martingale = False
            return None

    def get_martingale_info(self) -> Dict[str, any]:
        """📊 Get current Martingale status information"""
        return {
            'enabled': self.use_martingale,
            'current_step': self.current_martingale_step,
            'consecutive_losses': self.consecutive_losses,
            'current_amount': self.trade_amount,
            'base_amount': self.base_trade_amount,
            'multiplier': self.martingale_multiplier,
            'max_steps': self.max_martingale_steps,
            'next_amount': self.calculate_martingale_amount() if self.current_martingale_step < self.max_martingale_steps else self.base_trade_amount
        }

    def show_price_action_insights(self):
        """🎯 Display production-grade price action analysis insights"""
        try:
            logger.info("=" * 70)
            logger.info("🎯 PRODUCTION-GRADE PRICE ACTION ANALYSIS v3.0.0")
            logger.info("=" * 70)

            logger.info("� PURE RULE-BASED LOGIC COMPONENTS:")
            logger.info("   📈 Trend Detection (Higher Highs/Lows Analysis)")
            logger.info("   🕯️ Candlestick Pattern Recognition:")
            logger.info("      • Bullish/Bearish Engulfing")
            logger.info("      • Pin Bar (Hammer/Shooting Star)")
            logger.info("      • Doji (Reversal Signals)")
            logger.info("      • Inside Bar (Consolidation)")
            logger.info("      • Marubozu (Strong Momentum)")
            logger.info("   🔍 Structure Break Detection (Swing High/Low Breaks)")
            logger.info("   📊 Support/Resistance Zone Analysis")
            logger.info("   🎯 Supply/Demand Rejection Analysis")
            logger.info("   🧠 AI-like Confidence Scoring (≥90% threshold)")
            logger.info("   📝 Detailed Reasoning & Explanations")

            logger.info("\n🔥 ENSEMBLE CONFIGURATION:")
            logger.info(f"   🎯 Price Action Weight: {self.model_weights['price_action']:.0%}")
            logger.info(f"   🧠 Intelligent Model Weight: {self.model_weights['intelligent']:.0%}")
            logger.info(f"   ⏱️ Optimal Expiry: 60s/90s only (maximum accuracy)")
            logger.info(f"   � Enhanced Mode: {'✅ ENABLED' if self.enhanced_mode else '❌ DISABLED'}")

            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"❌ Error displaying price action insights: {e}")

    async def execute_trade(self, signal: str, confidence: float, optimal_expiry: int = 60, opportunity_data: Dict = None) -> Optional[Dict]:
        """💰 Execute a binary options trade with intelligent expiry time"""
        try:
            if not self.selected_asset:
                logger.error("❌ No asset selected for trading")
                return None

            # Convert signal to direction
            direction = "call" if signal == "UP" else "put"

            # Validate and fix expiry time
            duration = self._validate_expiry_time(optimal_expiry)

            # Use current Martingale amount
            current_amount = self.trade_amount

            # Format expiry time nicely
            def format_expiry_time(seconds: int) -> str:
                """Format expiry time in a readable format"""
                if seconds < 60:
                    return f"{seconds}s"
                elif seconds < 3600:
                    minutes = seconds // 60
                    return f"{minutes}m"
                else:
                    hours = seconds // 3600
                    return f"{hours}h"

            logger.info(f"🚀 Executing {direction.upper()} trade on {self.selected_asset}")
            logger.info(f"   💵 Amount: ${current_amount:.2f}")
            if self.current_martingale_step > 0:
                logger.info(f"   🎲 Martingale Step: {self.current_martingale_step} (x{self.martingale_multiplier ** self.current_martingale_step:.1f})")
            logger.info(f"   🕐 Smart Expiry: {format_expiry_time(duration)}")
            logger.info(f"   🎯 Confidence: {confidence:.1%}")

            # Debug logging for trade parameters
            logger.debug(f"🔧 Trade Parameters: asset={self.selected_asset}, direction={direction}, duration={duration}, amount={current_amount}")

            # Execute the trade
            success, trade_info = await self.client.buy(
                amount=current_amount,
                asset=self.selected_asset,
                direction=direction,
                duration=duration
            )

            if success and trade_info:
                trade_id = trade_info.get('id')
                logger.info(f"✅ Trade executed successfully! ID: {trade_id}")

                # Update tracking
                self.last_trade_time = datetime.now()
                self.trades_this_hour += 1

                # Return trade info for monitoring (including opportunity data for AI learning)
                return {
                    'id': trade_id,
                    'asset': self.selected_asset,
                    'direction': direction,
                    'amount': current_amount,
                    'confidence': confidence,
                    'timestamp': self.last_trade_time,
                    'duration': duration,
                    'optimal_expiry': optimal_expiry,
                    'martingale_step': self.current_martingale_step,
                    'opportunity_data': opportunity_data  # For AI learning
                }
            else:
                # Enhanced error handling for expiration issues
                error_msg = str(trade_info) if trade_info else "Unknown error"
                logger.error(f"❌ Trade execution failed: {error_msg}")

                if "expiration" in error_msg.lower():
                    logger.warning("⚠️ Expiration error detected. Possible causes:")
                    logger.warning(f"   - Invalid expiry time: {duration}s")
                    logger.warning(f"   - Asset {self.selected_asset} may not support this expiry")
                    logger.warning(f"   - Market may be closed for this asset")
                    logger.warning("💡 Try using standard expiry times: 60s, 120s, 300s")

                return None

        except Exception as e:
            logger.error(f"❌ Error executing trade: {str(e)}")
            return None

    def _validate_expiry_time(self, expiry: int) -> int:
        """🔧 Validate expiry time - SUPPORTS OPTIMAL EXPIRY SELECTION (15s, 30s, 60s, 90s)"""
        try:
            # ⚡ UPDATED: Support all optimal expiry times from enhanced analysis
            # The enhanced price action model now intelligently selects optimal expiry

            # Supported expiry times by Quotex API
            allowed_durations = [15, 30, 60, 90]

            # If expiry is already in allowed list, use it directly
            if expiry in allowed_durations:
                logger.info(f"⚡ Using optimal expiry: {expiry}s (selected by enhanced analysis)")
                return expiry

            # If expiry is not in allowed list, find closest supported value
            closest_expiry = min(allowed_durations, key=lambda x: abs(x - expiry))

            if expiry != closest_expiry:
                logger.info(f"🔧 Adjusted expiry from {expiry}s to {closest_expiry}s (nearest supported)")

            return closest_expiry

        except Exception as e:
            logger.warning(f"⚠️ Error validating expiry time: {e}, using safe default 60s")
            return 60

    async def monitor_trade(self, trade_info: Dict) -> bool:
        """📊 Monitor a trade until completion and update statistics"""
        try:
            trade_id = trade_info['id']
            duration = trade_info['duration']
            asset = trade_info['asset']
            direction = trade_info['direction']
            amount = trade_info['amount']
            confidence = trade_info['confidence']

            logger.info(f"⏳ Monitoring trade {trade_id} ({asset} {direction.upper()}) for {duration} seconds...")
            logger.info(f"💰 Trade Amount: ${amount:.2f} | Confidence: {confidence:.1%}")

            # Show countdown timer
            for remaining in range(duration, 0, -10):
                if remaining <= 30:
                    logger.info(f"⏰ {remaining} seconds remaining...")
                elif remaining % 30 == 0:
                    logger.info(f"⏰ {remaining} seconds remaining...")
                await asyncio.sleep(10)

            logger.info("🔍 Trade duration completed - checking result...")

            # Wait a bit more for result to be available
            await asyncio.sleep(10)

            # Check trade result
            result, trade_result = await self.client.get_result(trade_id)

            if result:
                profit = float(trade_result.get('profitAmount', 0))
                is_win = profit > 0

                # Update balance
                self.stats.current_balance = await self.client.get_balance()

                # Log detailed result
                result_emoji = "🎉" if is_win else "😞"
                result_text = "WIN" if is_win else "LOSS"

                logger.info("=" * 50)
                logger.info(f"{result_emoji} TRADE RESULT: {result_text}")
                logger.info(f"📊 Asset: {asset}")
                logger.info(f"📈 Direction: {direction.upper()}")
                logger.info(f"💰 Profit/Loss: ${profit:.2f}")
                logger.info(f"💳 New Balance: ${self.stats.current_balance:.2f}")
                logger.info(f"🎯 Confidence: {confidence:.1%}")
                logger.info("=" * 50)

                # Update statistics
                self.stats.add_trade(
                    asset=asset,
                    direction=direction,
                    amount=amount,
                    result='win' if is_win else 'loss',
                    profit=profit,
                    confidence=confidence
                )

                # Show updated stats
                logger.info(f"📊 Total Trades: {self.stats.total_trades} | Win Rate: {self.stats.get_win_rate():.1f}%")
                logger.info(f"📈 Net P&L: ${self.stats.get_net_profit():.2f}")

                # Handle Martingale logic with opportunity data for AI learning
                opportunity_data = trade_info.get('opportunity_data')
                self.handle_trade_result(is_win, profit, opportunity_data)

                return is_win
            else:
                logger.error(f"❌ Could not get result for trade {trade_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Error monitoring trade: {str(e)}")
            return False

    async def analyze_single_asset(self, asset: str, payout: float) -> Optional[Dict]:
        """🎯 Price Action analysis of a single asset"""
        try:
            # Fetch candle data for price action analysis
            data = await self.fetch_candle_data(asset, count=100, verbose=False)
            if data is None:
                # Try once more with verbose logging to see what's failing
                logger.debug(f"🔍 Retrying {asset} with detailed logging...")
                data = await self.fetch_candle_data(asset, count=100, verbose=True)
                if data is None:
                    logger.info(f"❌ {asset}: No candle data available")
                    return None

            # 🔥 ENSEMBLE PREDICTION: Use both Price Action + Intelligent Model
            signal, confidence, optimal_expiry = await self.analyze_and_predict(data)

            # Log detailed prediction result for each asset
            status_emoji = "🎯" if signal != "NEUTRAL" else "⚪"
            confidence_emoji = "🔥" if confidence >= 0.85 else "⚡" if confidence >= 0.75 else "💫"
            expiry_emoji = "⚡" if optimal_expiry <= 60 else "🕐" if optimal_expiry <= 120 else "⏰"

            # Format expiry time nicely
            def format_expiry_time(seconds: int) -> str:
                """Format expiry time in a readable format"""
                if seconds < 60:
                    return f"{seconds}s"
                elif seconds < 3600:
                    minutes = seconds // 60
                    return f"{minutes}m"
                else:
                    hours = seconds // 3600
                    return f"{hours}h"

            expiry_formatted = format_expiry_time(optimal_expiry)

            # Show detailed analysis result with expiry and ensemble insights
            ensemble_info = " | 🔥 ENSEMBLE"
            logger.info(f"{status_emoji} {asset}: {signal} ({confidence:.1%}) {confidence_emoji}{ensemble_info} | "
                       f"Expiry: {expiry_formatted} {expiry_emoji} | Payout: {payout:.1f}%")

            # Check if signal meets our criteria for trading
            if signal != "NEUTRAL" and confidence >= self.confidence_threshold:
                logger.info(f"✅ {asset}: TRADING OPPORTUNITY DETECTED!")
                return {
                    'asset': asset,
                    'signal': signal,
                    'confidence': confidence,
                    'optimal_expiry': optimal_expiry,
                    'payout': payout,
                    'data': data,
                    'analysis_type': 'ensemble_prediction',
                    'pattern_detected': 'ensemble_confluence'
                }

            return None

        except Exception as e:
            logger.error(f"❌ {asset}: Analysis error - {str(e)}")
            return None

    async def analyze_all_assets_simultaneously(self, assets: List[Tuple[str, float]]) -> List[Dict]:
        """🔄 Analyze all assets simultaneously with detailed logging and batching"""
        try:
            logger.info("=" * 80)
            logger.info(f"🧠 SIMULTANEOUS ANALYSIS OF {len(assets)} ASSETS")
            logger.info("=" * 80)

            # Process in batches to avoid overwhelming the API
            batch_size = 18  # Process more assets at once for faster analysis
            all_opportunities = []
            total_start_time = time.time()

            for i in range(0, len(assets), batch_size):
                batch = assets[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(assets) + batch_size - 1) // batch_size

                logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} assets)...")

                # Create analysis tasks for this batch
                analysis_tasks = []
                asset_names = []

                for asset, payout in batch:
                    task = asyncio.create_task(self.analyze_single_asset(asset, payout))
                    analysis_tasks.append(task)
                    asset_names.append(asset)

                # Wait for batch to complete
                batch_start_time = time.time()
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                batch_time = time.time() - batch_start_time

                logger.info(f"⚡ Batch {batch_num} completed in {batch_time:.2f}s")

                # Process batch results
                for j, result in enumerate(results):
                    if isinstance(result, dict) and result is not None:
                        all_opportunities.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"❌ {asset_names[j]}: Task failed - {str(result)}")

                # Minimal delay between batches for speed
                if i + batch_size < len(assets):
                    await asyncio.sleep(0.1)

            total_analysis_time = time.time() - total_start_time

            logger.info("=" * 80)
            logger.info(f"⚡ Total analysis completed in {total_analysis_time:.2f} seconds")
            logger.info("=" * 80)

            # Sort opportunities by confidence (highest first)
            all_opportunities.sort(key=lambda x: x['confidence'], reverse=True)

            # Show comprehensive summary
            neutral_count = len(assets) - len(all_opportunities)
            logger.info("📊 ANALYSIS SUMMARY:")
            logger.info(f"   🎯 Trading Opportunities: {len(all_opportunities)}")
            logger.info(f"   ⚪ Neutral Signals: {neutral_count}")
            logger.info(f"   ⏱️ Total Analysis Time: {total_analysis_time:.2f}s")
            logger.info(f"   🚀 Average Time per Asset: {total_analysis_time/len(assets):.3f}s")

            # 🧠 Show AI insights periodically (every 10th cycle)
            if hasattr(self, '_cycle_count'):
                self._cycle_count += 1
            else:
                self._cycle_count = 1

            if self._cycle_count % 10 == 0:
                logger.info("\n🔥 ENSEMBLE SYSTEM UPDATE (Every 10 cycles):")
                logger.info("   📊 Dual AI Model System Active")
                logger.info("   🎯 Price Action + Intelligent Model Running")
                logger.info("   📈 Consensus-based signal generation")
                logger.info("   ⚡ 60s/90s optimized trading")

            if all_opportunities:
                logger.info("=" * 80)
                logger.info("🎯 TRADING OPPORTUNITIES FOUND:")
                logger.info("=" * 80)
                for i, opp in enumerate(all_opportunities, 1):
                    confidence_bar = "🔥" * min(10, int(opp['confidence'] * 10))
                    score = opp['confidence'] * opp['payout'] / 100
                    logger.info(f"#{i} {opp['asset']}: {opp['signal']} ({opp['confidence']:.1%}) {confidence_bar}")
                    logger.info(f"    💰 Payout: {opp['payout']:.1f}% | Score: {score:.1f}")
                logger.info("=" * 80)
            else:
                logger.info("⚠️ No trading opportunities found in this cycle")
                logger.info("💡 Continuing to scan for signals...")

            return all_opportunities

        except Exception as e:
            logger.error(f"❌ Error in simultaneous analysis: {str(e)}")
            return []

    async def execute_best_trade(self, opportunities: List[Dict]) -> Optional[Dict]:
        """💰 Execute the best trading opportunity from the list"""
        if not opportunities:
            return None

        # Select the best opportunity (highest confidence)
        best_opportunity = opportunities[0]
        asset = best_opportunity['asset']
        signal = best_opportunity['signal']
        confidence = best_opportunity['confidence']
        payout = best_opportunity['payout']

        # Final check before trading
        if not self.should_trade(signal, confidence):
            return None

        # Set the selected asset for trading
        self.selected_asset = asset

        logger.info(f"🚀 Selected best opportunity: {asset} ({signal}) - Confidence: {confidence:.1%}, Payout: {payout:.1f}%")

        # Execute the trade with optimal expiry and opportunity data
        optimal_expiry = best_opportunity.get('optimal_expiry', 60)
        return await self.execute_trade(signal, confidence, optimal_expiry, best_opportunity)

    async def run_trading_cycle(self, assets: List[Tuple[str, float]]):
        """🔄 Run one complete trading cycle with simultaneous analysis and trade completion"""
        try:
            # Always analyze all assets first to find the best current signal
            opportunities = await self.analyze_all_assets_simultaneously(assets)

            if not opportunities:
                # No opportunities found
                if self.pending_martingale:
                    logger.info("⚠️ No trading opportunities found, but Martingale is pending")
                    logger.info("💡 Waiting for better signal before executing Martingale trade")
                return True  # Continue running but no trades

            # Check if we have a pending Martingale trade
            if self.pending_martingale:
                logger.info("🎲 PENDING MARTINGALE TRADE DETECTED!")
                logger.info("🎯 Using BEST CURRENT SIGNAL for Martingale trade")

                # Use the best current opportunity for Martingale
                best_opportunity = opportunities[0]
                martingale_trade = await self.execute_martingale_trade_with_signal(best_opportunity)

                if martingale_trade:
                    # Wait for Martingale trade to complete
                    logger.info("=" * 60)
                    logger.info("⏳ WAITING FOR MARTINGALE TRADE COMPLETION...")
                    logger.info("=" * 60)

                    martingale_result = await self.monitor_trade(martingale_trade)

                    # Show Martingale trade completion
                    logger.info("=" * 60)
                    if martingale_result:
                        logger.info("🎉 MARTINGALE TRADE COMPLETED - WIN!")
                    else:
                        logger.info("😞 MARTINGALE TRADE COMPLETED - LOSS")
                    logger.info("=" * 60)

                    return True  # Return after Martingale trade
                else:
                    logger.error("❌ Martingale trade execution failed")
                    self.pending_martingale = False
                    # Continue to regular trade execution

            # Execute the best trade (normal trading)
            trade_info = await self.execute_best_trade(opportunities)
            if trade_info is None:
                logger.info("⚠️ No trade executed (filtered by risk management)")
                return True  # Continue running

            # Wait for trade to complete and show result
            logger.info("=" * 60)
            logger.info("⏳ WAITING FOR TRADE COMPLETION...")
            logger.info("=" * 60)

            trade_result = await self.monitor_trade(trade_info)

            # Show trade completion
            logger.info("=" * 60)
            if trade_result:
                logger.info("🎉 TRADE COMPLETED - WIN!")
            else:
                logger.info("😞 TRADE COMPLETED - LOSS")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {str(e)}")
            return False

    async def run(self, duration_minutes: Optional[int] = None):
        """🚀 Main trading bot execution loop with simultaneous asset analysis"""
        try:
            logger.info("🚀 Starting Multi-Asset Trading Bot...")

            # Connect and login
            if not await self.connect_and_login():
                return False

            # Discover all tradeable assets
            assets = await self.discover_tradeable_assets()
            if not assets:
                logger.error("❌ No suitable assets found")
                return False

            logger.info(f"✅ Trading bot is now running with {len(assets)} assets!")
            logger.info("🧠 Bot will analyze ALL assets simultaneously each cycle")
            self.running = True

            # Set end time if duration specified
            end_time = None
            if duration_minutes:
                end_time = datetime.now() + timedelta(minutes=duration_minutes)
                logger.info(f"⏰ Bot will run for {duration_minutes} minutes")

            # Main trading loop
            cycle_count = 0
            last_asset_refresh = datetime.now()

            while self.running:
                try:
                    # Check if we should stop
                    if end_time and datetime.now() >= end_time:
                        logger.info("⏰ Time limit reached")
                        break

                    # Check balance
                    current_balance = await self.client.get_balance()
                    self.stats.current_balance = current_balance

                    if current_balance < self.trade_amount:
                        logger.info("🛑 Stopping trading: Insufficient balance for next trade")
                        break

                    # Refresh asset list every 10 minutes (in case new assets open/close)
                    if (datetime.now() - last_asset_refresh).total_seconds() > 600:
                        logger.info("🔄 Refreshing asset list...")
                        new_assets = await self.discover_tradeable_assets()
                        if new_assets:
                            assets = new_assets
                            last_asset_refresh = datetime.now()

                    # Run trading cycle with all assets
                    cycle_count += 1
                    logger.info(f"🔄 Multi-Asset Analysis Cycle #{cycle_count}")

                    success = await self.run_trading_cycle(assets)
                    if not success:
                        logger.error("❌ Trading cycle failed")
                        break

                    # Print periodic statistics
                    if cycle_count % 5 == 0:  # More frequent stats for multi-asset
                        self.print_periodic_stats()

                    # Faster cycles for quicker trade execution (10 seconds)
                    await asyncio.sleep(10)

                except KeyboardInterrupt:
                    logger.info("⏹️ Received stop signal")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {str(e)}")
                    await asyncio.sleep(10)  # Wait before retrying

            return True

        except Exception as e:
            logger.error(f"❌ Critical error in bot execution: {str(e)}")
            return False
        finally:
            await self.shutdown()

    def print_periodic_stats(self):
        """📊 Print periodic statistics during trading"""
        logger.info("=" * 40)
        logger.info("📊 PERIODIC STATS UPDATE")
        logger.info(f"💰 Balance: ${self.stats.current_balance:.2f}")
        logger.info(f"🎯 Trades: {self.stats.total_trades}")
        logger.info(f"🏆 Win Rate: {self.stats.get_win_rate():.1f}%")
        logger.info(f"📈 Net P&L: ${self.stats.get_net_profit():.2f}")
        logger.info("=" * 40)

    async def shutdown(self):
        """🏁 Gracefully shutdown the trading bot"""
        try:
            logger.info("🏁 Trading bot stopped")
            self.running = False

            # Print final statistics
            self.stats.print_summary()

            # Close connection
            if self.client:
                self.client.close()

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")


# Demo and testing functions
async def demo_test(duration_minutes: int = 30):
    """🧪 Run a demo test of the ensemble trading bot"""
    logger.info(f"🧪 Starting {duration_minutes}-minute ensemble demo test...")
    logger.info("🔥 Bot will analyze ALL available assets with DUAL AI ENSEMBLE!")
    logger.info("🎯 Price Action + Intelligent Model working together!")

    bot = QuotexTradingBot(
        confidence_threshold=0.50,  # 70% minimum confidence for maximum accuracy
        min_payout=85.0,           # Lower payout requirement for demo
        trade_amount=10.0,         # Reasonable trade amount for demo
        max_trades_per_hour=10,    # Fewer, higher quality trades
        demo_mode=True             # Use demo account
    )

    success = await bot.run(duration_minutes=duration_minutes)

    if success:
        logger.info("✅ Ensemble demo test completed!")
    else:
        logger.error("❌ Demo test failed!")

    return success


async def live_trading(confidence_threshold: float = 0.85,
                      trade_amount: float = 10.0):
    """💰 Run live trading with specified parameters"""
    logger.info("💰 Starting LIVE trading mode...")
    logger.warning("⚠️ WARNING: This will use real money!")

    # Confirmation prompt
    try:
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != 'CONFIRM':
            logger.info("❌ Live trading cancelled")
            return False
    except:
        logger.info("❌ Live trading cancelled")
        return False

    bot = QuotexTradingBot(
        confidence_threshold=confidence_threshold,
        min_payout=85.0,
        trade_amount=trade_amount,
        max_trades_per_hour=15,
        demo_mode=False  # Use live account
    )

    return await bot.run()


async def main():
    """🎮 Main entry point with menu system"""
    print("🤖 QUOTEX ENSEMBLE TRADING BOT 🤖")
    print("=" * 55)
    print("🔥 ENSEMBLE MODEL SYSTEM (DUAL AI)")
    print("📈 PRICE ACTION + INTELLIGENT MODEL")
    print("🎯 MAXIMUM ACCURACY THROUGH CONSENSUS")
    print("=" * 55)
    print("1. Multi-Asset Demo Test (30 minutes)")
    print("2. Multi-Asset Demo Test (Custom duration)")
    print("3. Live Trading (⚠️ Real Money)")
    print("4. Quick Ensemble Analysis Test")
    print("5. Exit")
    print("=" * 55)
    print("🔥 ENSEMBLE FEATURES:")
    print("💡 Price Action Model: Market Structure, S/R, Patterns")
    print("💡 Intelligent Model: AI-Enhanced Multi-Timeframe")
    print("💡 Consensus Voting: Agreement Boost, Disagreement Filter")
    print("💡 Only 60s/90s trades for maximum accuracy!")
    print("=" * 55)

    try:
        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            await demo_test(30)
        elif choice == "2":
            try:
                duration = int(input("Enter duration in minutes: "))
                await demo_test(duration)
            except ValueError:
                logger.error("❌ Invalid duration")
        elif choice == "3":
            await live_trading()
        elif choice == "4":
            # Quick test of ensemble analysis
            logger.info("🧪 Running quick ensemble analysis test...")
            bot = QuotexTradingBot(demo_mode=True)

            # Generate mock OHLC data for testing
            import numpy as np
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
            np.random.seed(42)  # For reproducible results

            # Generate realistic OHLC data
            close_prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
            high_prices = close_prices + np.random.uniform(0, 0.5, 100)
            low_prices = close_prices - np.random.uniform(0, 0.5, 100)
            open_prices = np.roll(close_prices, 1)
            open_prices[0] = close_prices[0]

            mock_data = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices
            })

            signal, confidence, optimal_expiry = await bot.analyze_and_predict(mock_data)
            logger.info(f"✅ Ensemble analysis test complete: {signal} ({confidence:.1%}) - Optimal Expiry: {optimal_expiry}s")
        elif choice == "5":
            logger.info("👋 Goodbye!")
        else:
            logger.error("❌ Invalid choice")

    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
