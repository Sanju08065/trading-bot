#!/usr/bin/env python3
"""
🎯 ENHANCED QUOTEX TRADING BOT GUI 🎯

Advanced PyQt5 GUI interface for the Quotex Trading Bot with:
- Real-time charts and technical indicators
- Advanced monitoring and analytics
- Modern dark theme with animations
- Performance metrics and insights
- Enhanced controls and visualization
"""

import sys
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import math
import random

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QGridLayout, QFormLayout, QGroupBox,
                                QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
                                QDoubleSpinBox, QCheckBox, QMessageBox, QTabWidget,
                                QProgressBar, QSlider, QComboBox, QTableWidget,
                                QTableWidgetItem, QHeaderView, QSplitter, QFrame,
                                QScrollArea, QToolButton, QButtonGroup, QRadioButton,
                                QSystemTrayIcon, QMenu, QAction, QStatusBar,
                                QToolBar, QSizePolicy, QSpacerItem)
    from PyQt5.QtCore import (QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation,
                             QEasingCurve, QRect, QPoint, QSize, QParallelAnimationGroup,
                             QSequentialAnimationGroup, pyqtProperty, QSettings)
    from PyQt5.QtWidgets import QShortcut
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtGui import (QFont, QPixmap, QIcon, QPainter, QPen, QBrush, QColor,
                            QLinearGradient, QRadialGradient, QPalette, QFontMetrics,
                            QPolygon, QPainterPath, QMovie)
    print("✅ Enhanced PyQt5 imported successfully")
except ImportError:
    print("❌ PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)

try:
    from bot import QuotexTradingBot
    print("✅ Trading bot imported successfully")
except ImportError as e:
    print(f"❌ Failed to import trading bot: {e}")
    sys.exit(1)


# ============================================================================
# ENHANCED CUSTOM WIDGETS
# ============================================================================

class AnimatedButton(QPushButton):
    """Enhanced button with hover animations and gradient effects"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

        # Gradient colors
        self.normal_color = "#0078d4"
        self.hover_color = "#106ebe"
        self.pressed_color = "#005a9e"
        self.disabled_color = "#555555"

    def enterEvent(self, event):
        """Animate on hover"""
        self.animate_scale(1.05)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Animate on leave"""
        self.animate_scale(1.0)
        super().leaveEvent(event)

    def animate_scale(self, scale_factor):
        """Animate button scaling"""
        current_rect = self.geometry()
        center = current_rect.center()
        new_width = int(current_rect.width() * scale_factor)
        new_height = int(current_rect.height() * scale_factor)
        new_rect = QRect(0, 0, new_width, new_height)
        new_rect.moveCenter(center)

        self.animation.setStartValue(current_rect)
        self.animation.setEndValue(new_rect)
        self.animation.start()


class CircularProgressBar(QWidget):
    """Custom circular progress bar with gradient and animations"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self._value = 0
        self._max_value = 100
        self._min_value = 0
        self._start_angle = 90
        self._span_angle = 360

        # Colors
        self.bg_color = QColor(45, 45, 45)
        self.progress_color = QColor(0, 120, 212)
        self.text_color = QColor(255, 255, 255)

        # Animation
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(1000)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

    @pyqtProperty(int)
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self.update()

    def set_value_animated(self, value):
        """Set value with animation"""
        self.animation.setStartValue(self._value)
        self.animation.setEndValue(value)
        self.animation.start()

    def paintEvent(self, event):
        """Custom paint event for circular progress"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate dimensions
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2 - 10

        # Draw background circle
        painter.setPen(QPen(self.bg_color, 8))
        painter.drawEllipse(center, radius, radius)

        # Draw progress arc
        if self._value > 0:
            progress_angle = int((self._value / self._max_value) * 360)
            painter.setPen(QPen(self.progress_color, 8, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(center.x() - radius, center.y() - radius,
                          radius * 2, radius * 2, self._start_angle * 16,
                          -progress_angle * 16)

        # Draw text
        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{self._value}%")


class TradingChart(QWidget):
    """Simple trading chart widget with candlesticks"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.data_points = []
        self.max_points = 50

        # Colors
        self.bg_color = QColor(20, 20, 20)
        self.grid_color = QColor(40, 40, 40)
        self.bull_color = QColor(0, 255, 0)
        self.bear_color = QColor(255, 0, 0)

        # Generate sample data
        self.generate_sample_data()

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(2000)  # Update every 2 seconds

    def generate_sample_data(self):
        """Generate sample candlestick data"""
        base_price = 1.2000
        for i in range(self.max_points):
            change = random.uniform(-0.0020, 0.0020)
            base_price += change

            open_price = base_price
            close_price = base_price + random.uniform(-0.0010, 0.0010)
            high_price = max(open_price, close_price) + random.uniform(0, 0.0005)
            low_price = min(open_price, close_price) - random.uniform(0, 0.0005)

            self.data_points.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'time': datetime.now() - timedelta(minutes=(self.max_points - i))
            })

    def update_data(self):
        """Add new data point"""
        if len(self.data_points) >= self.max_points:
            self.data_points.pop(0)

        last_price = self.data_points[-1]['close'] if self.data_points else 1.2000
        change = random.uniform(-0.0015, 0.0015)
        new_price = last_price + change

        open_price = last_price
        close_price = new_price
        high_price = max(open_price, close_price) + random.uniform(0, 0.0003)
        low_price = min(open_price, close_price) - random.uniform(0, 0.0003)

        self.data_points.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'time': datetime.now()
        })

        self.update()

    def paintEvent(self, event):
        """Paint the chart"""
        if not self.data_points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), self.bg_color)

        # Calculate dimensions
        rect = self.rect()
        margin = 20
        chart_rect = rect.adjusted(margin, margin, -margin, -margin)

        # Calculate price range
        all_prices = []
        for point in self.data_points:
            all_prices.extend([point['high'], point['low']])

        if not all_prices:
            return

        min_price = min(all_prices)
        max_price = max(all_prices)
        price_range = max_price - min_price

        if price_range == 0:
            price_range = 0.001

        # Draw grid
        painter.setPen(QPen(self.grid_color, 1))
        for i in range(5):
            y = int(chart_rect.top() + (chart_rect.height() * i / 4))
            painter.drawLine(chart_rect.left(), y, chart_rect.right(), y)

        # Draw candlesticks
        candle_width = max(2, chart_rect.width() // len(self.data_points) - 2)

        for i, point in enumerate(self.data_points):
            x = chart_rect.left() + (i * chart_rect.width() // len(self.data_points))

            # Calculate y positions
            open_y = int(chart_rect.bottom() - ((point['open'] - min_price) / price_range * chart_rect.height()))
            close_y = int(chart_rect.bottom() - ((point['close'] - min_price) / price_range * chart_rect.height()))
            high_y = int(chart_rect.bottom() - ((point['high'] - min_price) / price_range * chart_rect.height()))
            low_y = int(chart_rect.bottom() - ((point['low'] - min_price) / price_range * chart_rect.height()))

            # Choose color
            color = self.bull_color if point['close'] > point['open'] else self.bear_color
            painter.setPen(QPen(color, 1))
            painter.setBrush(QBrush(color))

            # Draw high-low line
            painter.drawLine(x + candle_width//2, high_y, x + candle_width//2, low_y)

            # Draw body
            body_top = min(open_y, close_y)
            body_height = abs(close_y - open_y)
            if body_height < 1:
                body_height = 1

            painter.drawRect(x, body_top, candle_width, body_height)


class BeautifulMetricCard(QFrame):
    """Beautiful metric card with modern design and animations"""

    def __init__(self, title="", value="", icon="", color="#4CAF50", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumHeight(120)
        self.setMinimumWidth(200)
        self.color = color

        # Animation for hover effects
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(20, 15, 20, 15)

        # Icon and title row
        header_layout = QHBoxLayout()

        # Icon
        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 24px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        header_layout.addWidget(self.icon_label)

        header_layout.addStretch()

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #bbb;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        self.title_label.setAlignment(Qt.AlignRight)
        header_layout.addWidget(self.title_label)

        layout.addLayout(header_layout)

        # Value (main display)
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 28px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

        # Apply beautiful styling
        self.apply_styling()

    def apply_styling(self):
        """Apply beautiful card styling"""
        self.setStyleSheet(f"""
            BeautifulMetricCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e1e1e, stop:0.5 #2a2a2a, stop:1 #1e1e1e);
                border: 2px solid #333;
                border-radius: 15px;
                margin: 5px;
            }}
            BeautifulMetricCard:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2a2a2a, stop:0.5 #3a3a3a, stop:1 #2a2a2a);
                border: 2px solid {self.color};
            }}
        """)

    def update_value(self, value, animated=True):
        """Update the value with smooth animation"""
        if animated:
            # Pulse effect
            self.value_label.setStyleSheet(f"""
                QLabel {{
                    color: #ffffff;
                    font-size: 30px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                }}
            """)
            QTimer.singleShot(300, lambda: self.value_label.setStyleSheet(f"""
                QLabel {{
                    color: {self.color};
                    font-size: 28px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                }}
            """))

        self.value_label.setText(str(value))

    def enterEvent(self, event):
        """Enhanced hover effect"""
        self.setStyleSheet(f"""
            BeautifulMetricCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2a2a2a, stop:0.5 #3a3a3a, stop:1 #2a2a2a);
                border: 2px solid {self.color};
                border-radius: 15px;
                margin: 5px;
            }}
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Return to normal state"""
        self.apply_styling()
        super().leaveEvent(event)


class MiniStatCard(QFrame):
    """Compact, beautiful mini statistics card"""

    def __init__(self, title="", value="", icon="", color="#4CAF50", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumHeight(80)
        self.setMinimumWidth(120)
        self.setMaximumWidth(150)
        self.color = color

        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(15, 10, 15, 10)

        # Icon and value row
        top_layout = QHBoxLayout()

        # Icon
        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 20px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        top_layout.addWidget(self.icon_label)

        top_layout.addStretch()

        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        self.value_label.setAlignment(Qt.AlignRight)
        top_layout.addWidget(self.value_label)

        layout.addLayout(top_layout)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 10px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        self.title_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.title_label)

        # Apply styling
        self.apply_styling()

    def apply_styling(self):
        """Apply beautiful mini card styling"""
        self.setStyleSheet(f"""
            MiniStatCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2a2a2a);
                border: 1px solid #333;
                border-radius: 10px;
                margin: 2px;
            }}
            MiniStatCard:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2a2a2a, stop:1 #3a3a3a);
                border: 1px solid {self.color};
            }}
        """)

    def update_value(self, value, animated=True):
        """Update the value with smooth animation"""
        if animated:
            # Quick pulse effect
            self.value_label.setStyleSheet(f"""
                QLabel {{
                    color: #ffffff;
                    font-size: 20px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                }}
            """)
            QTimer.singleShot(200, lambda: self.value_label.setStyleSheet(f"""
                QLabel {{
                    color: {self.color};
                    font-size: 18px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                }}
            """))

        self.value_label.setText(str(value))

    def enterEvent(self, event):
        """Hover effect"""
        self.setStyleSheet(f"""
            MiniStatCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2a2a2a, stop:1 #3a3a3a);
                border: 1px solid {self.color};
                border-radius: 10px;
                margin: 2px;
            }}
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Return to normal state"""
        self.apply_styling()
        super().leaveEvent(event)


class GUIIntegratedBot(QuotexTradingBot):
    """Trading bot with GUI integration for live updates"""

    def __init__(self, gui_signals=None, **kwargs):
        super().__init__(**kwargs)
        self.gui_signals = gui_signals

    async def execute_trade(self, signal: str, confidence: float, optimal_expiry: int = 60, opportunity_data: dict = None):
        """Override execute_trade to emit GUI signals with AI learning support"""
        # Call parent method with all parameters
        trade_result = await super().execute_trade(signal, confidence, optimal_expiry, opportunity_data)

        if trade_result and self.gui_signals:
            # Emit trade started signal
            trade_info = {
                'id': trade_result['id'],
                'asset': trade_result['asset'],
                'direction': trade_result['direction'],
                'amount': trade_result['amount'],
                'confidence': trade_result['confidence'],
                'duration': trade_result['duration'],
                'timestamp': trade_result['timestamp'],
                'status': 'active'
            }
            self.gui_signals['trade_started'].emit(trade_info)

        return trade_result

    async def monitor_trade(self, trade_info: dict) -> bool:
        """Override monitor_trade to emit completion signals"""
        # Call parent method
        result = await super().monitor_trade(trade_info)

        if self.gui_signals:
            # Emit trade completed signal
            completion_info = {
                'id': trade_info['id'],
                'asset': trade_info['asset'],
                'direction': trade_info['direction'],
                'amount': trade_info['amount'],
                'result': 'win' if result else 'loss',
                'timestamp': trade_info['timestamp']
            }
            self.gui_signals['trade_completed'].emit(completion_info)

        return result


class BotWorker(QThread):
    """Worker thread to run the trading bot"""

    # Signals for communication with GUI
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    balance_signal = pyqtSignal(float)
    trade_signal = pyqtSignal(dict)
    stats_signal = pyqtSignal(dict)
    trade_started_signal = pyqtSignal(dict)
    trade_completed_signal = pyqtSignal(dict)
    psychology_analysis_signal = pyqtSignal(dict)  # New signal for psychology analysis

    def __init__(self, bot_config: Dict):
        super().__init__()
        self.bot_config = bot_config
        self.bot = None
        self.running = False
        self.balance_update_requested = False

    def run(self):
        """Run the trading bot in this thread"""
        try:
            self.log_signal.emit("🚀 Starting bot worker thread...")

            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.log_signal.emit("✅ Event loop created")

            # Initialize bot with GUI signals
            gui_signals = {
                'trade_started': self.trade_started_signal,
                'trade_completed': self.trade_completed_signal
            }

            self.log_signal.emit("🔧 Initializing enhanced trading bot...")
            self.log_signal.emit(f"📊 Bot config: {self.bot_config}")

            self.bot = GUIIntegratedBot(gui_signals=gui_signals, **self.bot_config)
            self.running = True
            self.log_signal.emit("✅ Bot initialized successfully")

            # Run bot
            self.log_signal.emit("🚀 Starting bot execution...")
            loop.run_until_complete(self.run_bot())

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_signal.emit(f"❌ Bot error: {str(e)}")
            self.log_signal.emit(f"🔍 Error details: {error_details}")
        finally:
            self.running = False
            self.status_signal.emit("Stopped")

    async def run_bot(self):
        """Run the bot with GUI integration"""
        try:
            self.log_signal.emit("🔄 Starting bot execution sequence...")
            self.status_signal.emit("Connecting...")
            self.log_signal.emit("🔌 Connecting to Quotex API...")

            # Connect to API
            self.log_signal.emit("📡 Attempting API connection...")
            connected = await self.bot.connect_and_login()
            if not connected:
                self.log_signal.emit("❌ Failed to connect to API")
                self.status_signal.emit("Connection Failed")
                return

            self.log_signal.emit("✅ Connected successfully!")
            self.status_signal.emit("Connected")

            # Get initial balance
            try:
                balance = await self.bot.client.get_balance()
                self.log_signal.emit(f"🔍 Initial balance from API: {balance} (type: {type(balance)}, formatted: {balance:.10f})")
                self.balance_signal.emit(balance)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Could not get balance: {str(e)}")
                self.balance_signal.emit(0.0)

            # Discover assets
            self.status_signal.emit("Discovering Assets...")
            self.log_signal.emit("🔍 Discovering tradeable assets...")

            assets = await self.bot.discover_tradeable_assets()
            if not assets:
                self.log_signal.emit("❌ No tradeable assets found")
                return

            self.log_signal.emit(f"✅ Found {len(assets)} tradeable assets")
            self.status_signal.emit("Running")

            # Main trading loop
            cycle = 0
            while self.running:
                cycle += 1
                self.log_signal.emit(f"🔄 Analysis Cycle #{cycle}")

                # Run trading cycle
                success = await self.bot.run_trading_cycle(assets)
                if not success:
                    self.log_signal.emit("⚠️ Trading cycle failed")

                # Update balance (regular cycle update)
                try:
                    balance = await self.bot.client.get_balance()
                    # Only log every 5th cycle to avoid spam
                    if cycle % 5 == 0:
                        self.log_signal.emit(f"🔍 Cycle #{cycle} balance from API: {balance:.10f}")
                    self.balance_signal.emit(balance)
                except Exception as e:
                    self.log_signal.emit(f"⚠️ Balance update failed: {str(e)}")
                    pass

                # Check for balance update requests (continuous updates)
                if self.balance_update_requested:
                    try:
                        balance = await self.bot.client.get_balance()
                        self.log_signal.emit(f"🔍 Requested balance from API: {balance:.10f}")
                        self.balance_signal.emit(balance)
                        self.balance_update_requested = False
                    except Exception as e:
                        self.log_signal.emit(f"⚠️ Requested balance update failed: {str(e)}")
                        self.balance_update_requested = False

                # Update stats
                stats = {
                    'total_trades': self.bot.stats.total_trades,
                    'wins': self.bot.stats.winning_trades,
                    'losses': self.bot.stats.losing_trades,
                    'win_rate': self.bot.stats.get_win_rate(),
                    'net_profit': self.bot.stats.get_net_profit(),
                    'martingale_step': self.bot.current_martingale_step,
                    'pending_martingale': self.bot.pending_martingale
                }
                self.stats_signal.emit(stats)

                # ⚡ ULTRA-FAST cycles for lightning-speed analysis
                await asyncio.sleep(3)

        except Exception as e:
            self.log_signal.emit(f"❌ Bot runtime error: {str(e)}")
        finally:
            if self.bot and self.bot.client:
                self.bot.client.close()
            self.status_signal.emit("Stopped")

    def request_balance_update(self):
        """Request balance update from the bot"""
        self.balance_update_requested = True

    def stop_bot(self):
        """Stop the trading bot"""
        self.running = False
        if self.bot and self.bot.client:
            self.bot.client.close()


class EnhancedTradingBotGUI(QMainWindow):
    """Enhanced main GUI window with modern features and animations"""

    def __init__(self):
        super().__init__()
        self.bot_worker = None
        self.active_trade = None
        self.trade_timer = QTimer()
        self.trade_timer.timeout.connect(self.update_trade_timer)

        # Settings
        self.settings = QSettings("TradingBot", "QuotexGUI")

        # System tray
        self.tray_icon = None
        self.setup_system_tray()

        # Performance tracking
        self.performance_data = {
            'balance_history': [],
            'trade_history': [],
            'hourly_stats': {}
        }

        # UI Update timers
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self.update_ui_animations)
        self.ui_update_timer.start(100)  # Update every 100ms for smooth animations

        # Balance update timer - fetch balance every 10 seconds
        self.balance_update_timer = QTimer()
        self.balance_update_timer.timeout.connect(self.fetch_current_balance)
        # Will start when bot connects

        self.init_ui()
        self.load_settings()

    def setup_system_tray(self):
        """Setup system tray icon"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)

            # Create a simple icon using a pixmap
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(0, 120, 212))  # Blue color
            icon = QIcon(pixmap)
            self.tray_icon.setIcon(icon)

            # Create tray menu
            tray_menu = QMenu()

            show_action = QAction("Show", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)

            hide_action = QAction("Hide", self)
            hide_action.triggered.connect(self.hide)
            tray_menu.addAction(hide_action)

            tray_menu.addSeparator()

            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.close)
            tray_menu.addAction(quit_action)

            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()

    def setup_responsive_window(self):
        """🖥️ Setup responsive window sizing for all screen resolutions"""
        # Get screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Calculate optimal window size (80% of screen for normal, 95% for large screens)
        if screen_width >= 1920:  # Large screens (1920x1080+)
            window_width = int(screen_width * 0.85)
            window_height = int(screen_height * 0.85)
        elif screen_width >= 1366:  # Medium screens (1366x768+)
            window_width = int(screen_width * 0.90)
            window_height = int(screen_height * 0.90)
        else:  # Small screens (1024x768 or smaller)
            window_width = int(screen_width * 0.95)
            window_height = int(screen_height * 0.95)

        # Set minimum size to ensure usability
        min_width = min(1200, screen_width - 100)
        min_height = min(800, screen_height - 100)
        self.setMinimumSize(min_width, min_height)

        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.setGeometry(x, y, window_width, window_height)

        # Enable window state saving
        self.setWindowState(Qt.WindowNoState)

        # Add keyboard shortcuts for window management
        self.setup_window_shortcuts()

    def setup_window_shortcuts(self):
        """Setup keyboard shortcuts for window management"""
        # F11 for fullscreen toggle
        fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen)

        # Ctrl+M for maximize/restore
        maximize_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        maximize_shortcut.activated.connect(self.toggle_maximize)

        # Ctrl+- for minimize
        minimize_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        minimize_shortcut.activated.connect(self.showMinimized)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode with proper layout adjustment"""
        if self.isFullScreen():
            self.showNormal()
            self.add_log("🖥️ Exited fullscreen mode")
        else:
            self.showFullScreen()
            self.add_log("🖥️ Entered fullscreen mode")

    def toggle_maximize(self):
        """Toggle maximize/restore window"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def setup_responsive_margins(self, layout):
        """Setup responsive margins based on window size"""
        window_width = self.width()

        if window_width >= 1920:  # Large screens
            margins = 15
        elif window_width >= 1366:  # Medium screens
            margins = 10
        else:  # Small screens
            margins = 5

        layout.setContentsMargins(margins, margins, margins, margins)

    def resizeEvent(self, event):
        """Handle window resize events for responsive layout"""
        super().resizeEvent(event)

        # Adjust layouts based on new size
        if hasattr(self, 'tab_widget'):
            self.adjust_responsive_layouts()

    def adjust_responsive_layouts(self):
        """Adjust layouts based on current window size"""
        window_width = self.width()
        window_height = self.height()

        # Adjust font sizes based on screen size
        if window_width >= 1920:
            base_font_size = 12
            title_font_size = 28
        elif window_width >= 1366:
            base_font_size = 11
            title_font_size = 24
        else:
            base_font_size = 10
            title_font_size = 20

        # Update header title size if it exists
        if hasattr(self, 'header_title_label'):
            self.header_title_label.setStyleSheet(f"""
                QLabel {{
                    font-size: {title_font_size}px;
                    font-weight: bold;
                    color: #FFFFFF;
                    margin: 5px;
                    padding: 10px;
                    background: transparent;
                    border: none;
                }}
            """)

        # Adjust metrics layout for smaller screens
        if hasattr(self, 'metrics_layout') and hasattr(self, 'metrics_cards'):
            self.adjust_metrics_layout(window_width)

    def adjust_metrics_layout(self, window_width):
        """Adjust metrics cards layout based on window width"""
        if not hasattr(self, 'metrics_cards'):
            return

        # Clear current layout
        for i in reversed(range(self.metrics_layout.count())):
            self.metrics_layout.itemAt(i).widget().setParent(None)

        # Rearrange based on window width
        if window_width >= 1400:  # Wide screens - 4 columns
            for i, card in enumerate(self.metrics_cards):
                self.metrics_layout.addWidget(card, 0, i)
        elif window_width >= 1000:  # Medium screens - 2x2 grid
            for i, card in enumerate(self.metrics_cards):
                row = i // 2
                col = i % 2
                self.metrics_layout.addWidget(card, row, col)
        else:  # Small screens - vertical stack
            for i, card in enumerate(self.metrics_cards):
                self.metrics_layout.addWidget(card, i, 0)

    def init_ui(self):
        """Initialize the enhanced user interface with responsive design"""
        self.setWindowTitle("🚀 Professional Trading Bot")

        # 🖥️ RESPONSIVE WINDOW SIZING
        self.setup_responsive_window()

        # Apply enhanced dark theme
        self.apply_enhanced_theme()

        # Create toolbar
        self.create_toolbar()

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with responsive margins
        main_layout = QVBoxLayout(central_widget)
        self.setup_responsive_margins(main_layout)
        main_layout.setSpacing(8)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_dashboard_tab()
        self.create_trading_tab()
        self.create_analytics_tab()
        self.create_settings_tab()
        self.create_logs_tab()

        # Enhanced status bar
        self.create_enhanced_status_bar()

    def apply_enhanced_theme(self):
        """Apply enhanced dark theme with gradients and animations"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2d2d2d);
                color: #ffffff;
            }

            QTabWidget::pane {
                border: 1px solid #444;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #1e1e1e);
                border-radius: 8px;
            }

            QTabWidget::tab-bar {
                alignment: center;
            }

            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d3d3d, stop:1 #2d2d2d);
                border: 1px solid #555;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #ccc;
                font-weight: bold;
                min-width: 100px;
            }

            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                color: white;
                border-bottom: none;
            }

            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4d4d4d, stop:1 #3d3d3d);
                color: white;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2d2d2d, stop:1 #1e1e1e);
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #0078d4;
                font-size: 14px;
            }

            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                min-height: 20px;
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #106ebe, stop:1 #0078d4);
            }

            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #005a9e, stop:1 #004578);
            }

            QPushButton:disabled {
                background: #555;
                color: #999;
            }

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d3d3d, stop:1 #2d2d2d);
                border: 2px solid #555;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                min-height: 20px;
            }

            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #0078d4;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4d4d4d, stop:1 #3d3d3d);
            }

            QTextEdit {
                background: #000;
                color: #00ff00;
                border: 2px solid #555;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border-radius: 6px;
                padding: 8px;
            }

            QLabel {
                color: #ffffff;
                font-size: 12px;
            }

            QProgressBar {
                border: 2px solid #555;
                border-radius: 6px;
                text-align: center;
                background: #2d2d2d;
                color: white;
                font-weight: bold;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078d4, stop:1 #00ff00);
                border-radius: 4px;
            }

            QTableWidget {
                background: #2d2d2d;
                alternate-background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 6px;
                gridline-color: #555;
                color: white;
            }

            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #555;
            }

            QTableWidget::item:selected {
                background: #0078d4;
            }

            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4d4d4d, stop:1 #3d3d3d);
                color: white;
                padding: 8px;
                border: 1px solid #555;
                font-weight: bold;
            }

            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background: #0078d4;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: #106ebe;
            }

            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d3d3d, stop:1 #2d2d2d);
                border: none;
                spacing: 3px;
                padding: 5px;
            }

            QToolButton {
                background: transparent;
                border: none;
                padding: 8px;
                border-radius: 4px;
                color: white;
            }

            QToolButton:hover {
                background: #0078d4;
            }

            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #1e1e1e);
                border-top: 1px solid #555;
                color: white;
                padding: 5px;
            }
        """)

    def create_toolbar(self):
        """Create enhanced toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        # Start/Stop actions
        self.start_action = QAction("▶️ Start Bot", self)
        self.start_action.triggered.connect(self.start_bot)
        toolbar.addAction(self.start_action)

        self.stop_action = QAction("⏹️ Stop Bot", self)
        self.stop_action.triggered.connect(self.stop_bot)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        # Quick settings
        toolbar.addWidget(QLabel("Quick Amount: "))
        self.quick_amount = QDoubleSpinBox()
        self.quick_amount.setRange(1.0, 1000.0)
        self.quick_amount.setValue(10.0)
        self.quick_amount.setPrefix("$")
        self.quick_amount.setMaximumWidth(100)
        toolbar.addWidget(self.quick_amount)

        toolbar.addSeparator()

        # Connection status
        self.connection_status = QLabel("🔴 Disconnected")
        toolbar.addWidget(self.connection_status)

        toolbar.addSeparator()

        # Fullscreen toggle
        fullscreen_action = QAction("🖥️ Fullscreen (F11)", self)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        toolbar.addAction(fullscreen_action)

        # Add stretch to push items to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Current time
        self.time_label = QLabel()
        self.update_time()
        toolbar.addWidget(self.time_label)

        # Timer for updating time
        time_timer = QTimer(self)
        time_timer.timeout.connect(self.update_time)
        time_timer.start(1000)

    def update_time(self):
        """Update current time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"🕐 {current_time}")

    def update_ui_animations(self):
        """Update UI animations and effects"""
        # This can be used for smooth animations and real-time updates
        pass

    def update_psychology_analysis(self, analysis_data: Dict):
        """Update the advanced psychology analysis display"""
        try:
            if hasattr(self, 'psychology_score_label'):
                # Update candlestick psychology score
                psychology_score = analysis_data.get('candle_psychology', {})
                buy_score = psychology_score.get('buy', 0.0)
                sell_score = psychology_score.get('sell', 0.0)
                max_score = max(buy_score, sell_score)

                self.psychology_score_label.setText(f"{max_score:.1%}")
                if max_score > 0.7:
                    self.psychology_score_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
                elif max_score > 0.5:
                    self.psychology_score_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 14px;")
                else:
                    self.psychology_score_label.setStyleSheet("color: #F44336; font-weight: bold; font-size: 14px;")

                # Update pattern strength
                pattern_data = analysis_data.get('pattern_analysis', {})
                pattern_buy = pattern_data.get('buy', 0.0)
                pattern_sell = pattern_data.get('sell', 0.0)
                pattern_strength = max(pattern_buy, pattern_sell)

                self.pattern_strength_label.setText(f"{pattern_strength:.1%}")
                if pattern_strength > 0.6:
                    self.pattern_strength_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
                else:
                    self.pattern_strength_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 14px;")

                # Update market structure
                structure_data = analysis_data.get('structure_analysis', {})
                structure_buy = structure_data.get('buy', 0.0)
                structure_sell = structure_data.get('sell', 0.0)

                if structure_buy > structure_sell and structure_buy > 0.5:
                    self.structure_shift_label.setText("BULLISH BOS")
                    self.structure_shift_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
                elif structure_sell > structure_buy and structure_sell > 0.5:
                    self.structure_shift_label.setText("BEARISH BOS")
                    self.structure_shift_label.setStyleSheet("color: #F44336; font-weight: bold; font-size: 14px;")
                else:
                    self.structure_shift_label.setText("NEUTRAL")
                    self.structure_shift_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 14px;")

                # Update smart money activity
                volume_data = analysis_data.get('volume_analysis', {})
                volume_buy = volume_data.get('buy', 0.0)
                volume_sell = volume_data.get('sell', 0.0)
                smart_money_activity = max(volume_buy, volume_sell)

                if smart_money_activity > 0.6:
                    self.smart_money_label.setText("ACTIVE")
                    self.smart_money_label.setStyleSheet("color: #9C27B0; font-weight: bold; font-size: 14px;")
                elif smart_money_activity > 0.3:
                    self.smart_money_label.setText("MODERATE")
                    self.smart_money_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 14px;")
                else:
                    self.smart_money_label.setText("INACTIVE")
                    self.smart_money_label.setStyleSheet("color: #757575; font-weight: bold; font-size: 14px;")

        except Exception as e:
            pass  # Silently handle any display errors

    def create_dashboard_tab(self):
        """Create a beautiful, focused dashboard with essential metrics only"""
        dashboard_widget = QWidget()
        main_layout = QVBoxLayout(dashboard_widget)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # 🎯 ESSENTIAL METRICS CARDS (Top Section)
        metrics_section = self.create_essential_metrics()
        main_layout.addWidget(metrics_section)

        # 📊 STATISTICS OVERVIEW (Middle Section)
        stats_section = self.create_statistics_section()
        main_layout.addWidget(stats_section)

        # 📈 ACTIVE TRADES MONITOR (Bottom Section)
        trades_section = self.create_active_trades_section()
        main_layout.addWidget(trades_section, 1)  # Takes remaining space

        self.tab_widget.addTab(dashboard_widget, "📊 Dashboard")

    def create_essential_metrics(self):
        """Create beautiful essential metrics cards"""
        metrics_widget = QWidget()
        metrics_layout = QGridLayout(metrics_widget)
        metrics_layout.setSpacing(20)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        # 💰 Account Balance Card
        self.balance_card = BeautifulMetricCard("Account Balance", "$0.00", "💰", "#4CAF50")
        metrics_layout.addWidget(self.balance_card, 0, 0)

        # 🎯 Win Rate Card
        self.win_rate_card = BeautifulMetricCard("Win Rate", "0.0%", "🎯", "#2196F3")
        metrics_layout.addWidget(self.win_rate_card, 0, 1)

        # 📊 Active Trades Card
        self.active_trades_card = BeautifulMetricCard("Active Trades", "0", "📊", "#FF9800")
        metrics_layout.addWidget(self.active_trades_card, 0, 2)

        # 📈 Total Profit Card
        self.profit_card = BeautifulMetricCard("Total Profit", "$0.00", "📈", "#9C27B0")
        metrics_layout.addWidget(self.profit_card, 0, 3)

        # Store cards for updates
        self.essential_cards = [self.balance_card, self.win_rate_card, self.active_trades_card, self.profit_card]

        return metrics_widget

    def create_statistics_section(self):
        """Create beautiful statistics overview with mini cards"""
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setSpacing(15)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        # Create mini statistics cards
        self.stats_cards = {}

        stats_items = [
            ("Total Trades", "0", "📊", "#2196F3"),
            ("Wins", "0", "✅", "#4CAF50"),
            ("Losses", "0", "❌", "#F44336"),
            ("Best Streak", "0", "🔥", "#FF9800"),
            ("Current Streak", "0", "⚡", "#9C27B0"),
            ("Avg Confidence", "0%", "🎯", "#00BCD4")
        ]

        for title, value, icon, color in stats_items:
            # Create mini card
            card = MiniStatCard(title, value, icon, color)
            stats_layout.addWidget(card)

            # Store reference
            key = title.lower().replace(" ", "_")
            self.stats_cards[key] = card

        return stats_widget

    def create_active_trades_section(self):
        """Create beautiful active trades monitor"""
        trades_widget = QWidget()
        trades_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2a2a2a);
                border-radius: 15px;
                border: 2px solid #333;
            }
        """)

        trades_layout = QVBoxLayout(trades_widget)
        trades_layout.setContentsMargins(25, 20, 25, 20)
        trades_layout.setSpacing(15)

        # Section title
        title = QLabel("📈 Active Trades Monitor")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 5px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        trades_layout.addWidget(title)

        # Active trades display
        self.active_trades_display = QLabel("No active trades")
        self.active_trades_display.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 14px;
                background: #1e1e1e;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }
        """)
        self.active_trades_display.setAlignment(Qt.AlignCenter)
        self.active_trades_display.setMinimumHeight(100)
        trades_layout.addWidget(self.active_trades_display)

        return trades_widget

    def create_trading_tab(self):
        """Create the trading configuration tab"""
        trading_widget = QWidget()
        layout = QHBoxLayout(trading_widget)

        # Left side - Configuration
        config_scroll = QScrollArea()
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # Bot Configuration
        bot_config_group = QGroupBox("🤖 Bot Configuration")
        bot_config_layout = QFormLayout(bot_config_group)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.30)
        self.confidence_spin.setSuffix("%")
        bot_config_layout.addRow("Confidence Threshold:", self.confidence_spin)

        self.trade_amount_spin = QDoubleSpinBox()
        self.trade_amount_spin.setRange(1.0, 1000.0)
        self.trade_amount_spin.setValue(10.0)
        self.trade_amount_spin.setPrefix("$")
        bot_config_layout.addRow("Trade Amount:", self.trade_amount_spin)

        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(1, 100)
        self.max_trades_spin.setValue(15)
        bot_config_layout.addRow("Max Trades/Hour:", self.max_trades_spin)

        self.demo_check = QCheckBox("Demo Mode")
        self.demo_check.setChecked(True)
        bot_config_layout.addRow("", self.demo_check)

        config_layout.addWidget(bot_config_group)

        # Martingale Configuration
        martingale_group = QGroupBox("🎲 Martingale Strategy")
        martingale_layout = QFormLayout(martingale_group)

        self.martingale_check = QCheckBox("Enable Martingale")
        self.martingale_check.setChecked(True)
        martingale_layout.addRow("", self.martingale_check)

        self.multiplier_spin = QDoubleSpinBox()
        self.multiplier_spin.setRange(1.1, 5.0)
        self.multiplier_spin.setSingleStep(0.1)
        self.multiplier_spin.setValue(2.2)
        self.multiplier_spin.setSuffix("x")
        martingale_layout.addRow("Multiplier:", self.multiplier_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 10)
        self.max_steps_spin.setValue(5)
        martingale_layout.addRow("Max Steps:", self.max_steps_spin)

        config_layout.addWidget(martingale_group)

        # Advanced Settings
        advanced_group = QGroupBox("⚙️ Advanced Settings")
        advanced_layout = QFormLayout(advanced_group)

        self.min_payout_spin = QDoubleSpinBox()
        self.min_payout_spin.setRange(70.0, 95.0)
        self.min_payout_spin.setValue(85.0)
        self.min_payout_spin.setSuffix("%")
        advanced_layout.addRow("Min Payout:", self.min_payout_spin)

        self.analysis_mode = QComboBox()
        self.analysis_mode.addItems(["Standard", "Conservative", "Aggressive"])
        self.analysis_mode.setCurrentText("Standard")
        advanced_layout.addRow("Analysis Mode:", self.analysis_mode)

        # Enhanced Expiry Time Selection (60s/90s only)
        expiry_label = QLabel("🎯 Enhanced Expiry Time:")
        expiry_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        advanced_layout.addRow(expiry_label)

        self.expiry_60_radio = QRadioButton("⚡ 60 seconds (Fast Scalping)")
        self.expiry_90_radio = QRadioButton("🎯 90 seconds (Balanced Accuracy)")
        self.expiry_60_radio.setChecked(True)  # Default to 60s

        # Style the radio buttons
        radio_style = """
            QRadioButton {
                color: #ffffff;
                font-size: 12px;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
            }
            QRadioButton::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #ffffff;
                border-radius: 8px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #555555;
                border: 2px solid #888888;
                border-radius: 8px;
            }
        """
        self.expiry_60_radio.setStyleSheet(radio_style)
        self.expiry_90_radio.setStyleSheet(radio_style)

        advanced_layout.addRow("", self.expiry_60_radio)
        advanced_layout.addRow("", self.expiry_90_radio)

        # Add explanation
        expiry_info = QLabel("🚀 Optimized for maximum accuracy\n⚡ 60s: Quick profits, higher frequency\n🎯 90s: Better confirmation, lower risk")
        expiry_info.setStyleSheet("color: #888; font-size: 10px; padding: 5px;")
        advanced_layout.addRow(expiry_info)

        config_layout.addWidget(advanced_group)

        # Control Buttons
        button_group = QGroupBox("🎮 Controls")
        button_layout = QVBoxLayout(button_group)

        self.start_button = AnimatedButton("🚀 Start Trading Bot")
        self.start_button.clicked.connect(self.start_bot)
        button_layout.addWidget(self.start_button)

        self.stop_button = AnimatedButton("⏹️ Stop Trading Bot")
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.pause_button = AnimatedButton("⏸️ Pause Bot")
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)

        config_layout.addWidget(button_group)
        config_layout.addStretch()

        config_scroll.setWidget(config_widget)
        config_scroll.setWidgetResizable(True)
        config_scroll.setMaximumWidth(400)
        layout.addWidget(config_scroll)

        # Right side - Asset Monitor
        asset_group = QGroupBox("📊 Asset Monitor")
        asset_layout = QVBoxLayout(asset_group)

        # Asset table
        self.asset_table = QTableWidget()
        self.asset_table.setColumnCount(4)
        self.asset_table.setHorizontalHeaderLabels(["Asset", "Payout", "Signal", "Confidence"])
        self.asset_table.horizontalHeader().setStretchLastSection(True)
        self.asset_table.setAlternatingRowColors(True)
        self.asset_table.setSelectionBehavior(QTableWidget.SelectRows)
        asset_layout.addWidget(self.asset_table)

        layout.addWidget(asset_group)

        self.tab_widget.addTab(trading_widget, "⚙️ Trading")

    def create_analytics_tab(self):
        """Create the analytics and performance tab"""
        analytics_widget = QWidget()
        layout = QVBoxLayout(analytics_widget)

        # Performance summary
        summary_group = QGroupBox("📈 Performance Summary")
        summary_layout = QGridLayout(summary_group)

        # Performance metrics
        metrics = [
            ("Total Trades", "0", "📊"),
            ("Win Rate", "0.0%", "🎯"),
            ("Best Streak", "0", "🔥"),
            ("Worst Streak", "0", "❄️"),
            ("Avg Confidence", "0.0%", "🧠"),
            ("Total Profit", "$0.00", "💰")
        ]

        self.analytics_cards = {}
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]

        for i, (title, value, icon) in enumerate(metrics):
            color = colors[i % len(colors)]
            card = BeautifulMetricCard(title, value, icon, color)
            self.analytics_cards[title.lower().replace(" ", "_")] = card
            row, col = divmod(i, 3)
            summary_layout.addWidget(card, row, col)

        layout.addWidget(summary_group)

        # Trade history table
        history_group = QGroupBox("📋 Trade History")
        history_layout = QVBoxLayout(history_group)

        self.trade_history_table = QTableWidget()
        self.trade_history_table.setColumnCount(7)
        self.trade_history_table.setHorizontalHeaderLabels([
            "Time", "Asset", "Direction", "Amount", "Result", "Profit", "Confidence"
        ])
        self.trade_history_table.horizontalHeader().setStretchLastSection(True)
        self.trade_history_table.setAlternatingRowColors(True)
        history_layout.addWidget(self.trade_history_table)

        layout.addWidget(history_group)

        self.tab_widget.addTab(analytics_widget, "📈 Analytics")

    def create_settings_tab(self):
        """Create the settings and preferences tab"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)

        # Account settings
        account_group = QGroupBox("👤 Account Settings")
        account_layout = QFormLayout(account_group)

        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("Enter your Quotex email")
        account_layout.addRow("Email:", self.email_edit)

        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("Enter your password")
        account_layout.addRow("Password:", self.password_edit)

        layout.addWidget(account_group)

        # Notification settings
        notification_group = QGroupBox("🔔 Notifications")
        notification_layout = QFormLayout(notification_group)

        self.sound_notifications = QCheckBox("Sound Notifications")
        self.sound_notifications.setChecked(True)
        notification_layout.addRow("", self.sound_notifications)

        self.system_tray_notifications = QCheckBox("System Tray Notifications")
        self.system_tray_notifications.setChecked(True)
        notification_layout.addRow("", self.system_tray_notifications)

        layout.addWidget(notification_group)

        # Risk management
        risk_group = QGroupBox("⚠️ Risk Management")
        risk_layout = QFormLayout(risk_group)

        self.daily_loss_limit = QDoubleSpinBox()
        self.daily_loss_limit.setRange(0, 10000)
        self.daily_loss_limit.setValue(100)
        self.daily_loss_limit.setPrefix("$")
        risk_layout.addRow("Daily Loss Limit:", self.daily_loss_limit)

        self.daily_profit_target = QDoubleSpinBox()
        self.daily_profit_target.setRange(0, 10000)
        self.daily_profit_target.setValue(50)
        self.daily_profit_target.setPrefix("$")
        risk_layout.addRow("Daily Profit Target:", self.daily_profit_target)

        layout.addWidget(risk_group)

        # Save/Load settings
        settings_buttons_layout = QHBoxLayout()

        save_settings_btn = AnimatedButton("💾 Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        settings_buttons_layout.addWidget(save_settings_btn)

        load_settings_btn = AnimatedButton("📂 Load Settings")
        load_settings_btn.clicked.connect(self.load_settings)
        settings_buttons_layout.addWidget(load_settings_btn)

        reset_settings_btn = AnimatedButton("🔄 Reset to Defaults")
        reset_settings_btn.clicked.connect(self.reset_settings)
        settings_buttons_layout.addWidget(reset_settings_btn)

        layout.addLayout(settings_buttons_layout)
        layout.addStretch()

        self.tab_widget.addTab(settings_widget, "⚙️ Settings")

    def create_logs_tab(self):
        """Create the logs and debugging tab"""
        logs_widget = QWidget()
        layout = QVBoxLayout(logs_widget)

        # Log controls
        controls_layout = QHBoxLayout()

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["All", "Info", "Warning", "Error"])
        controls_layout.addWidget(QLabel("Log Level:"))
        controls_layout.addWidget(self.log_level_combo)

        controls_layout.addStretch()

        clear_logs_btn = AnimatedButton("🗑️ Clear Logs")
        clear_logs_btn.clicked.connect(self.clear_log)
        controls_layout.addWidget(clear_logs_btn)

        export_logs_btn = AnimatedButton("📤 Export Logs")
        export_logs_btn.clicked.connect(self.export_logs)
        controls_layout.addWidget(export_logs_btn)

        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_display)

        self.tab_widget.addTab(logs_widget, "📝 Logs")

    def create_enhanced_status_bar(self):
        """Create enhanced status bar with multiple indicators"""
        status_bar = self.statusBar()

        # Main status
        self.main_status = QLabel("Ready")
        status_bar.addWidget(self.main_status)

        # Connection indicator
        self.connection_indicator = QLabel("🔴 Disconnected")
        status_bar.addPermanentWidget(self.connection_indicator)

        # Balance indicator
        self.balance_indicator = QLabel("Balance: $0.00")
        status_bar.addPermanentWidget(self.balance_indicator)

        # Trade count indicator
        self.trade_count_indicator = QLabel("Trades: 0")
        status_bar.addPermanentWidget(self.trade_count_indicator)

    def save_settings(self):
        """Save current enhanced settings"""
        self.settings.setValue("confidence_threshold", self.confidence_spin.value())
        self.settings.setValue("trade_amount", self.trade_amount_spin.value())
        self.settings.setValue("max_trades", self.max_trades_spin.value())
        self.settings.setValue("demo_mode", self.demo_check.isChecked())
        self.settings.setValue("martingale_enabled", self.martingale_check.isChecked())
        self.settings.setValue("martingale_multiplier", self.multiplier_spin.value())
        self.settings.setValue("max_martingale_steps", self.max_steps_spin.value())
        self.settings.setValue("min_payout", self.min_payout_spin.value())
        self.settings.setValue("analysis_mode", self.analysis_mode.currentText())
        self.settings.setValue("enhanced_expiry_60s", self.expiry_60_radio.isChecked())  # 🚀 Enhanced expiry

        QMessageBox.information(self, "Settings", "Enhanced settings saved successfully!")

    def load_settings(self):
        """Load saved enhanced settings"""
        self.confidence_spin.setValue(self.settings.value("confidence_threshold", 0.30, float))
        self.trade_amount_spin.setValue(self.settings.value("trade_amount", 10.0, float))
        self.max_trades_spin.setValue(self.settings.value("max_trades", 15, int))
        self.demo_check.setChecked(self.settings.value("demo_mode", True, bool))
        self.martingale_check.setChecked(self.settings.value("martingale_enabled", True, bool))
        self.multiplier_spin.setValue(self.settings.value("martingale_multiplier", 2.2, float))
        self.max_steps_spin.setValue(self.settings.value("max_martingale_steps", 5, int))
        self.min_payout_spin.setValue(self.settings.value("min_payout", 85.0, float))
        self.analysis_mode.setCurrentText(self.settings.value("analysis_mode", "Standard", str))

        # Load enhanced expiry settings
        expiry_60s = self.settings.value("enhanced_expiry_60s", True, bool)
        self.expiry_60_radio.setChecked(expiry_60s)
        self.expiry_90_radio.setChecked(not expiry_60s)

    def reset_settings(self):
        """Reset enhanced settings to defaults"""
        reply = QMessageBox.question(self, "Reset Enhanced Settings",
                                   "Are you sure you want to reset all enhanced settings to defaults?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.confidence_spin.setValue(0.30)
            self.trade_amount_spin.setValue(10.0)
            self.max_trades_spin.setValue(15)
            self.demo_check.setChecked(True)
            self.martingale_check.setChecked(True)
            self.multiplier_spin.setValue(2.2)
            self.max_steps_spin.setValue(5)
            self.min_payout_spin.setValue(85.0)
            self.analysis_mode.setCurrentText("Standard")
            # Reset enhanced expiry to 60s default
            self.expiry_60_radio.setChecked(True)
            self.expiry_90_radio.setChecked(False)

    def export_logs(self):
        """Export logs to file"""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Export Logs",
                                                "trading_logs.txt", "Text Files (*.txt)")
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_display.toPlainText())
            QMessageBox.information(self, "Export", f"Logs exported to {filename}")

    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Bot Configuration
        config_group = QGroupBox("🤖 Bot Configuration")
        config_layout = QFormLayout(config_group)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.30)
        self.confidence_spin.setSuffix("%")
        config_layout.addRow("Confidence Threshold:", self.confidence_spin)

        self.trade_amount_spin = QDoubleSpinBox()
        self.trade_amount_spin.setRange(1.0, 1000.0)
        self.trade_amount_spin.setValue(10.0)
        self.trade_amount_spin.setPrefix("$")
        config_layout.addRow("Trade Amount:", self.trade_amount_spin)

        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(1, 100)
        self.max_trades_spin.setValue(15)
        config_layout.addRow("Max Trades/Hour:", self.max_trades_spin)

        self.demo_check = QCheckBox("Demo Mode")
        self.demo_check.setChecked(True)
        config_layout.addRow("", self.demo_check)

        layout.addWidget(config_group)

        # Martingale Configuration
        martingale_group = QGroupBox("🎲 Martingale Strategy")
        martingale_layout = QFormLayout(martingale_group)

        self.martingale_check = QCheckBox("Enable Martingale")
        self.martingale_check.setChecked(True)
        martingale_layout.addRow("", self.martingale_check)

        self.multiplier_spin = QDoubleSpinBox()
        self.multiplier_spin.setRange(1.1, 5.0)
        self.multiplier_spin.setSingleStep(0.1)
        self.multiplier_spin.setValue(2.2)
        self.multiplier_spin.setSuffix("x")
        martingale_layout.addRow("Multiplier:", self.multiplier_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 10)
        self.max_steps_spin.setValue(5)
        martingale_layout.addRow("Max Steps:", self.max_steps_spin)

        layout.addWidget(martingale_group)

        # Control Buttons
        button_group = QGroupBox("🎮 Controls")
        button_layout = QVBoxLayout(button_group)

        self.start_button = QPushButton("🚀 Start Bot")
        self.start_button.clicked.connect(self.start_bot)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("⏹️ Stop Bot")
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.clear_log_button = QPushButton("🗑️ Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        button_layout.addWidget(self.clear_log_button)

        layout.addWidget(button_group)

        # Spacer
        layout.addStretch()

        return panel

    def create_monitoring_panel(self):
        """Create the monitoring panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Status and Balance
        status_group = QGroupBox("📊 Status & Balance")
        status_layout = QGridLayout(status_group)

        status_layout.addWidget(QLabel("Status:"), 0, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        status_layout.addWidget(self.status_label, 0, 1)

        status_layout.addWidget(QLabel("Balance:"), 1, 0)
        self.balance_label = QLabel("$0.00")
        self.balance_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 16px;")
        status_layout.addWidget(self.balance_label, 1, 1)

        layout.addWidget(status_group)

        # Trading Statistics
        stats_group = QGroupBox("📈 Trading Statistics")
        stats_layout = QGridLayout(stats_group)

        stats_layout.addWidget(QLabel("Total Trades:"), 0, 0)
        self.total_trades_label = QLabel("0")
        stats_layout.addWidget(self.total_trades_label, 0, 1)

        stats_layout.addWidget(QLabel("Wins:"), 0, 2)
        self.wins_label = QLabel("0")
        self.wins_label.setStyleSheet("color: #00ff00;")
        stats_layout.addWidget(self.wins_label, 0, 3)

        stats_layout.addWidget(QLabel("Losses:"), 1, 0)
        self.losses_label = QLabel("0")
        self.losses_label.setStyleSheet("color: #ff0000;")
        stats_layout.addWidget(self.losses_label, 1, 1)

        stats_layout.addWidget(QLabel("Win Rate:"), 1, 2)
        self.win_rate_label = QLabel("0.0%")
        stats_layout.addWidget(self.win_rate_label, 1, 3)

        stats_layout.addWidget(QLabel("Net P&L:"), 2, 0)
        self.profit_label = QLabel("$0.00")
        stats_layout.addWidget(self.profit_label, 2, 1)

        stats_layout.addWidget(QLabel("Martingale Step:"), 2, 2)
        self.martingale_label = QLabel("0")
        stats_layout.addWidget(self.martingale_label, 2, 3)

        layout.addWidget(stats_group)

        # Active Trade Monitor
        trade_group = QGroupBox("🎯 Active Trade")
        trade_layout = QGridLayout(trade_group)

        trade_layout.addWidget(QLabel("Status:"), 0, 0)
        self.trade_status_label = QLabel("No Active Trade")
        self.trade_status_label.setStyleSheet("color: #888888;")
        trade_layout.addWidget(self.trade_status_label, 0, 1)

        trade_layout.addWidget(QLabel("Asset:"), 1, 0)
        self.trade_asset_label = QLabel("-")
        trade_layout.addWidget(self.trade_asset_label, 1, 1)

        trade_layout.addWidget(QLabel("Direction:"), 1, 2)
        self.trade_direction_label = QLabel("-")
        trade_layout.addWidget(self.trade_direction_label, 1, 3)

        trade_layout.addWidget(QLabel("Amount:"), 2, 0)
        self.trade_amount_label = QLabel("-")
        trade_layout.addWidget(self.trade_amount_label, 2, 1)

        # Time Left removed - now handled in beautiful dashboard

        layout.addWidget(trade_group)

        # Log Display
        log_group = QGroupBox("📝 Bot Log")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        # Note: QTextEdit doesn't have setMaximumBlockCount, we'll manage log size manually
        log_layout.addWidget(self.log_display)

        layout.addWidget(log_group)

        return panel

    # Enhanced methods for the new GUI
    def start_bot(self):
        """Start the trading bot with enhanced configuration"""
        if self.bot_worker and self.bot_worker.isRunning():
            return

        # Get enhanced configuration from GUI
        selected_expiry = 60 if self.expiry_60_radio.isChecked() else 90

        bot_config = {
            'confidence_threshold': self.confidence_spin.value() / 100,
            'trade_amount': self.trade_amount_spin.value(),
            'max_trades_per_hour': self.max_trades_spin.value(),
            'demo_mode': self.demo_check.isChecked(),
            'use_martingale': self.martingale_check.isChecked(),
            'martingale_multiplier': self.multiplier_spin.value(),
            'max_martingale_steps': self.max_steps_spin.value(),
            'min_payout': self.min_payout_spin.value(),
            'enhanced_expiry_time': selected_expiry,  # 🚀 Enhanced expiry selection
            'enhanced_mode': True,  # 🚀 Enable enhanced features
            'analysis_mode': self.analysis_mode.currentText()
        }

        # Create and start worker thread
        self.bot_worker = BotWorker(bot_config)
        self.bot_worker.log_signal.connect(self.add_log)
        self.bot_worker.status_signal.connect(self.update_status)
        self.bot_worker.balance_signal.connect(self.update_balance)
        self.bot_worker.stats_signal.connect(self.update_stats)
        self.bot_worker.trade_started_signal.connect(self.on_trade_started)
        self.bot_worker.trade_completed_signal.connect(self.on_trade_completed)
        self.bot_worker.psychology_analysis_signal.connect(self.update_psychology_analysis)
        self.bot_worker.finished.connect(self.on_bot_finished)

        self.bot_worker.start()

        # Start continuous balance updates when bot starts
        self.balance_update_timer.start(10000)  # Update balance every 10 seconds

        # Update UI
        self.start_button.setEnabled(False)
        self.start_action.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.connection_status.setText("🟡 Connecting...")
        self.add_log("🚀 Starting enhanced trading bot...")
        self.add_log("💰 Continuous balance updates enabled (every 10 seconds)")

    def fetch_current_balance(self):
        """Fetch current balance from bot worker"""
        if self.bot_worker and self.bot_worker.bot and self.bot_worker.running:
            try:
                # Request balance update from worker thread
                self.bot_worker.request_balance_update()
            except Exception as e:
                self.add_log(f"⚠️ Balance fetch error: {str(e)}")

    def stop_bot(self):
        """Stop the trading bot"""
        if self.bot_worker and self.bot_worker.isRunning():
            self.add_log("⏹️ Stopping trading bot...")
            self.bot_worker.stop_bot()
            self.bot_worker.wait(5000)

        # Stop balance update timer
        self.balance_update_timer.stop()
        self.add_log("💰 Continuous balance updates stopped")

        self.on_bot_finished()

    def on_bot_finished(self):
        """Handle bot finished with enhanced UI updates"""
        self.start_button.setEnabled(True)
        self.start_action.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.connection_status.setText("🔴 Disconnected")
        self.connection_indicator.setText("🔴 Disconnected")

        # Stop all timers
        self.trade_timer.stop()
        self.balance_update_timer.stop()

        # Reset display
        self.active_trade = None
        self.reset_trade_display()

    def add_log(self, message: str):
        """Add message to log display with enhanced formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_display.append(formatted_message)

        # Enhanced log management
        document = self.log_display.document()
        if document.blockCount() > 1000:
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 200)
            cursor.removeSelectedText()

        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_status(self, status: str):
        """Update status with enhanced indicators"""
        self.main_status.setText(status)

        # Update connection status
        if status == "Running":
            self.connection_status.setText("🟢 Connected")
            self.connection_indicator.setText("🟢 Connected")
        elif status == "Stopped":
            self.connection_status.setText("🔴 Disconnected")
            self.connection_indicator.setText("🔴 Disconnected")
        elif status == "Connecting...":
            self.connection_status.setText("🟡 Connecting...")
            self.connection_indicator.setText("🟡 Connecting...")

    def update_balance(self, balance: float):
        """Update balance display with beautiful animations and proper formatting"""
        try:
            # Ensure balance is a proper float and handle edge cases
            if isinstance(balance, str):
                balance = float(balance)
            elif balance is None:
                balance = 0.0

            # Debug: Log only the first few balance updates to check for issues
            if not hasattr(self, '_balance_debug_count'):
                self._balance_debug_count = 0

            if self._balance_debug_count < 5:  # Log first 5 updates for debugging
                self.add_log(f"💰 Balance debug #{self._balance_debug_count + 1}: Raw={balance}, Type={type(balance)}, Formatted={balance:.10f}")
                self._balance_debug_count += 1

            # Ensure we have a valid positive number
            if balance < 0:
                self.add_log(f"⚠️ Warning: Negative balance received: {balance}")
                balance = 0.0

            # Format balance with proper number formatting for large amounts
            if balance >= 10000:
                # For amounts >= 10,000, use comma separators
                formatted_balance = f"${balance:,.2f}"
            else:
                # For smaller amounts, use standard formatting
                formatted_balance = f"${balance:.2f}"

            # Update balance card
            if hasattr(self, 'balance_card'):
                self.balance_card.update_value(formatted_balance, animated=True)

            # Update balance indicator
            if hasattr(self, 'balance_indicator'):
                self.balance_indicator.setText(f"Balance: {formatted_balance}")

            # Update balance history for performance tracking
            self.performance_data['balance_history'].append({
                'timestamp': time.time(),
                'balance': balance
            })

            # Keep only last 100 balance points
            if len(self.performance_data['balance_history']) > 100:
                self.performance_data['balance_history'] = self.performance_data['balance_history'][-100:]

        except Exception as e:
            self.add_log(f"❌ Balance update error: {str(e)}")
            # Fallback display
            if hasattr(self, 'balance_card'):
                self.balance_card.update_value("$0.00", animated=False)
            if hasattr(self, 'balance_indicator'):
                self.balance_indicator.setText("Balance: $0.00")

    def update_stats(self, stats: Dict):
        """Update beautiful dashboard statistics"""
        # Update essential metric cards
        if hasattr(self, 'win_rate_card'):
            self.win_rate_card.update_value(f"{stats['win_rate']:.1f}%", animated=True)

        if hasattr(self, 'profit_card'):
            self.profit_card.update_value(f"${stats['net_profit']:.2f}", animated=True)

        if hasattr(self, 'active_trades_card'):
            # Count active trades (for now, use pending martingale as proxy)
            active_count = 1 if stats.get('pending_martingale', False) else 0
            self.active_trades_card.update_value(str(active_count), animated=True)

        # Update beautiful mini statistics cards
        if hasattr(self, 'stats_cards'):
            self.stats_cards['total_trades'].update_value(str(stats['total_trades']), animated=True)
            self.stats_cards['wins'].update_value(str(stats['wins']), animated=True)
            self.stats_cards['losses'].update_value(str(stats['losses']), animated=True)

            # Calculate streaks (simplified)
            best_streak = max(stats['wins'], stats['losses'])
            current_streak = 1  # Simplified for now
            avg_confidence = 75.0  # Placeholder

            self.stats_cards['best_streak'].update_value(str(best_streak), animated=True)
            self.stats_cards['current_streak'].update_value(str(current_streak), animated=True)
            self.stats_cards['avg_confidence'].update_value(f"{avg_confidence:.0f}%", animated=True)

        # Update active trades display
        if hasattr(self, 'active_trades_display'):
            if stats.get('pending_martingale', False):
                self.active_trades_display.setText("🎯 Martingale trade pending...")
                self.active_trades_display.setStyleSheet("""
                    QLabel {
                        color: #FF9800;
                        font-size: 14px;
                        font-weight: bold;
                        background: #1e1e1e;
                        border: 1px solid #FF9800;
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                    }
                """)
            else:
                self.active_trades_display.setText("No active trades")
                self.active_trades_display.setStyleSheet("""
                    QLabel {
                        color: #888;
                        font-size: 14px;
                        background: #1e1e1e;
                        border: 1px solid #444;
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                    }
                """)

    def on_trade_started(self, trade_info: Dict):
        """Handle trade started with beautiful dashboard updates"""
        self.active_trade = trade_info.copy()
        self.active_trade['start_time'] = time.time()

        # Update active trades card
        if hasattr(self, 'active_trades_card'):
            self.active_trades_card.update_value("1", animated=True)

        # Update active trades display
        if hasattr(self, 'active_trades_display'):
            direction = trade_info['direction'].upper()
            direction_emoji = "📈" if direction == "CALL" else "📉"

            trade_text = f"""
            {direction_emoji} ACTIVE TRADE

            Asset: {trade_info['asset']}
            Direction: {direction}
            Amount: ${trade_info['amount']:.2f}
            Duration: {trade_info['duration']}s
            Confidence: {trade_info.get('confidence', 0):.1f}%
            """

            self.active_trades_display.setText(trade_text)
            self.active_trades_display.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 14px;
                    font-weight: bold;
                    background: #1e1e1e;
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }
            """)

        # Start countdown timer if it exists
        if hasattr(self, 'trade_timer'):
            self.trade_timer.start(1000)

        # Log trade start
        direction = trade_info['direction'].upper()
        self.add_log(f"🚀 TRADE STARTED: {direction} {trade_info['asset']} - ${trade_info['amount']:.2f}")

    def on_trade_completed(self, completion_info: Dict):
        """Handle trade completed with beautiful dashboard updates"""
        self.active_trade = None

        # Stop timer if it exists
        if hasattr(self, 'trade_timer'):
            self.trade_timer.stop()

        # Update active trades card
        if hasattr(self, 'active_trades_card'):
            self.active_trades_card.update_value("0", animated=True)

        # Update active trades display with result
        if hasattr(self, 'active_trades_display'):
            result = completion_info['result'].upper()

            if result == "WIN":
                result_emoji = "🎉"
                result_color = "#4CAF50"
                result_text = "TRADE WON!"
            else:
                result_emoji = "😞"
                result_color = "#F44336"
                result_text = "TRADE LOST"

            completion_text = f"""
            {result_emoji} {result_text}

            Asset: {completion_info['asset']}
            Result: {result}
            """

            self.active_trades_display.setText(completion_text)
            self.active_trades_display.setStyleSheet(f"""
                QLabel {{
                    color: {result_color};
                    font-size: 14px;
                    font-weight: bold;
                    background: #1e1e1e;
                    border: 2px solid {result_color};
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }}
            """)

        # Log trade completion
        result = completion_info['result'].upper()
        self.add_log(f"✅ TRADE COMPLETED: {result} - {completion_info['asset']}")

        # Reset after 3 seconds
        QTimer.singleShot(3000, self.reset_trade_display)

    def reset_trade_display(self):
        """Reset beautiful dashboard to default state"""
        # Reset active trades display
        if hasattr(self, 'active_trades_display'):
            self.active_trades_display.setText("No active trades")
            self.active_trades_display.setStyleSheet("""
                QLabel {
                    color: #888;
                    font-size: 14px;
                    background: #1e1e1e;
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }
            """)

        # Reset active trades card
        if hasattr(self, 'active_trades_card'):
            self.active_trades_card.update_value("0", animated=True)

    def update_trade_timer(self):
        """Update the trade countdown timer in the beautiful dashboard"""
        if not self.active_trade:
            return

        # Calculate remaining time
        elapsed = time.time() - self.active_trade['start_time']
        remaining = max(0, self.active_trade['duration'] - elapsed)

        # Update active trades display with countdown
        if hasattr(self, 'active_trades_display'):
            direction = self.active_trade['direction'].upper()
            direction_emoji = "📈" if direction == "CALL" else "📉"

            if remaining > 0:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)

                if minutes > 0:
                    time_text = f"{minutes}:{seconds:02d}"
                else:
                    time_text = f"{seconds}s"

                # Enhanced color coding
                if remaining > 30:
                    border_color = "#4CAF50"  # Green
                elif remaining > 10:
                    border_color = "#FF9800"  # Orange
                else:
                    border_color = "#F44336"  # Red

                trade_text = f"""
                {direction_emoji} ACTIVE TRADE - {time_text}

                Asset: {self.active_trade['asset']}
                Direction: {direction}
                Amount: ${self.active_trade['amount']:.2f}
                Confidence: {self.active_trade.get('confidence', 0):.1f}%
                """

                self.active_trades_display.setText(trade_text)
                self.active_trades_display.setStyleSheet(f"""
                    QLabel {{
                        color: {border_color};
                        font-size: 14px;
                        font-weight: bold;
                        background: #1e1e1e;
                        border: 2px solid {border_color};
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                    }}
                """)
            else:
                # Trade expired
                self.active_trades_display.setText(f"""
                {direction_emoji} TRADE EXPIRED

                Asset: {self.active_trade['asset']}
                Direction: {direction}
                Status: WAITING FOR RESULT...
                """)
                self.active_trades_display.setStyleSheet("""
                    QLabel {
                        color: #F44336;
                        font-size: 14px;
                        font-weight: bold;
                        background: #1e1e1e;
                        border: 2px solid #F44336;
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                    }
                """)

    def clear_log(self):
        """Clear the log display"""
        self.log_display.clear()
        self.add_log("� Log cleared")

    def closeEvent(self, event):
        """Handle window close event with enhanced cleanup"""
        if self.bot_worker and self.bot_worker.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Trading bot is still running. Stop it before closing?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self.stop_bot()
                event.accept()
            else:
                event.ignore()
        else:
            # Save settings before closing
            self.save_settings()
            event.accept()


def main():
    """Main function to run the enhanced GUI"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Quotex Trading Bot Professional")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("Trading Bot Solutions")

    # Create and show main window
    window = EnhancedTradingBotGUI()
    window.show()

    # Add enhanced welcome message
    window.add_log("🤖 Quotex Trading Bot Started - Professional Edition")
    window.add_log("🔥 ADVANCED FEATURES:")
    window.add_log("   � Advanced Technical Analysis - Professional trading signals")
    window.add_log("   📊 Advanced Price Action Patterns - Professional trader level")
    window.add_log("   🏗️ Market Structure Analysis - Smart money concepts")
    window.add_log("   � Liquidity Analysis - Institutional footprints")
    window.add_log("   🧠 AI-Enhanced Ensemble System - Maximum accuracy")
    window.add_log("⚡ OPTIMIZED TIMEFRAMES:")
    window.add_log("   ⚡ 60 seconds - Fast scalping with high frequency")
    window.add_log("   🎯 90 seconds - Balanced accuracy with confirmation")
    window.add_log("🎯 PREDICTION CAPABILITIES:")
    window.add_log("   📈 Next 1-2 minute direction prediction")
    window.add_log("   �️ Individual candle psychology reading")
    window.add_log("   🔍 Pattern formation detection")
    window.add_log("   📊 Market structure shift identification")
    window.add_log("🚀 Ready for professional candlestick psychology trading!")
    window.add_log("🖥️ RESPONSIVE DESIGN:")
    window.add_log("   📱 Optimized for all screen sizes and resolutions")
    window.add_log("   🖥️ F11 for fullscreen, Ctrl+M to maximize")
    window.add_log("   📐 Auto-adjusting layouts for optimal viewing")
    window.add_log("   🔄 Responsive metrics and charts")

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
