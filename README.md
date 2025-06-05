# 🤖 Quotex Trading Bot

Advanced Binary Options Trading Bot with Intelligent Analysis and Neural Network Predictions.

## 🌟 Features

- **Intelligent Trading**: Neural network-based price action analysis
- **Risk Management**: Built-in Martingale strategy with configurable parameters
- **Real-time Monitoring**: Live performance tracking and statistics
- **Modern GUI**: PyQt5-based interface with real-time charts and metrics
- **Multiple Analysis Modes**: Standard, Conservative, and Aggressive trading strategies
- **Asset Discovery**: Automatic detection of high-payout trading opportunities
- **Session Management**: Automatic login and connection handling
- **Demo & Live Trading**: Support for both demo and live account trading

## 📋 Prerequisites

- Python 3.10 (recommended for TensorFlow compatibility)
- Windows 10/11
- Visual C++ Build Tools
- Chrome Browser (for web automation)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quotex-trading-bot.git
cd quotex-trading-bot
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv --python=python3.10

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install TA-Lib:
   - Download the appropriate wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - Install using: `pip install TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl`

## ⚙️ Configuration

1. Create a `.env` file in the project root:
```env
QUOTEX_EMAIL=your_email@example.com
QUOTEX_PASSWORD=your_password
```

2. Adjust trading parameters in `settings/config.json`:
```json
{
    "confidence_threshold": 0.50,
    "min_payout": 85.0,
    "trade_amount": 10.0,
    "max_trades_per_hour": 20,
    "demo_mode": true,
    "use_martingale": true,
    "martingale_multiplier": 2.2,
    "max_martingale_steps": 5
}
```

## 🎮 Usage

1. Start the GUI:
```bash
python trading_bot_gui.py
```

2. Or run in console mode:
```bash
python bot.py
```

### GUI Features
- Dashboard with real-time performance metrics
- Trading controls and settings
- Asset performance analysis
- Trade history and statistics
- System tray integration

### Trading Modes
- **Demo Mode**: Practice trading with virtual money
- **Live Mode**: Real trading with actual funds
- **Analysis Mode**: Market analysis without trading

## ⚠️ Risk Warning

Trading binary options involves significant risk of loss. This bot is for educational purposes only. Always:
- Start with demo trading
- Use proper risk management
- Never trade with money you cannot afford to lose
- Monitor the bot's performance regularly

## 📊 Performance Tracking

The bot maintains detailed statistics including:
- Win rate and ROI
- Profit/loss tracking
- Asset performance analysis
- Trade history
- Risk management metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Quotex API for trading platform integration
- TensorFlow and Keras for neural network capabilities
- PyQt5 for the graphical interface
- All contributors and users of this project

## 📞 Support

For support, please:
1. Check the [Issues](https://github.com/yourusername/quotex-trading-bot/issues) section
2. Create a new issue if needed
3. Join our community chat (if available)

---

**Disclaimer**: This software is for educational purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred through the use of this bot. 