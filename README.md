# Pump Trader Bot

A sophisticated trading bot for Solana tokens that monitors new token launches and executes trades based on configurable parameters and risk management rules.

## Features

### 1. Real-time Token Monitoring
- Uses PumpPortal WebSocket API for real-time token discovery
- Monitors new token launches and trending tokens
- Automatic fallback to test mode if API connection fails
- Configurable token discovery parameters

### 2. Token Analysis
- Liquidity checks
- Holder distribution analysis
- Buy/Sell tax verification
- Market cap validation
- Configurable thresholds for all parameters

### 3. Risk Management
- Maximum portfolio risk limits
- Position size management
- Stop-loss and take-profit automation
- Maximum open positions limit
- Portfolio value tracking
- Dynamic position sizing based on risk parameters

### 4. Trading Execution
- Integration with Jupiter DEX for optimal trading
- Slippage protection
- Automatic token account creation
- Transaction confirmation monitoring
- Support for both buy and sell operations

### 5. Position Management
- Real-time position monitoring
- Automatic stop-loss execution
- Take-profit target tracking
- Position tracking with entry prices and timestamps
- P&L calculation and logging

### 6. Test Mode
- Automatic fallback to test mode when API is unavailable
- Simulated token discovery
- Price simulation with configurable volatility
- Test token configuration via JSON file
- Realistic market behavior simulation

## Configuration

### Environment Variables
Create a `.env` file with the following parameters:

```env
# Solana Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
WALLET_PRIVATE_KEY=your_private_key_here

# Trading Parameters
MIN_SOL_BALANCE=0.1
MAX_TRADE_AMOUNT=0.5
MIN_LIQUIDITY=1000
MAX_HOLDER_PERCENTAGE=20
MIN_HOLDERS=50
MAX_TAX=10
MIN_MARKET_CAP=50000

# Risk Management
MAX_PORTFOLIO_RISK=2.0
MAX_POSITION_SIZE=5.0
STOP_LOSS_PERCENTAGE=10.0
TAKE_PROFIT_PERCENTAGE=30.0
MAX_OPEN_POSITIONS=5

# Trading Execution
SLIPPAGE_TOLERANCE=1.0

# API Configuration
PUMP_FUN_API_URL=https://api.pump.fun/v1
```

### Test Tokens Configuration
Create a `test_tokens.json` file for test mode:

```json
[
    {
        "address": "So11111111111111111111111111111111111111112",
        "name": "Test Token 1",
        "initial_price": 1.0,
        "volatility": 5.0
    },
    {
        "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "name": "Test Token 2",
        "initial_price": 1.0,
        "volatility": 2.0
    }
]
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- solana==0.30.2
- requests==2.31.0
- python-dotenv==1.0.0
- aiohttp==3.9.1
- asyncio==3.4.3
- web3==6.11.1
- base58==2.1.1
- typing-extensions==4.8.0
- dataclasses==0.6
- pydantic==2.5.2
- tenacity==8.2.3
- websockets>=9.0,<12.0

## Usage

1. Set up your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the bot:
   ```bash
   python pump_trader.py
   ```

## Logging

The bot maintains detailed logs in `trading.log` with the following information:
- Token discovery and analysis
- Trade execution details
- Position management updates
- Error and warning messages
- Portfolio value changes
- Risk management decisions

## Safety Features

1. **Risk Controls**
   - Maximum portfolio risk limits
   - Position size restrictions
   - Stop-loss protection
   - Maximum open positions limit

2. **Error Handling**
   - Automatic reconnection for WebSocket
   - Fallback to test mode on API failure
   - Transaction confirmation monitoring
   - Comprehensive error logging

3. **Market Protection**
   - Liquidity checks
   - Holder distribution analysis
   - Tax verification
   - Market cap validation

## Development

The bot is built with modularity in mind, making it easy to:
- Add new analysis criteria
- Modify risk management rules
- Implement additional trading strategies
- Add new data sources
- Customize logging and monitoring

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies carries significant risks. Always:
- Test thoroughly in test mode first
- Start with small amounts
- Monitor the bot's performance
- Keep your private keys secure
- Understand all risks involved 