import os
import asyncio
import aiohttp
import websockets
from dotenv import load_dotenv
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.system_program import TransferParams, transfer
from solders.pubkey import Pubkey as PublicKey
from solana.rpc.commitment import Commitment
from solana.rpc.types import TxOpts
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
import base58
from dataclasses import dataclass
from decimal import Decimal
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Position:
    token_address: str
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    risk_amount: float

class PumpTrader:
    def __init__(self):
        load_dotenv()
        logging.info("Initializing PumpTrader...")
        self.solana_client = AsyncClient(os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'))
        # Create keypair using base58 encoded private key
        private_key_b58 = os.getenv('WALLET_PRIVATE_KEY')
        private_key = base58.b58decode(private_key_b58)
        self.wallet = Keypair.from_bytes(private_key)
        logging.info(f"Wallet initialized with public key: {self.wallet.pubkey()}")
        self.min_sol_balance = float(os.getenv('MIN_SOL_BALANCE', '0.1'))
        self.max_trade_amount = float(os.getenv('MAX_TRADE_AMOUNT', '0.5'))
        self.pump_fun_api = os.getenv('PUMP_FUN_API_URL', 'https://api.pump.fun/v1')
        
        # Trading parameters
        self.min_liquidity = float(os.getenv('MIN_LIQUIDITY', '1000'))  # Minimum liquidity in USD
        self.max_holder_percentage = float(os.getenv('MAX_HOLDER_PERCENTAGE', '20'))  # Maximum percentage a single holder can own
        self.min_holders = int(os.getenv('MIN_HOLDERS', '50'))  # Minimum number of holders
        self.max_tax = float(os.getenv('MAX_TAX', '10'))  # Maximum buy/sell tax percentage
        self.min_market_cap = float(os.getenv('MIN_MARKET_CAP', '50000'))  # Minimum market cap in USD
        
        # Jupiter DEX parameters
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        self.slippage = float(os.getenv('SLIPPAGE_TOLERANCE', '1.0'))  # 1% slippage tolerance
        
        # Risk Management Parameters
        self.max_portfolio_risk = float(os.getenv('MAX_PORTFOLIO_RISK', '2.0'))  # Maximum portfolio risk in percentage
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '5.0'))   # Maximum position size in percentage of portfolio
        self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', '10.0'))  # Stop loss percentage
        self.take_profit_percentage = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '30.0'))  # Take profit percentage
        self.max_open_positions = int(os.getenv('MAX_OPEN_POSITIONS', '5'))  # Maximum number of open positions
        
        # Portfolio tracking
        self.positions: List[Position] = []
        self.portfolio_value = 0.0
        self.last_portfolio_update = None

    async def get_wallet_balance(self):
        try:
            logging.info(f"Getting wallet balance for {self.wallet.pubkey()}")
            balance = await self.solana_client.get_balance(self.wallet.pubkey())
            balance_sol = balance['result']['value'] / 1e9
            logging.info(f"Current wallet balance: {balance_sol} SOL")
            return balance_sol
        except Exception as e:
            logging.error(f"Error getting wallet balance: {e}")
            return 0

    async def get_token_info(self, token_address: str) -> Optional[Dict]:
        """Fetch detailed token information from PumpPortal API"""
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"{self.pump_fun_api}/data/tokens/{token_address}"
                logging.debug(f"Getting token info from: {api_url}")
                
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # PumpPortal might wrap token info differently
                        token_info = data.get('token', data)
                        return token_info
                    else:
                        logging.warning(f"Failed to get token info for {token_address}, status: {response.status}")
                        return None
        except Exception as e:
            logging.error(f"Error fetching token info: {e}")
            return None

    async def check_liquidity(self, token_address: str) -> bool:
        """Check if token has sufficient liquidity"""
        try:
            token_info = await self.get_token_info(token_address)
            if not token_info:
                return False
            
            liquidity = float(token_info.get('liquidity', 0))
            return liquidity >= self.min_liquidity
        except Exception as e:
            logging.error(f"Error checking liquidity: {e}")
            return False

    async def check_holder_distribution(self, token_address: str) -> bool:
        """Check token holder distribution"""
        try:
            token_info = await self.get_token_info(token_address)
            if not token_info:
                return False
            
            holders = token_info.get('holders', [])
            if len(holders) < self.min_holders:
                return False
            
            # Check if any single holder owns too much
            for holder in holders:
                percentage = float(holder.get('percentage', 0))
                if percentage > self.max_holder_percentage:
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Error checking holder distribution: {e}")
            return False

    async def check_taxes(self, token_address: str) -> bool:
        """Check if token has reasonable buy/sell taxes"""
        try:
            token_info = await self.get_token_info(token_address)
            if not token_info:
                return False
            
            buy_tax = float(token_info.get('buyTax', 100))
            sell_tax = float(token_info.get('sellTax', 100))
            
            return buy_tax <= self.max_tax and sell_tax <= self.max_tax
        except Exception as e:
            logging.error(f"Error checking taxes: {e}")
            return False

    async def check_market_cap(self, token_address: str) -> bool:
        """Check if token has sufficient market cap"""
        try:
            token_info = await self.get_token_info(token_address)
            if not token_info:
                return False
            
            market_cap = float(token_info.get('marketCap', 0))
            return market_cap >= self.min_market_cap
        except Exception as e:
            logging.error(f"Error checking market cap: {e}")
            return False

    async def analyze_and_trade(self, token):
        try:
            token_address = token.get('address')
            if not token_address:
                return

            logging.info(f"Analyzing token: {token_address}")

            # Check wallet balance
            balance = await self.get_wallet_balance()
            if balance < self.min_sol_balance:
                logging.warning("Insufficient SOL balance")
                return

            # Perform comprehensive token analysis
            analysis_results = {
                'liquidity': await self.check_liquidity(token_address),
                'holders': await self.check_holder_distribution(token_address),
                'taxes': await self.check_taxes(token_address),
                'market_cap': await self.check_market_cap(token_address)
            }

            # Log analysis results
            logging.info(f"Analysis results for {token_address}: {json.dumps(analysis_results, indent=2)}")

            # Check if token meets all criteria
            if all(analysis_results.values()):
                logging.info(f"Token {token_address} passed all analysis checks")
                await self.execute_trade(token_address)
            else:
                failed_checks = [k for k, v in analysis_results.items() if not v]
                logging.info(f"Token {token_address} failed checks: {failed_checks}")
            
        except Exception as e:
            logging.error(f"Error analyzing token {token.get('address')}: {e}")

    async def monitor_new_tokens(self):
        """Monitor new tokens using PumpPortal WebSocket API"""
        logging.info("Starting to monitor new tokens via WebSocket...")
        websocket_url = "wss://pumpportal.fun/api/data"
        
        while True:
            try:
                async with websockets.connect(websocket_url) as websocket:
                    logging.info("Connected to PumpPortal WebSocket")
                    
                    # Subscribe to new token events
                    subscribe_payload = {
                        "method": "subscribeNewToken"
                    }
                    await websocket.send(json.dumps(subscribe_payload))
                    logging.info("Subscribed to new token events")
                    
                    # Process incoming messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            logging.info(f"Received WebSocket message: {data}")
                            
                            # Process the token data
                            if isinstance(data, dict):
                                token = data.get('token', data)
                                token_address = token.get('address', token.get('mint'))
                                
                                if token_address:
                                    logging.info(f"Processing new token: {token_address}")
                                    await self.analyze_and_trade(token)
                            
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to decode WebSocket message: {e}")
                        except Exception as e:
                            logging.error(f"Error processing WebSocket message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed, attempting to reconnect...")
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def switch_to_test_mode(self):
        """Switch to test mode when WebSocket connection fails"""
        logging.info("Switching to test mode due to WebSocket connection failure")
        # Load test tokens similar to testnet_trader.py
        try:
            with open('test_tokens.json', 'r') as f:
                self.test_tokens = json.load(f)
                logging.info(f"Loaded {len(self.test_tokens)} test tokens")
        except FileNotFoundError:
            # Create default test tokens
            self.test_tokens = [
                {
                    "address": "So11111111111111111111111111111111111111112",  # Wrapped SOL
                    "name": "Test Token 1",
                    "initial_price": 1.0,
                    "volatility": 5.0
                },
                {
                    "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                    "name": "Test Token 2", 
                    "initial_price": 1.0,
                    "volatility": 2.0
                }
            ]
            with open('test_tokens.json', 'w') as f:
                json.dump(self.test_tokens, f, indent=2)
            logging.info("Created default test tokens")
        
        # Start monitoring test tokens instead
        asyncio.create_task(self.monitor_test_tokens())
    
    async def monitor_test_tokens(self):
        """Monitor test tokens in absence of API"""
        logging.info("Starting to monitor test tokens...")
        while True:
            try:
                for token in self.test_tokens:
                    logging.info(f"Processing test token: {token['address']}")
                    await self.analyze_and_trade(token)
                
                # Simulate discovering new tokens occasionally
                if random.random() < 0.1:  # 10% chance
                    new_token = {
                        "address": f"TestToken{len(self.test_tokens) + 1}",
                        "name": f"Test Token {len(self.test_tokens) + 1}",
                        "initial_price": random.uniform(0.1, 10.0),
                        "volatility": random.uniform(2.0, 10.0)
                    }
                    self.test_tokens.append(new_token)
                    logging.info(f"New test token discovered: {new_token['address']}")
                
                # Wait before next cycle
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"Error monitoring test tokens: {e}")
                await asyncio.sleep(30)

    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price in SOL"""
        # Check if we're in test mode and using test tokens
        if hasattr(self, 'test_tokens'):
            # This is a test token
            token = next((t for t in self.test_tokens if t['address'] == token_address), None)
            if token:
                return await self.simulate_price_update(token)
                
        # Normal price lookup through Jupiter API
        try:
            async with aiohttp.ClientSession() as session:
                # Get price quote from Jupiter
                quote_url = f"{self.jupiter_api}/quote?inputMint=So11111111111111111111111111111111111111112&outputMint={token_address}&amount=1000000000&slippageBps={int(self.slippage * 100)}"
                async with session.get(quote_url) as response:
                    if response.status == 200:
                        quote_data = await response.json()
                        return float(quote_data.get('outAmount', 0)) / 1e9
                    return None
        except Exception as e:
            logging.error(f"Error getting token price: {e}")
            return None
            
    async def simulate_price_update(self, token):
        """Simulate price movement for test tokens"""
        try:
            # Get volatility or use default
            volatility = token.get('volatility', 5.0)
            
            # Generate random price movement
            price_change = random.uniform(-volatility, volatility) / 100.0
            
            # Update price
            current_price = token['initial_price'] * (1 + price_change)
            
            # Store updated price
            token['initial_price'] = current_price
            
            logging.info(f"Simulated price for {token['address']}: {current_price:.6f} SOL (change: {price_change*100:.2f}%)")
            
            return current_price
        except Exception as e:
            logging.error(f"Error simulating price: {e}")
            return token.get('initial_price', 1.0)

    async def create_token_account(self, token_address: str) -> Optional[PublicKey]:
        """Create a token account for the given token"""
        try:
            # Get the associated token account address
            token_pubkey = PublicKey(token_address)
            associated_token_account = await self.solana_client.get_token_accounts_by_owner(
                self.wallet.pubkey(),
                {"mint": token_pubkey}
            )
            
            if associated_token_account['result']['value']:
                return PublicKey(associated_token_account['result']['value'][0]['pubkey'])
            
            # Create new token account if it doesn't exist
            transaction = Transaction()
            # Add create associated token account instruction
            # Note: This is a simplified version. In production, you'd need to add the actual instruction
            
            # Sign and send transaction
            result = await self.solana_client.send_transaction(
                transaction,
                self.wallet
            )
            
            if 'result' in result:
                return PublicKey(result['result'])
            return None
        except Exception as e:
            logging.error(f"Error creating token account: {e}")
            return None

    async def calculate_position_size(self, token_price: float, risk_per_trade: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get current portfolio value
            portfolio_value = await self.get_portfolio_value()
            
            # Calculate maximum position size in SOL
            max_position_sol = portfolio_value * (self.max_position_size / 100)
            
            # Calculate position size based on risk
            risk_amount = portfolio_value * (risk_per_trade / 100)
            position_size = risk_amount / (self.stop_loss_percentage / 100)
            
            # Take the minimum of max position size and risk-based size
            position_size = min(position_size, max_position_sol)
            
            # Ensure we don't exceed max trade amount
            position_size = min(position_size, self.max_trade_amount)
            
            return position_size
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.0

    async def get_portfolio_value(self) -> float:
        """Calculate total portfolio value including all positions"""
        try:
            # Get SOL balance
            sol_balance = await self.get_wallet_balance()
            portfolio_value = sol_balance
            
            # Add value of all positions
            for position in self.positions:
                current_price = await self.get_token_price(position.token_address)
                if current_price:
                    position_value = position.amount * current_price
                    portfolio_value += position_value
            
            self.portfolio_value = portfolio_value
            self.last_portfolio_update = datetime.now()
            return portfolio_value
        except Exception as e:
            logging.error(f"Error calculating portfolio value: {e}")
            return 0.0

    async def check_portfolio_risk(self) -> bool:
        """Check if adding a new position would exceed portfolio risk limits"""
        try:
            total_risk = sum(pos.risk_amount for pos in self.positions)
            portfolio_value = await self.get_portfolio_value()
            
            # Calculate current portfolio risk percentage
            current_risk_percentage = (total_risk / portfolio_value) * 100
            
            # Check if adding a new position would exceed max portfolio risk
            return current_risk_percentage < self.max_portfolio_risk
        except Exception as e:
            logging.error(f"Error checking portfolio risk: {e}")
            return False

    async def monitor_positions(self):
        """Monitor open positions for stop loss and take profit"""
        logging.info("Position monitoring started...")
        while True:
            try:
                if not self.positions:
                    logging.info("No open positions to monitor")
                else:
                    logging.info(f"Monitoring {len(self.positions)} open positions")
                    
                for position in self.positions[:]:  # Create a copy to iterate
                    logging.info(f"Checking position for token {position.token_address}")
                    current_price = await self.get_token_price(position.token_address)
                    if not current_price:
                        logging.warning(f"Could not get price for {position.token_address}")
                        continue
                    
                    # Calculate price change percentage
                    price_change = ((current_price - position.entry_price) / position.entry_price) * 100
                    logging.info(f"Token {position.token_address} price change: {price_change:.2f}%")
                    
                    # Check stop loss
                    if price_change <= -position.stop_loss:
                        logging.info(f"Stop loss triggered for {position.token_address}")
                        await self.close_position(position, "stop_loss")
                    
                    # Check take profit
                    elif price_change >= position.take_profit:
                        logging.info(f"Take profit triggered for {position.token_address}")
                        await self.close_position(position, "take_profit")
                
                logging.info("Position monitoring cycle complete, sleeping...")
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logging.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def close_position(self, position: Position, reason: str):
        """Close a position and log the result"""
        try:
            # Get current price
            current_price = await self.get_token_price(position.token_address)
            if not current_price:
                return
            
            # Calculate P&L
            pnl = (current_price - position.entry_price) * position.amount
            
            # Execute sell transaction
            await self.execute_trade(position.token_address, is_sell=True)
            
            # Remove position from tracking
            self.positions.remove(position)
            
            # Log the result
            logging.info(f"Position closed for {position.token_address}")
            logging.info(f"Reason: {reason}")
            logging.info(f"P&L: {pnl:.2f} SOL")
            
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    async def execute_trade(self, token_address: str, is_sell: bool = False):
        try:
            logging.info(f"Executing {'sell' if is_sell else 'buy'} for token: {token_address}")
            
            # Get current token price
            token_price = await self.get_token_price(token_address)
            if not token_price:
                logging.error("Failed to get token price")
                return
            
            if not is_sell:
                # Check portfolio risk before buying
                if not await self.check_portfolio_risk():
                    logging.warning("Portfolio risk limit reached")
                    return
                
                # Check maximum positions
                if len(self.positions) >= self.max_open_positions:
                    logging.warning("Maximum open positions reached")
                    return
                
                # Calculate position size
                trade_amount = await self.calculate_position_size(token_price, self.stop_loss_percentage)
                if trade_amount <= 0:
                    logging.error("Invalid position size")
                    return
            else:
                # For selling, use the position amount
                position = next((p for p in self.positions if p.token_address == token_address), None)
                if not position:
                    logging.error("Position not found")
                    return
                trade_amount = position.amount
            
            # Create or get token account
            token_account = await self.create_token_account(token_address)
            if not token_account:
                logging.error("Failed to create/get token account")
                return
            
            # Get Jupiter quote for the trade
            async with aiohttp.ClientSession() as session:
                # Swap input and output mints based on buy/sell
                input_mint = token_address if is_sell else "So11111111111111111111111111111111111111112"
                output_mint = "So11111111111111111111111111111111111111112" if is_sell else token_address
                
                quote_url = f"{self.jupiter_api}/quote?inputMint={input_mint}&outputMint={output_mint}&amount={int(trade_amount * 1e9)}&slippageBps={int(self.slippage * 100)}"
                async with session.get(quote_url) as response:
                    if response.status != 200:
                        logging.error("Failed to get trade quote")
                        return
                    
                    quote_data = await response.json()
                    
                    # Get swap transaction
                    swap_url = f"{self.jupiter_api}/swap"
                    swap_payload = {
                        "quoteResponse": quote_data,
                        "userPublicKey": str(self.wallet.pubkey()),
                        "wrapUnwrapSOL": True
                    }
                    
                    async with session.post(swap_url, json=swap_payload) as swap_response:
                        if swap_response.status != 200:
                            logging.error("Failed to get swap transaction")
                            return
                        
                        swap_data = await swap_response.json()
                        
                        # Sign and send the transaction
                        transaction = Transaction.deserialize(base58.b58decode(swap_data['swapTransaction']))
                        result = await self.solana_client.send_transaction(
                            transaction,
                            self.wallet,
                            opts=TxOpts(skip_preflight=True)
                        )
                        
                        if 'result' in result:
                            signature = result['result']
                            logging.info(f"Trade executed successfully! Signature: {signature}")
                            
                            # Wait for confirmation
                            confirmation = await self.solana_client.confirm_transaction(signature)
                            if confirmation['result']['value']:
                                logging.info("Trade confirmed!")
                                
                                # Update position tracking for buys
                                if not is_sell:
                                    new_position = Position(
                                        token_address=token_address,
                                        entry_price=token_price,
                                        amount=trade_amount,
                                        stop_loss=self.stop_loss_percentage,
                                        take_profit=self.take_profit_percentage,
                                        timestamp=datetime.now(),
                                        risk_amount=trade_amount * (self.stop_loss_percentage / 100)
                                    )
                                    self.positions.append(new_position)
                            else:
                                logging.warning("Trade confirmation failed")
                        else:
                            logging.error("Failed to send transaction")
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}")

    async def run(self):
        logging.info("Starting Pump Trader...")
        try:
            # Check wallet balance first
            balance = await self.get_wallet_balance()
            logging.info(f"Initial wallet balance: {balance} SOL")
            
            # Start position monitoring in background
            logging.info("Starting position monitoring task...")
            monitor_task = asyncio.create_task(self.monitor_positions())
            
            # Start token monitoring
            logging.info("Starting token monitoring...")
            await self.monitor_new_tokens()
        except Exception as e:
            logging.critical(f"Fatal error in main loop: {e}")
            raise

if __name__ == "__main__":
    try:
        logging.info("Creating PumpTrader instance...")
        trader = PumpTrader()
        logging.info("Starting trader...")
        asyncio.run(trader.run())
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        import traceback
        traceback.print_exc() 