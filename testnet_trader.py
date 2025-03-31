import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.system_program import TransferParams, transfer
from solana.publickey import PublicKey
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
from rpc_client import RPCClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testnet_trading.log'),
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

class TestnetPumpTrader:
    def __init__(self):
        load_dotenv()
        # Initialize RPC client
        self.rpc_client = RPCClient()
        self.wallet = Keypair.from_secret_key(bytes.fromhex(os.getenv('WALLET_PRIVATE_KEY')))
        self.min_sol_balance = float(os.getenv('MIN_SOL_BALANCE', '0.1'))
        self.max_trade_amount = float(os.getenv('MAX_TRADE_AMOUNT', '0.5'))
        self.pump_fun_api = os.getenv('PUMP_FUN_API_URL', 'https://api.pump.fun/v1')
        
        # Trading parameters
        self.min_liquidity = float(os.getenv('MIN_LIQUIDITY', '1000'))
        self.max_holder_percentage = float(os.getenv('MAX_HOLDER_PERCENTAGE', '20'))
        self.min_holders = int(os.getenv('MIN_HOLDERS', '50'))
        self.max_tax = float(os.getenv('MAX_TAX', '10'))
        self.min_market_cap = float(os.getenv('MIN_MARKET_CAP', '50000'))
        
        # Jupiter DEX parameters (using devnet)
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        self.slippage = float(os.getenv('SLIPPAGE_TOLERANCE', '1.0'))
        
        # Risk Management Parameters
        self.max_portfolio_risk = float(os.getenv('MAX_PORTFOLIO_RISK', '2.0'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '5.0'))
        self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', '10.0'))
        self.take_profit_percentage = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '30.0'))
        self.max_open_positions = int(os.getenv('MAX_OPEN_POSITIONS', '5'))
        
        # Portfolio tracking
        self.positions: List[Position] = []
        self.portfolio_value = 0.0
        self.last_portfolio_update = None
        
        # Test-specific parameters
        self.test_mode = True
        self.simulation_delay = float(os.getenv('SIMULATION_DELAY', '1.0'))
        self.price_volatility = float(os.getenv('PRICE_VOLATILITY', '5.0'))
        self.test_tokens = self.load_test_tokens()

    def load_test_tokens(self) -> List[Dict]:
        """Load test tokens from a configuration file"""
        try:
            with open('test_tokens.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default test tokens if file doesn't exist
            test_tokens = [
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
                json.dump(test_tokens, f, indent=2)
            return test_tokens

    async def get_wallet_balance(self):
        """Get wallet balance using RPC client"""
        try:
            return await self.rpc_client.get_balance(str(self.wallet.pubkey()))
        except Exception as e:
            logging.error(f"Error getting wallet balance: {e}")
            return 0

    async def request_airdrop(self):
        """Request SOL airdrop for testing"""
        try:
            # Note: Airdrop functionality might not be available with all RPC providers
            # You might need to implement a custom airdrop method based on your provider
            logging.warning("Airdrop functionality may not be available with your RPC provider")
        except Exception as e:
            logging.error(f"Error requesting airdrop: {e}")

    async def create_token_account(self, token_address: str) -> Optional[PublicKey]:
        """Create a token account using RPC client"""
        try:
            token_pubkey = PublicKey(token_address)
            associated_token_account = await self.rpc_client.get_token_accounts_by_owner(
                str(self.wallet.pubkey()),
                {"mint": token_pubkey}
            )
            
            if associated_token_account['result']['value']:
                return PublicKey(associated_token_account['result']['value'][0]['pubkey'])
            
            # Create new token account if it doesn't exist
            transaction = Transaction()
            # Add create associated token account instruction
            
            # Sign and send transaction using RPC client
            signature = await self.rpc_client.send_transaction(
                transaction,
                [self.wallet]
            )
            
            if signature:
                confirmed = await self.rpc_client.confirm_transaction(signature)
                if confirmed:
                    return PublicKey(signature)
            return None
        except Exception as e:
            logging.error(f"Error creating token account: {e}")
            return None

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
                        
                        # Sign and send the transaction using RPC client
                        transaction = Transaction.deserialize(base58.b58decode(swap_data['swapTransaction']))
                        signature = await self.rpc_client.send_transaction(
                            transaction,
                            [self.wallet]
                        )
                        
                        if signature:
                            logging.info(f"Trade executed successfully! Signature: {signature}")
                            
                            # Wait for confirmation using RPC client
                            confirmed = await self.rpc_client.confirm_transaction(signature)
                            if confirmed:
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

    async def simulate_price_update(self, token_address: str) -> Optional[float]:
        """Simulate price updates for test tokens"""
        try:
            token = next((t for t in self.test_tokens if t['address'] == token_address), None)
            if not token:
                return None
            
            # Simulate price movement
            volatility = token.get('volatility', self.price_volatility)
            price_change = random.uniform(-volatility, volatility) / 100
            current_price = token['initial_price'] * (1 + price_change)
            
            # Update token's current price
            token['initial_price'] = current_price
            
            return current_price
        except Exception as e:
            logging.error(f"Error simulating price update: {e}")
            return None

    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get token price (simulated in testnet)"""
        if self.test_mode:
            return await self.simulate_price_update(token_address)
        
        try:
            async with aiohttp.ClientSession() as session:
                quote_url = f"{self.jupiter_api}/quote?inputMint=So11111111111111111111111111111111111111112&outputMint={token_address}&amount=1000000000&slippageBps={int(self.slippage * 100)}"
                async with session.get(quote_url) as response:
                    if response.status == 200:
                        quote_data = await response.json()
                        return float(quote_data.get('outAmount', 0)) / 1e9
                    return None
        except Exception as e:
            logging.error(f"Error getting token price: {e}")
            return None

    async def monitor_new_tokens(self):
        """Monitor for new test tokens"""
        while True:
            try:
                # Simulate new token discovery
                if random.random() < 0.1:  # 10% chance of new token
                    new_token = {
                        "address": f"TestToken{len(self.test_tokens)}",
                        "name": f"Test Token {len(self.test_tokens)}",
                        "initial_price": random.uniform(0.1, 10.0),
                        "volatility": random.uniform(2.0, 10.0)
                    }
                    self.test_tokens.append(new_token)
                    logging.info(f"New test token discovered: {new_token['name']}")
                    await self.analyze_and_trade(new_token)
                
                await asyncio.sleep(self.simulation_delay)
            except Exception as e:
                logging.error(f"Error monitoring new tokens: {e}")
                await asyncio.sleep(5)

    async def run(self):
        logging.info("Starting Testnet Pump Trader...")
        
        try:
            # Request initial airdrop
            await self.request_airdrop()
            
            # Start position monitoring in background
            asyncio.create_task(self.monitor_positions())
            
            # Start token monitoring
            await self.monitor_new_tokens()
        finally:
            # Cleanup RPC connection
            await self.rpc_client.close()

if __name__ == "__main__":
    trader = TestnetPumpTrader()
    asyncio.run(trader.run()) 