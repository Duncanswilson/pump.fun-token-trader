import os
import asyncio
from typing import Optional, Dict, Any
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment
from solana.rpc.types import TxOpts
from solders.keypair import Keypair
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

class RPCClient:
    def __init__(self):
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Get environment variables
        self.provider = os.getenv('RPC_PROVIDER', 'quicknode')
        self.endpoint = os.getenv('RPC_ENDPOINT')
        self.ws_endpoint = os.getenv('RPC_WS_ENDPOINT')
        
        try:
            self.retry_count = int(os.getenv('RPC_RETRY_COUNT', '3'))
            self.timeout = int(os.getenv('RPC_TIMEOUT', '30'))
        except ValueError as e:
            self.logger.error(f"Error parsing environment variables: {e}")
            self.retry_count = 3
            self.timeout = 30
        
        if not self.endpoint:
            raise ValueError("RPC endpoint not configured")
        
        # Initialize the client with custom configuration
        self.client = AsyncClient(
            self.endpoint,
            commitment=Commitment("confirmed"),
            timeout=self.timeout
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def get_balance(self, public_key: str) -> Optional[float]:
        """Get SOL balance with retry logic"""
        try:
            result = await self.client.get_balance(public_key)
            if 'result' in result:
                return result['result']['value'] / 1e9
            return None
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def send_transaction(self, transaction: Any, signers: list, opts: Optional[TxOpts] = None) -> Optional[str]:
        """Send transaction with retry logic"""
        try:
            result = await self.client.send_transaction(
                transaction,
                *signers,
                opts=opts or TxOpts(skip_preflight=True)
            )
            if 'result' in result:
                return result['result']
            return None
        except Exception as e:
            self.logger.error(f"Error sending transaction: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def confirm_transaction(self, signature: str) -> bool:
        """Confirm transaction with retry logic"""
        try:
            result = await self.client.confirm_transaction(signature)
            return result['result']['value']
        except Exception as e:
            self.logger.error(f"Error confirming transaction: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def get_token_accounts_by_owner(self, owner: str, mint: str) -> Optional[Dict]:
        """Get token accounts with retry logic"""
        try:
            result = await self.client.get_token_accounts_by_owner(
                owner,
                {"mint": mint}
            )
            return result
        except Exception as e:
            self.logger.error(f"Error getting token accounts: {e}")
            raise

    async def close(self):
        """Close the RPC connection"""
        try:
            await self.client.close()
        except Exception as e:
            self.logger.error(f"Error closing RPC connection: {e}")

    def __del__(self):
        """Cleanup on object destruction"""
        asyncio.create_task(self.close()) 