import asyncio
from rpc_client import RPCClient
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Print environment variables
logger.info(f"RPC_PROVIDER: {os.getenv('RPC_PROVIDER')}")
logger.info(f"RPC_ENDPOINT: {os.getenv('RPC_ENDPOINT')}")
logger.info(f"RPC_RETRY_COUNT: {os.getenv('RPC_RETRY_COUNT')}")
logger.info(f"RPC_TIMEOUT: {os.getenv('RPC_TIMEOUT')}")

async def test_connection():
    try:
        # Initialize RPC client
        rpc_client = RPCClient()
        
        # Test basic RPC call
        logger.info("Testing QuickNode connection...")
        
        # Get current slot
        result = await rpc_client.client.get_slot()
        logger.info(f"Successfully connected to QuickNode! Current slot: {result}")
            
        # Test WebSocket connection if configured
        if rpc_client.ws_endpoint:
            logger.info("Testing WebSocket connection...")
            # Add WebSocket test here if needed
            
        # Cleanup
        await rpc_client.close()
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection()) 