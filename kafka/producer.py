import asyncio
from time import monotonic

from aiokafka import AIOKafkaProducer


async def send_one():
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',
    )
    # Get cluster layout and initial topic/partition leadership information
    await producer.start()
    try:
        # Produce message
        while True:
            message = f"Super message: {monotonic()}"
            await producer.send_and_wait("t1", message.encode())
            await asyncio.sleep(5)
    finally:
        # Wait for all pending messages to be delivered or expire.
        await producer.stop()
        
if __name__ == "__main__":

    asyncio.run(send_one())