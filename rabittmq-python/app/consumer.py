# consumer.py
import asyncio
from aio_pika import connect, IncomingMessage
import random

async def on_message(message: IncomingMessage):
    async with message.process():
        # print message metadata
        print(message.info())
        print(message.body)

        await asyncio.sleep(random.randint(1, 10))
        a = [f"{i}" for i in range(100_000_000)]
        print(f"Received task: {message.body.decode()}")

async def main():
    connection = await connect("amqp://guest:guest@rabbitmq:5672",)
    channel = await connection.channel()
    
    channel_second = await connection.channel()
    
    await channel.set_qos(prefetch_count=1)
    
    await channel_second.set_qos(prefetch_count=1)

    queue = await channel.declare_queue("tasks", durable=True, arguments={
        'x-dead-letter-exchange': 'dead.letter.exchange'
    })
    
    queue_second = await channel_second.declare_queue("tasks", durable=True, arguments={
        'x-dead-letter-exchange': 'dead.letter.exchange'
    })
    
    await queue.consume(on_message)
    await queue_second.consume(on_message)

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())