# consumer.py
import asyncio
from aio_pika import connect, IncomingMessage

async def on_message(message: IncomingMessage):
    async with message.process():
        await asyncio.sleep(100)
        print(f"Received task: {message.body.decode()}")

async def main():
    connection = await connect("amqp://guest:guest@rabbitmq:5672",)
    channel = await connection.channel()
    
    channel_second = await connection.channel()
    
    await channel.set_qos(prefetch_count=10)
    
    await channel_second.set_qos(prefetch_count=100)

    queue = await channel.declare_queue("tasks", durable=True)
    
    queue_second = await channel_second.declare_queue("tasks", durable=True)
    
    await queue.consume(on_message)
    await queue_second.consume(on_message)

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())