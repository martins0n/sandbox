# producer.py
import asyncio
from aio_pika import connect, Message

async def main():
    connection = await connect("amqp://guest:guest@rabbitmq:5672")
    channel = await connection.channel()

    queue = await channel.declare_queue("tasks", durable=True)

    for i in range(1000):
        message = Message(f"Task {i}".encode())
        await channel.default_exchange.publish(message, routing_key=queue.name)
        print(f"Sent task {i}")

    await connection.close()

if __name__ == "__main__":
    asyncio.run(main())
