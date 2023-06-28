import os
import random
from asyncio import sleep

from aio_pika import Channel, ExchangeType, Message, Queue, connect_robust

from common import (
    RABBIT_URL,
    RESULT_QUEUE_NAME,
    SEND_QUEUE_NAME,
    AddRequest,
    AddResponse,
)

channel_result: Channel = None


async def consumer(message: Message):
    async with message.process():
        request = AddRequest.parse_raw(message.body.decode())
        print(" [x] Received %r" % request)

        result = AddResponse(result=request.a + request.b)

        await sleep(5 * random.random())

        await channel_result.default_exchange.publish(
            Message(
                body=result.json().encode(),
                correlation_id=message.correlation_id,
            ),
            routing_key=RESULT_QUEUE_NAME,
        )
        print(" [x] Sent %r" % result)


async def main():
    global channel_result
    connection = await connect_robust(RABBIT_URL)
    channel = await connection.channel()
    channel_result = await connection.channel()
    await channel.set_qos(prefetch_count=int(os.environ["WORKERS"]))
    queue: Queue = await channel.declare_queue(SEND_QUEUE_NAME)

    await queue.consume(consumer)

    print(" [*] Waiting for messages. To exit press CTRL+C")

    await asyncio.Future()
    await connection.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
