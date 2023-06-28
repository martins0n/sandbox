import asyncio
import json
import uuid
from asyncio import Queue as AsyncQueue
from collections import defaultdict

from aio_pika import Channel, Connection, Message, Queue, connect_robust
from fastapi import FastAPI

from common import (
    RABBIT_URL,
    RESULT_QUEUE_NAME,
    SEND_QUEUE_NAME,
    AddRequest,
    AddResponse,
)

app = FastAPI()

# create a connection to rabbitmq

channel_send: Channel = None
queue_result: Queue = None
connection: Connection = None


async def init_rabbitmq():
    global channel_send, queue_result, connection
    connection = await connect_robust(RABBIT_URL)
    channel_send = await connection.channel()
    channel_result = await connection.channel()
    await channel_result.set_qos(prefetch_count=1)
    queue_result = await channel_result.declare_queue(RESULT_QUEUE_NAME)


async def publish_to_rabbitmq(message: dict):
    global channel_send

    correlation_id = str(uuid.uuid4())

    CHANNELS[correlation_id] = AsyncQueue()

    message_ = Message(
        body=json.dumps(message).encode(),
        correlation_id=correlation_id,
    )
    await channel_send.default_exchange.publish(
        message_,
        routing_key=SEND_QUEUE_NAME,
    )

    return correlation_id


async def consume_from_rabbitmq():
    global queue_result
    async with queue_result.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                print(" [x] Received %r" % message.body)
                correlation_id = message.correlation_id

                CHANNELS[correlation_id].put_nowait(message.body.decode())


CHANNELS = dict()


@app.on_event("startup")
async def startup_event():
    await init_rabbitmq()
    loop = asyncio.get_event_loop()

    loop.create_task(consume_from_rabbitmq())


@app.on_event("shutdown")
async def shutdown_event():
    await connection.close()


@app.post("/")
async def add(request: AddRequest):
    print(" [x] Received %r" % request)
    correlation_id = await publish_to_rabbitmq(request.dict())
    result = await CHANNELS[correlation_id].get()
    del CHANNELS[correlation_id]
    return AddResponse.parse_raw(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
