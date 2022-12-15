from argparse import ArgumentParser
import asyncio

from aiokafka import AIOKafkaConsumer

async def consume(group_id):
    consumer = AIOKafkaConsumer(
        't1',
        bootstrap_servers='localhost:9092',
        group_id=group_id,
    )
    await consumer.start()
    try:
        # Consume messages
        async for msg in consumer:
            print("consumed: ", msg.topic, msg.partition, msg.offset,
                  msg.key, msg.value, msg.timestamp)
    finally:
        # Will leave consumer group; perform autocommit if enabled.
        await consumer.stop()
        
if __name__ == "__main__":
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--groupid', type=int)
    args = arg_parser.parse_args()    

    asyncio.run(consume(args.groupid))
