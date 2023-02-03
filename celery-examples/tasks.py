from celery import Celery
import time
import random
broker_url = 'amqp://guest:guest@localhost:5672'


app = Celery('tasks', broker=broker_url, backend="redis://localhost:6379/0")

@app.task
def add(x, y):
    time.sleep(y)
    if random.randint(0, 10) > 5:
        raise Exception("Something went wrong")
    return x + y

@app.task
def notify(message: str):
    print(message)
