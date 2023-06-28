from pydantic import BaseModel


class AddRequest(BaseModel):
    a: int
    b: int


class AddResponse(BaseModel):
    result: int


SEND_QUEUE_NAME = "add_queue"
RESULT_QUEUE_NAME = "result_queue"
RABBIT_URL = "amqp://admin:admin@localhost/"
