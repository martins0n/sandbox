from tasks import add, notify
from celery.result import AsyncResult
from celery import group

validation_tasks = group(add.si(1, 1), add.si(2, 2), add.si(3, 10))
#validation_tasks.link_error(notify.si("Validation failed"))
send_result = notify.si("Validation succeeded")
validation_tasks.link(notify.si("Validation succeeded linked"))
total_task = validation_tasks | send_result

result: AsyncResult = total_task.delay()
print(result.wait())