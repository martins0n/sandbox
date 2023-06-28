from locust import HttpUser, TaskSet, task


class MyTaskSet(TaskSet):
    @task
    def create(self):
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        self.client.post("/", json={"a": 0, "b": 1}, headers=headers)


class MyLocust(HttpUser):
    tasks = [MyTaskSet]
    min_wait = 5000
    max_wait = 9000
