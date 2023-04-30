from z3 import *
from collections import namedtuple

MAX_TIME = 100

task = namedtuple('task', ['name', 'duration'])

worker = namedtuple('worker', ['name', 'work_slot'])


tasks: list[task] = [
    task('A', 3),
    task('B', 2),
    task('C', 2),
    task('D', 1),
    task('E', 4),
    task('F', 50),
]

workers: list[worker] = [
    worker('W1', (0, 50)),
    worker('W2', (0, 50)),
    worker('W3', (50, 100)),
    worker('W4', (50, 100)),
]



task_start_times = [Int(f'{t.name}_start') for t in tasks]

task_start_constraints = [
    And(0 <= t, t <= MAX_TIME - tasks[idx].duration)
    for idx, t in enumerate(task_start_times)
]

worker_x_task = [ [Bool(f'{w.name}_x_{t.name}') for t in tasks] for w in workers]

task_worker_constraints = []

for i in range(len(tasks)):
    for j in range(i + 1, len(tasks)):
        for idx, w in enumerate(workers):
            t1 = tasks[i]
            t2 = tasks[j]

            t1_start = task_start_times[i]
            t2_start = task_start_times[j]

            t1_end = t1_start + t1.duration
            t2_end = t2_start + t2.duration

            t1_worker = And(w.work_slot[0] <= t1_start, t1_end <= w.work_slot[1])
            t2_worker = And(w.work_slot[0] <= t2_start, t2_end <= w.work_slot[1])

            not_the_same_time = Or(t1_end <= t2_start, t2_end <= t1_start)
            
            task_worker_constraints.append(
                Implies(And(worker_x_task[idx][i], worker_x_task[idx][j]), And(not_the_same_time))
            )
            task_worker_constraints.append(
                Implies(worker_x_task[idx][i], t1_worker)
            )
            task_worker_constraints.append(
                Implies(worker_x_task[idx][j], t2_worker)
            )

# all tasks must be done only once constraint
all_tasks_done = [
    Sum([worker_x_task[idx][i] for idx in range(len(workers))]) == 1
    for i in range(len(tasks))
]




s = Solver()

s.add(task_start_constraints)
s.add(task_worker_constraints)
s.add(all_tasks_done)

if s.check() == sat:
    m = s.model()

    for t in tasks:
        print(f'{t.name} starts at {m[task_start_times[tasks.index(t)]]}')
        
    # plot worker_x_task
    for idx, w in enumerate(workers):
        for t in tasks:
            print(f'{w.name} does {t.name} at {m[worker_x_task[idx][tasks.index(t)]]}')