from logger import logger, d
from time import sleep
from multiprocessing import Pool
from multiprocessing import set_start_method

def job(i):
    sleep(1)
    d[i] = 1
    logger.log("job done")

if __name__ == "__main__":
    set_start_method("spawn")
    N_JOBS = 10
    N_WORKERS = 4
    with Pool(N_WORKERS) as p:
        p.map(job, range(N_JOBS))
    
    print(d)