from clickhouse_driver import Client
from functools import partial
import hydra
from time import monotonic
from omegaconf import DictConfig, OmegaConf
import hydra
import json
import pathlib
import subprocess


PATH = pathlib.Path(__file__).parent.resolve()


class Client:
    def __init__(self, settings) -> None:
        self.cmd = f"""clickhouse client --host {settings.host} --user {settings.user} --password {settings.password} --port {settings.port} --secure"""
    
    def execute(self, query):
        print(subprocess.check_output(self.cmd + f''' --query "{query}"''', shell=True))


@hydra.main(config_path="configs", config_name="config")
def app(cfg: DictConfig) -> None:
    settings = cfg.storage
    
    print(settings)
    client = Client(settings)    


    print(PATH)
    f = (PATH / f"data_{cfg.data.segments}.csv").resolve()
    
    
    q = client
    start = monotonic()
    q.execute('DROP TABLE IF EXISTS db1.test;')
    q.execute("""
        CREATE TABLE IF NOT EXISTS db1.test
        (timestamp timestamp, target FLOAT, segment varchar) ENGINE = MergeTree
        PRIMARY KEY (segment, timestamp)
    ;""")
    q.execute(f"INSERT INTO db1.test FROM INFILE '{f}' FORMAT CSV")

    end = monotonic()

    # print time in seconds
    
    total_time = end - start

    print("Time: ", total_time)
    print("Time per segment: ", total_time  / int(cfg.data.segments))

    t = q.execute('SELECT * FROM db1.test limit 1;')
    print(t)
    
    with open("results.json", "w") as f:
        json.dump({
            "time": total_time,
            "time_per_segment": total_time / int(cfg.data.segments),
            "segments": int(cfg.data.segments),
        }, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    app()
