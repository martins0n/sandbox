import psycopg2
from time import monotonic
from omegaconf import DictConfig, OmegaConf
import hydra
import json
import pathlib

PATH = pathlib.Path(__file__).parent.resolve()


@hydra.main(config_path="configs", config_name="config")
def app(cfg: DictConfig) -> None:
    settings = cfg.storage
    conn = psycopg2.connect(f"""
        host={settings.host}
        port={settings.port}
        dbname={settings.db}
        sslmode={settings.sslmode}
        user={settings.user}
        password={settings.password}
        target_session_attrs=read-write
    """)

    print(PATH)
    f = open(PATH / f"data_{cfg.data.segments}.csv", 'rb')
    q = conn.cursor()

    start = monotonic()
    q.execute('DROP TABLE IF EXISTS test;')
    q.execute('CREATE TABLE IF NOT EXISTS test (timestamp timestamp, target FLOAT, segment varchar);')
    q.copy_from(f, 'test', sep=',')

    end = monotonic()

    # print time in seconds
    
    total_time = end - start

    print("Time: ", total_time)
    print("Time per segment: ", total_time  / int(cfg.data.segments))

    q.execute('SELECT * FROM test;')
    print(q.fetchone())

    conn.close()
    f.close()
    
    with open("results.json", "w") as f:
        json.dump({
            "time": total_time,
            "time_per_segment": total_time / int(cfg.data.segments),
            "segments": int(cfg.data.segments),
        }, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    app()