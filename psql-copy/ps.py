import psycopg2
import os

from time import monotonic

conn = psycopg2.connect(f"""
    host=rc1b-ydyfsw15uvj5oshy.mdb.yandexcloud.net
    port=6432
    dbname=db1
    sslmode=verify-full
    user=user1
    password={os.environ.get('PASS')}
    target_session_attrs=read-write
""")

f = open(f"data_{os.environ.get('SEGMENTS')}.csv", 'rb')
q = conn.cursor()

start = monotonic()
q.execute('DROP TABLE IF EXISTS test;')
q.execute('CREATE TABLE IF NOT EXISTS test (timestamp timestamp, target FLOAT, segment varchar);')
q.copy_from(f, 'test', sep=',')

end = monotonic()

# print time in seconds

print("Time: ", (end - start))
print("Time per segment: ", (end - start)  / int(os.environ.get('SEGMENTS')))

q.execute('SELECT * FROM test;')
print(q.fetchone())

conn.close()
f.close()