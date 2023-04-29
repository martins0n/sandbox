* some experiments with psql COPY
* pick data usage about 1.28GB for click for 4.9GB csv file but table is about 415MB

```bash
SELECT table,
    formatReadableSize(sum(bytes)) as size,
    min(min_date) as min_date,
    max(max_date) as max_date
    FROM system.parts
    WHERE active
GROUP BY table
```

* it seems like managed click 10x faster than managed postgresql in the same vm instance
* and on the same level as localhosted postgresql on M1 macbook air
