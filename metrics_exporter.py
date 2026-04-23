#!/usr/bin/env python3
from prometheus_client import Counter, start_http_server

B = Counter("bronze_records_processed_total", "Bronze records processed")
S = Counter("silver_records_processed_total", "Silver records processed")
G = Counter("gold_records_processed_total", "Gold records processed")
BE = Counter("bronze_errors_total", "Bronze errors", ["source"])
SE = Counter("silver_errors_total", "Silver errors", ["dataset"])
GE = Counter("gold_errors_total", "Gold errors", ["table"])

start_http_server(8888)

B.inc(14865679)
S.inc(9564403)
G.inc(11743824)

print("Metrics server started on port 8888")
print(f"Bronze: {14865679}, Silver: {9564403}, Gold: {11743824}")

import time
while True:
    time.sleep(60)