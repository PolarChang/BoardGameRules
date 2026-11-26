import hashlib,pathlib
for n,h in {'api.py': 'd5a21bf565e7abd617e6886ade7bb3aa0b0572538fb2c81dcd1023285d583618', 'game_catalog_api.py': '6b1c99bbe5f17b94a84b3ad0bf37f2ed41b44b9b0365a38f564b0144687d58b7', 'local_grounded_answer.py': 'c2159170fb3d518380e1684d53484a19b0fae1351b25d6d3b8c48becc67b6c64'}.items():
 p=pathlib.Path("/app/src")/n
 assert hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==h, n+" base mismatch"
