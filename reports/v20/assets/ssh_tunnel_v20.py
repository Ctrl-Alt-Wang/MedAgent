#!/usr/bin/env python3
"""SSH tunnel: localhost:8015 -> 117.50.198.65:8001 (v20 step458 GRPO-SFT 14B)"""
import paramiko
import socket
import select
import threading
import sys
import time
import warnings
warnings.filterwarnings("ignore")

SSH_HOST = "117.50.198.65"
SSH_PORT = 23
SSH_USER = "root"
SSH_PASS = "Zw6832qa945sP0MA"
LOCAL_PORT = 8015
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 8001


def relay(sock_a, sock_b):
    try:
        while True:
            r, _, _ = select.select([sock_a, sock_b], [], [], 5)
            if sock_a in r:
                data = sock_a.recv(65536)
                if not data:
                    break
                sock_b.sendall(data)
            if sock_b in r:
                data = sock_b.recv(65536)
                if not data:
                    break
                sock_a.sendall(data)
    except Exception:
        pass
    finally:
        try: sock_a.close()
        except: pass
        try: sock_b.close()
        except: pass


def handle_client(local_sock, transport):
    try:
        chan = transport.open_channel(
            "direct-tcpip",
            (REMOTE_HOST, REMOTE_PORT),
            ("127.0.0.1", 0),
        )
    except Exception as e:
        print(f"[tunnel] channel open failed: {e}")
        local_sock.close()
        return
    relay(local_sock, chan)


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=30)
    transport = client.get_transport()
    transport.set_keepalive(30)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LOCAL_PORT))
    srv.listen(20)
    print(f"[tunnel] listening on localhost:{LOCAL_PORT} -> {SSH_HOST}:{REMOTE_PORT}")
    sys.stdout.flush()

    while True:
        try:
            local_sock, addr = srv.accept()
            t = threading.Thread(target=handle_client, args=(local_sock, transport), daemon=True)
            t.start()
        except Exception as e:
            print(f"[tunnel] accept error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
