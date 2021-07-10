import socket


if __name__ == "__main__":
    with socket.socket(
        socket.AF_INET,  # Internet Protocol v4
        type=socket.SOCK_STREAM  # TCP
    ) as sck:
        sck.bind(("localhost", 9434))
        sck.listen()
        while True:
            conn, addr = sck.accept()
            data = conn.recv(1024)
            conn.sendall(b"234234")
            print(conn, addr, data)
            conn.close()
