import socket


if __name__ == "__main__":
    sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sck.connect(("localhost", 9434))
    sck.sendall(b"HELLO WORLD")
    data = sck.recv(1024)
    print(data)
    sck.close()
