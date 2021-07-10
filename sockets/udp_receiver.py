import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5008


with socket.socket(
    socket.AF_INET,  # Internet
    socket.SOCK_DGRAM  # UDP
) as sck:
    sck.bind((UDP_IP, UDP_PORT))

    while True:
        data, addr = sck.recvfrom(1024)
        print("received message: %s" % data)
