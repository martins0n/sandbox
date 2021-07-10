import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5008
MESSAGE = b"Hello, World!"

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)

with socket.socket(
    socket.AF_INET,  # Internet
    socket.SOCK_DGRAM  # UDP
) as sck:
    sck.sendto(MESSAGE, (UDP_IP, UDP_PORT))
