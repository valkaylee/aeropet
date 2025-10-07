import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 8890))
print("Listening on port 8890...")
while True:
    data, _ = sock.recvfrom(1024)
    print(data)
