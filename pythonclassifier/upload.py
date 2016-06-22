import socket
import struct

HOST = "127.0.0.1"
PORT = 9999

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)

print('SERVER STARTED RUNNING')

while True:
    client, address = s.accept()
    buf = ''
    while len(buf) < 4:
        buf += client.recv(4 - len(buf))
    size = struct.unpack('!i', buf)[0]
    with open('/Users/joshuahowarth/dev/celldetector/pythonclassifier/uploaded_images/image.png', 'wb') as f:
        while size > 0:
            data = client.recv(1024)
            f.write(data)
            size -= len(data)
    print('Image Saved')
    client.sendall('Image Received')
    client.close()
    break

s.close()
print('SOCKET CLOSED')