import socket
import struct
import os, os.path

HOST = "127.0.0.1"
PORT = 9999


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)

print('IMAGE UPLOAD SERVER STARTED RUNNING')

while True:
    client, address = s.accept()
    buf = ''
    while len(buf) < 4:
        buf += client.recv(4 - len(buf))
    size = struct.unpack('!i', buf)[0]

    DIR = os.getcwd()+'/uploaded_images/'
    frame_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    print frame_count

    with open('/Users/joshuahowarth/dev/celldetector/pythonclassifier/uploaded_images/frame_%s.png' % str(frame_count).zfill(4), 'wb') as f:
        while size > 0:
            data = client.recv(1024)
            f.write(data)
            size -= len(data)
    print('Image Saved')
    client.sendall('Image Received')
    client.close()


s.close()
print('SERVER AND SOCKET CLOSED')