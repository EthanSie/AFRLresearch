import socket
import sys
sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")

from lmcp import LMCPFactory
from afrl.cmasi import AirVehicleConfiguration

HOST = '127.0.0.1'
PORT = 5555

def main():
    # 1) Construct an LMCP object
    msg = AirVehicleConfiguration.AirVehicleConfiguration()
    msg.ID = 1
    msg.Label = "MyDrone"

    # 2) Serialize (raw LMCP bytes, with 4-byte sync header)
    data_out = LMCPFactory.packMessage(msg, True)

    # 3) Connect a plain TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data_out)

        # 4) If your scenario or plugin sends anything back, read here
        data_in = s.recv(8192)
        if data_in:
            response, _ = LMCPFactory.getObject(data_in)
            print("Response from AMASE:", response)
        else:
            print("No data received.")

if __name__ == "__main__":
    main()
