import socket
import sys

# Add the LMCP Python modules directory to the path
sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")

from lmcp import LMCPFactory
from afrl.cmasi import AirVehicleConfiguration

HOST = '127.0.0.1'
PORT = 5560  # Make sure this matches the server's port

def main():
    # Construct an LMCP object
    msg = AirVehicleConfiguration.AirVehicleConfiguration()
    msg.ID = 1
    msg.Label = "MyDrone"

    # Serialize (raw LMCP bytes, with 4-byte sync header)
    data_out = LMCPFactory.packMessage(msg, True)

    # Connect to the TCP socket and send the message
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data_out)
        print("LMCP message sent to the server.")

        # Wait for a response from the server
        data_in = s.recv(8192)
        if data_in:
            response, _ = LMCPFactory.getObject(data_in)
            print("Response from Server:", response)
        else:
            print("No data received.")

if __name__ == "__main__":
    main()
