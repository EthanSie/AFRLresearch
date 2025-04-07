import socket
import sys

# Add the directory containing the LMCP Python modules to the path
sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")

from lmcp import LMCPFactory
from afrl.cmasi import AirVehicleConfiguration

# Bind to all interfaces
HOST = '0.0.0.0'
PORT = 5560  # Using a port above 1024 that is likely to be free

def start_server():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Allow reuse of the address
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(5)
            print("Server is listening on {}:{}".format(HOST, PORT))
            
            while True:
                client_socket, addr = server_socket.accept()
                with client_socket:
                    print("\nConnection from", addr)
                    data = client_socket.recv(8192)
                    if data:
                        # Unpack the LMCP message using LMCPFactory
                        message, _ = LMCPFactory.getObject(data)
                        if isinstance(message, AirVehicleConfiguration.AirVehicleConfiguration):
                            # Print a noticeable message
                            print("############################################")
                            print(">>> RECEIVED AIR VEHICLE CONFIGURATION <<<")
                            print("Drone ID   :", message.ID)
                            print("Drone Label:", message.Label)
                            print("############################################")
                            
                            # Create a response LMCP message (echo back with modified label)
                            resp_msg = AirVehicleConfiguration.AirVehicleConfiguration()
                            resp_msg.ID = message.ID
                            resp_msg.Label = "Received: " + message.Label
                            response_data = LMCPFactory.packMessage(resp_msg, True)
                            client_socket.sendall(response_data)
                        else:
                            print("Received unknown LMCP message type.")
                    else:
                        print("No data received from client", addr)
    except PermissionError as e:
        print("PermissionError:", e)
        print("Ensure no firewall/antivirus is blocking the connection, or try a different port.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    start_server()
