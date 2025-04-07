import sys
import time
import socket

sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")

from lmcp import LMCPFactory
from afrl.cmasi import AirVehicleConfiguration, AirVehicleState

HOST = '127.0.0.1'
PORT = 5555

def main():
    # ----------------------------------------------------------------------
    # 1) Create an AirVehicleConfiguration for a brand-new UAV (ID=99).
    #    This "spawns" a new UAV in AMASE if the scenario/plugins allow it.
    # ----------------------------------------------------------------------
    config_msg = AirVehicleConfiguration.AirVehicleConfiguration()
    config_msg.ID = 99
    config_msg.Label = "GIGANTIC_UAV_99"
    config_msg.VehicleType = "SuperDrone"
    # Optional flight performance fields:
    config_msg.MinimumSpeed = 10.0
    config_msg.MaximumSpeed = 250.0
    config_msg.NominalFlightProfile = None  # can omit or define FlightProfile
    # etc.

    # ----------------------------------------------------------------------
    # 2) Create an AirVehicleState so the new UAV has a position, altitude, etc.
    #    ID must match the config.
    # ----------------------------------------------------------------------
    state_msg = AirVehicleState.AirVehicleState()
    state_msg.ID = 99
    # We'll place it at some obvious coordinate:
    state_msg.Position.Latitude = 40.0
    state_msg.Position.Longitude = -120.0
    state_msg.Position.Altitude = 1000.0
    state_msg.Airspeed = 50.0  # knots or m/s depending on your AMASE setup
    state_msg.Heading = 90.0   # East
    # Optionally set more state fields if desired.

    # Serialize both messages (raw LMCP)
    config_data = LMCPFactory.packMessage(config_msg, True)
    state_data = LMCPFactory.packMessage(state_msg, True)

    # ----------------------------------------------------------------------
    # 3) Connect to AMASE over TCP and send both messages.
    # ----------------------------------------------------------------------
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to AMASE at {HOST}:{PORT}...")
        s.connect((HOST, PORT))

        # First send the AirVehicleConfiguration
        s.sendall(config_data)
        print("Sent AirVehicleConfiguration for UAV ID=99.")

        # Then send the AirVehicleState
        s.sendall(state_data)
        print("Sent AirVehicleState for UAV ID=99.")

        # ------------------------------------------------------------------
        # 4) Read for ~10 seconds to see if AMASE publishes any responses
        #    (like AirVehicleState or SessionStatus). Not all scenarios do.
        # ------------------------------------------------------------------
        s.settimeout(1.0)
        start_time = time.time()
        buffer = b""

        while time.time() - start_time < 10:
            try:
                chunk = s.recv(8192)
            except socket.timeout:
                continue
            if not chunk:
                print("Socket closed by AMASE.")
                break

            buffer += chunk
            while True:
                try:
                    obj, size = LMCPFactory.getObject(buffer)
                    if obj is None:
                        break
                    buffer = buffer[size:]
                    print("Received:", obj.FULL_LMCP_TYPE_NAME)
                except:
                    break

    print("Done. Check AMASE GUI for UAV labeled 'GIGANTIC_UAV_99' at lat=40, lon=-120, alt=1000.")

if __name__ == "__main__":
    main()
