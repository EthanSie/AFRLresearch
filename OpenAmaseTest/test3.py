import sys
import time
import socket

# 1) Make sure we can import the generated LMCP Python modules:
#    Adjust the path below to your "lmcp/py" folder if needed.
sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")

from lmcp import LMCPFactory
from afrl.cmasi import VehicleActionCommand, NavigationAction


HOST = '127.0.0.1'
PORT = 5555

def parse_lmcp_stream(data_buffer):
    """
    Given a bytes buffer, repeatedly attempt to parse LMCP messages.
    Returns (list_of_parsed_messages, leftover_bytes).
    """
    msgs = []
    while True:
        try:
            obj, size = LMCPFactory.getObject(data_buffer)
            if obj is None:
                break  # No (complete) message found
            msgs.append(obj)
            data_buffer = data_buffer[size:]
        except Exception:
            # Likely incomplete data; stop parsing for now
            break
    return msgs, data_buffer


def main():
    # -------------------------------------------------------------------------
    # 1) Construct a VehicleActionCommand
    #
    #    * We assume your scenario has an existing vehicle with ID=1.
    #    * This example sets a new desired altitude to 1500.0.
    #      When the scenario is running, you should see the vehicle climb or
    #      reflect this altitude in future AirVehicleState messages.
    # -------------------------------------------------------------------------
    cmd = VehicleActionCommand.VehicleActionCommand()
    cmd.VehicleIDList = [1]
    cmd.CommandID = 12345  # Arbitrary command ID
    # Mark it as pending (common practice)
    cmd.Status = 1  # 1 = Pending, 2 = Executed, 3 = Cancelled, etc.

    # Add a NavigationAction to, for example, set desired altitude
    nav = NavigationAction.NavigationAction()
    nav.DesiredAltitude = 1500.0
    nav.DesiredAltitudeType = 1  # MSL=1, AGL=2, etc.
    cmd.VehicleActionList = [nav]

    serialized_cmd = LMCPFactory.packMessage(cmd, True)

    # -------------------------------------------------------------------------
    # 2) Connect a plain TCP socket to AMASE
    # -------------------------------------------------------------------------
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to AMASE at {HOST}:{PORT}...")
        s.connect((HOST, PORT))

        # 3) Send the VehicleActionCommand
        s.sendall(serialized_cmd)
        print("Sent VehicleActionCommand (VehicleID=1, DesiredAltitude=1500.0).")
        print("Now listening for any LMCP messages (like AirVehicleState)...\n")

        # We'll read multiple chunks for ~10 seconds, parsing out LMCP messages
        buffer = b""
        s.settimeout(1.0)
        start_time = time.time()

        while time.time() - start_time < 10:
            try:
                chunk = s.recv(8192)
            except socket.timeout:
                # No data arrived this second, keep looping
                continue

            if not chunk:
                # Connection closed
                print("Socket closed by AMASE.")
                break

            # Accumulate data in our buffer
            buffer += chunk

            # Parse out one or more LMCP messages from the buffer
            parsed_msgs, buffer = parse_lmcp_stream(buffer)

            for m in parsed_msgs:
                # Print a short summary of what we received
                print(f"Received: {m.FULL_LMCP_TYPE_NAME}")

                # Example: if it's an AirVehicleState, show altitude
                if m.FULL_LMCP_TYPE_NAME == "afrl.cmasi.AirVehicleState":
                    print(f"  Vehicle ID: {m.ID}")
                    print(f"  Time:       {m.Time}")
                    print(f"  Latitude:   {m.Location.Latitude}")
                    print(f"  Longitude:  {m.Location.Longitude}")
                    print(f"  Altitude:   {m.Location.Altitude}")
                print("---")

        print("\nDone listening for LMCP messages.")


if __name__ == "__main__":
    main()
