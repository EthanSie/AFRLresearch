import sys
sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")

from lmcp import LMCPFactory
from afrl.cmasi import AirVehicleState

def main():
    # Create a test CMASI object
    state = AirVehicleState.AirVehicleState()
    state.ID = 123
    state.Altitude = 5000.0

    # Serialize the object
    data = LMCPFactory.packMessage(state, True)
    print("Serialized length:", len(data))

    # Deserialize using the internal factory instance
    obj = LMCPFactory.internalFactory.getObject(data)
    print("Deserialized:", obj)

if __name__ == "__main__":
    main()
