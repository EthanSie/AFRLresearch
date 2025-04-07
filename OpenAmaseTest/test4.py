import zmq, sys, time, copy, re
import xml.dom.minidom
from xml.dom.minidom import parse, parseString

# Include LMCP Python files
sys.path.insert(0, r"C:\Users\ethan\OneDrive\Desktop\AFRL\OpenAMASE-master\OpenAMASE-master\OpenAMASE\docs\lmcp\py")
from lmcp import LMCPFactory
from afrl.cmasi import *


def print_footprint(ID, PayloadID):
    ###################################################################################################################
    # Prepare a ZeroMQ context and socket, then connect to 5555. This is the port AMASE uses by convention.
    # The actual port is set by an XML element in the file Plugins.xml, whose location is specified with the AMASE
    # option --config <config_directory> when starting up AMASE. This XML element has the form:
    # <Plugin Class="avtas.amase.network.TcpServer">
    #    <TCPServer Port="5555"/>
    # </Plugin>
    # -----------------------------------------------------------------------------------------------------------------

    context = zmq.Context()
    socket = context.socket(zmq.STREAM)
    socket.connect("tcp://127.0.0.1:5555")
    # Store the client ID for this socket
    client_id, message = socket.recv_multipart()

    ###################################################################################################################
    # Initialize an LMCPFactory for creating LMCP messages
    # -----------------------------------------------------------------------------------------------------------------

    factory = LMCPFactory.LMCPFactory()

    ###################################################################################################################
    # Get messages from AMASE. Check for AirVehicleState messages from the specified air vehicle ID, then get the
    # PayloadState for the specified PayloadID. Assuming it has a Footprint, list the Footprint's Location3D points.
    # -----------------------------------------------------------------------------------------------------------------

    while True:
        msg_obj = get_from_amase(socket, factory)
        if msg_obj.FULL_LMCP_TYPE_NAME != 'afrl.cmasi.AirVehicleState':
            continue;
        if msg_obj.get_ID() != ID:
            continue;
        PayloadStateList = msg_obj.get_PayloadStateList()
        Footprint = []
        for PayloadState in msg_obj.get_PayloadStateList():
            if PayloadState.get_PayloadID() == PayloadID and hasattr(PayloadState, 'get_Footprint'):
                Footprint = PayloadState.get_Footprint()

        if len(Footprint):
            print("\nFootprint values for time " + str(msg_obj.get_Time()))
            for loc3d in Footprint:
                print("Lat: " + str(loc3d.get_Latitude()))
                print("Lon: " + str(loc3d.get_Longitude()))
                print("Alt: " + str(loc3d.get_Altitude()))

def send_to_amase(obj, socket, client_id):
    """
    Send an LCMP object to AMASE.

    Keyword arguments:
        obj -- an LMCP message object
        socket -- a ZMQ socket connected to AMASE
        client_id -- the client id for the AMASE socket
    """

    attributes = bytearray(str(obj.FULL_LMCP_TYPE_NAME) + "$lmcp|" + str(obj.FULL_LMCP_TYPE_NAME) + "||0|0$",
                           'ascii')
    smsg = LMCPFactory.packMessage(obj, True)

    sentinel = "+=+=+=+=";
    sentinel += str(len(attributes) + len(smsg));
    sentinel += "#@#@#@#@";

    addressedPayload = attributes;
    addressedPayload.extend(smsg);

    # sentinelized checksum
    val = 0;
    for i in range(0, len(addressedPayload)):
        val += int(addressedPayload[i] & 0xFF);

    footer = "!%!%!%!%" + str(val) + "?^?^?^?^";

    totalmsg = bytearray(sentinel, 'ascii');
    totalmsg.extend(addressedPayload);
    totalmsg.extend(bytearray(footer, 'ascii'));

    socket.send(client_id, flags=zmq.SNDMORE, copy=False)
    socket.send(totalmsg, copy=False)
    print("  Sent:   " + obj.FULL_LMCP_TYPE_NAME)


def get_from_amase(socket, factory):
    """
    Get an LCMP object from AMASE.

    Keyword arguments:
        socket -- a ZMQ socket connected to AMASE
        factory -- an LMCP factory
    """

    client_id, message = socket.recv_multipart()
    address, attributes, msg = message.split(bytearray('$', 'ascii'), 2)
    # msg_format, msg_type, msg_group, entityid, serviceid = attributes.split(bytearray('|', 'ascii'), 4)
    return factory.getObject(msg)


def lmcp_toprettyxml(obj):
    """
    Convert an LCMP object to a string that formats correctly with xml.dom.minidom's toprettyxml().

    Keyword arguments:
        obj -- an LMCP message object
    """

    # toprettyxml() assumes no newlines or whitespace between XML elements,
    # so remove them from what LMCP's toXML() returns.
    return re.sub(r'\n(\s)*', '', obj.toXML())


if __name__ == '__main__':
    ID = 500
    PayloadID = 10001
    print_footprint(ID, PayloadID)