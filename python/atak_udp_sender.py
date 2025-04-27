#!/usr/bin/env python3
"""
ATAK UDP Data Sender
-------------------
This script sends CoT (Cursor on Target) data to ATAK via UDP.
It can process both SDR/Axon and DJI Aeroscope data.
"""

import pandas as pd
import numpy as np
import socket
import time
import argparse
import logging
import os
import datetime
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ATAKUDPSender:
    """
    Sends CoT data to ATAK via UDP
    """
    def __init__(self, host, port):
        """
        Initialize the sender
        
        Parameters:
        - host: ATAK device IP address
        - port: UDP port (default is 4242 for ATAK)
        """
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"Initialized UDP sender for {host}:{port}")
    
    def send_cot_xml(self, xml_data):
        """
        Send CoT XML data to ATAK
        
        Parameters:
        - xml_data: CoT XML data as string
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            # Convert to bytes and send
            data_bytes = xml_data.encode('utf-8')
            self.socket.sendto(data_bytes, (self.host, self.port))
            return True
        except Exception as e:
            logger.error(f"Error sending CoT data: {e}")
            return False
    
    def send_cot_event(self, uid, lat, lon, event_type="a-f-G-U-C", 
                      callsign="Drone", remarks=None, team="Cyan"):
        """
        Send a single CoT event to ATAK
        
        Parameters:
        - uid: Unique identifier for the event
        - lat: Latitude
        - lon: Longitude
        - event_type: CoT event type
        - callsign: Contact callsign
        - remarks: Remarks text
        - team: Team color
        
        Returns:
        - True if successful, False otherwise
        """
        now = datetime.utcnow()
        time_str = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        stale_time = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Create CoT XML
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<event version="2.0" uid="{uid}" type="{event_type}" time="{time_str}" start="{time_str}" stale="{stale_time}" how="m-g">
    <point lat="{lat}" lon="{lon}" hae="0" ce="9999999" le="9999999"/>
    <detail>
        <contact callsign="{callsign}"/>
        <color argb="-1"/>
        <usericon iconsetpath="COT_MAPPING_2525B/a-f/a-f-G"/>
        <status readiness="true"/>
        <precisionlocation altsrc="DTED2"/>
        <track course="0.0" speed="0.0"/>
        <remarks>{remarks if remarks else ''}</remarks>
    </detail>
</event>"""
        
        # Send to ATAK
        return self.send_cot_xml(xml)
    
    def send_cot_file(self, filepath, delay=1.0, max_events=None, demo_mode=True):
        """
        Send a CoT XML file to ATAK, one event at a time
        
        Parameters:
        - filepath: Path to CoT XML file
        - delay: Delay between events in seconds
        - max_events: Maximum number of events to send (None for all)
        - demo_mode: If True, use a small subset of events for demo
        
        Returns:
        - Number of events sent
        """
        try:
            # Parse XML file
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Count total events
            events = root.findall(".//event")
            total_events = len(events)
            
            # For demo, limit the number of events
            if demo_mode and max_events is None:
                max_events = min(20, total_events)  # Default to 20 events in demo mode
            
            # Apply max_events limit if specified
            if max_events is not None and max_events < total_events:
                # Use evenly distributed events if we're sampling
                if total_events > max_events * 2:
                    step = total_events // max_events
                    events = events[::step][:max_events]
                else:
                    events = events[:max_events]
                total_to_send = len(events)
                logger.info(f"Sending {total_to_send} of {total_events} events from {filepath}")
            else:
                total_to_send = total_events
                logger.info(f"Sending all {total_events} events from {filepath}")
            
            # Process each event
            for i, event in enumerate(events):
                # Convert event to string
                event_str = ET.tostring(event, encoding='utf-8').decode('utf-8')
                
                # Add XML header
                xml_data = '<?xml version="1.0" encoding="UTF-8"?>\n' + event_str
                
                # Send to ATAK
                self.send_cot_xml(xml_data)
                
                # Log progress
                if (i+1) % 5 == 0 or i+1 == total_to_send:
                    logger.info(f"Sent {i+1}/{total_to_send} events")
                
                # Delay between events
                time.sleep(delay)
            
            return total_to_send
        except Exception as e:
            logger.error(f"Error sending CoT file: {e}")
            return 0

def process_data_to_cot(harmonizer_path, rf_file, dji_file, output_dir="cot_data"):
    """
    Process RF and DJI data to CoT XML files
    
    Parameters:
    - harmonizer_path: Path to ocusync_harmonizer.py
    - rf_file: Path to RF data CSV file
    - dji_file: Path to DJI data CSV file
    - output_dir: Output directory for CoT XML files
    
    Returns:
    - Tuple of (sdr_cot_file, dji_cot_file)
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set output file paths
        sdr_harmonized_file = os.path.join(output_dir, "sdr_harmonized.csv")
        sdr_cot_file = os.path.join(output_dir, "sdr_drones.xml")
        dji_cot_file = os.path.join(output_dir, "dji_drones.xml")
        
        # Run harmonizer on RF data
        cmd = f"python {harmonizer_path} --input {rf_file} --output {sdr_harmonized_file} " \
              f"--export-cot {sdr_cot_file} --source-type SDR " \
              f"--dji-input {dji_file} --dji-cot {dji_cot_file}"
        
        logger.info(f"Running harmonizer: {cmd}")
        os.system(cmd)
        
        # Check if files were created
        if not os.path.exists(sdr_cot_file):
            logger.error(f"SDR CoT file not created: {sdr_cot_file}")
            sdr_cot_file = None
        
        if not os.path.exists(dji_cot_file):
            logger.error(f"DJI CoT file not created: {dji_cot_file}")
            dji_cot_file = None
        
        return sdr_cot_file, dji_cot_file
    except Exception as e:
        logger.error(f"Error processing data to CoT: {e}")
        return None, None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ATAK UDP Data Sender')
    parser.add_argument('--host', type=str, required=True, help='ATAK device IP address')
    parser.add_argument('--port', type=int, default=4242, help='UDP port (default: 4242)')
    parser.add_argument('--rf-file', type=str, help='Path to RF data CSV file')
    parser.add_argument('--dji-file', type=str, help='Path to DJI data CSV file')
    parser.add_argument('--sdr-cot', type=str, help='Path to SDR CoT XML file')
    parser.add_argument('--dji-cot', type=str, help='Path to DJI CoT XML file')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between events (seconds)')
    parser.add_argument('--max-events', type=int, help='Maximum number of events to send from each file')
    parser.add_argument('--no-demo-mode', action='store_true', help='Disable demo mode (send all events)')
    parser.add_argument('--harmonizer', type=str, default='ocusync_harmonizer.py', 
                       help='Path to harmonizer script')
    
    args = parser.parse_args()
    
    # Initialize UDP sender
    sender = ATAKUDPSender(args.host, args.port)
    
    # Check if we need to process data
    if (args.rf_file or args.dji_file) and not (args.sdr_cot and args.dji_cot):
        # Process data to CoT
        sdr_cot, dji_cot = process_data_to_cot(
            args.harmonizer,
            args.rf_file or 'Site1.csv',
            args.dji_file or 'Site1_DJI_Data.csv'
        )
        
        # Set CoT files
        if not args.sdr_cot:
            args.sdr_cot = sdr_cot
        if not args.dji_cot:
            args.dji_cot = dji_cot
    
    # Send CoT data
    if args.sdr_cot and os.path.exists(args.sdr_cot):
        logger.info(f"Sending SDR data from {args.sdr_cot}...")
        sender.send_cot_file(
            args.sdr_cot, 
            args.delay, 
            max_events=args.max_events, 
            demo_mode=not args.no_demo_mode
        )
    
    if args.dji_cot and os.path.exists(args.dji_cot):
        logger.info(f"Sending DJI data from {args.dji_cot}...")
        sender.send_cot_file(
            args.dji_cot, 
            args.delay, 
            max_events=args.max_events, 
            demo_mode=not args.no_demo_mode
        )
    
    logger.info("Data transmission complete")

if __name__ == "__main__":
    main()