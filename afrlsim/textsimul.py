import json
import math
from statemachine import StateMachine, State
import random
from datetime import datetime

#region Helper Functions
def haversine(lat1, lon1, lat2, lon2):
    """Calculate geographic distance in kilometers."""
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2) * math.sin(dLat/2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon/2) * math.sin(dLon/2))
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_entities(json_file):
    """Load entities from JSON observation data."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['observation']['entities']

def deduplicate_entities(entities):
    """Return a list of unique entities keyed by their id."""
    unique = {}
    for e in entities:
        # If an entity with the same id already exists, you could decide whether
        # to replace it or keep the first one. Here we keep the first occurrence.
        if e['id'] not in unique:
            unique[e['id']] = e
    return list(unique.values())
#endregion

#region State Machines
class SensorStateMachine(StateMachine):
    idle = State('Idle', initial=True)
    scanning = State('Scanning')
    updating = State('Updating')

    scan = idle.to(scanning) | updating.to(scanning)
    detect_threat = scanning.to(updating)
    finish_update = updating.to(idle)

    def __init__(self, sensor_id):
        super().__init__()
        self.sensor_id = sensor_id
        self.detected_threats = []

    def safe_scan(self):
        """Handle scan attempts with state checking"""
        if self.current_state in [self.idle, self.updating]:
            self.scan()

class SAMStateMachine(StateMachine):
    idle = State('Idle', initial=True)
    preparing = State('Preparing')
    ready = State('Ready')
    firing = State('Firing')
    reloading = State('Reloading')

    prepare = idle.to(preparing)
    ready_up = preparing.to(ready)
    fire = ready.to(firing)
    reload = firing.to(reloading) | idle.to(reloading)
    reset = reloading.to(idle)

    def __init__(self, sam_id, ammo=3):
        super().__init__()
        self.sam_id = sam_id
        self.ammo = ammo
        self.target = None
        self.reload_time = 0

    def safe_prepare(self):
        if self.current_state == self.idle and self.ammo > 0:
            self.prepare()

    def safe_fire(self, target):
        if self.current_state == self.ready and self.ammo > 0:
            self.target = target
            self.fire()
            self.ammo -= 1
            return True
        return False
#endregion

#region Core Simulation
class Threat:
    def __init__(self, entity_data, base_lat, base_lon):
        self.id = entity_data['id']
        self.label = entity_data['label']
        self.lat = entity_data['location']['latitude']
        self.lon = entity_data['location']['longitude']
        self.distance = haversine(base_lat, base_lon, self.lat, self.lon)
        self.destroyed = False

class AFSIMSimulator:
    def __init__(self, scenario_json):
        raw_entities = load_entities(scenario_json)
        self.entities = deduplicate_entities(raw_entities)
        # Find the base (type "C2") to get its location
        self.base = next(e for e in self.entities if e['type'] == 'C2')
        self.base_lat = self.base['location']['latitude']
        self.base_lon = self.base['location']['longitude']
        
        # Initialize systems using unique entities
        self.sensors = [
            SensorStateMachine(s['id'])
            for s in self.entities if s['type'] == 'RADAR'
        ]
        # Only take 2 SAMs for our side (2 SAM missiles)
        self.sams = [
            SAMStateMachine(s['id'])
            for s in self.entities if s['type'] == 'SAM'
        ][:2]
        # Create threats from red side entities
        self.threats = [
            Threat(e, self.base_lat, self.base_lon)
            for e in self.entities if e['side'] == 'red'
        ]
        # Add an extra threat with custom coordinates (a different distance)
        extra_threat_data = {
            "id": "Extra_1",
            "label": "Extra Threat",
            "location": {
                # Choose coordinates offset from the base
                "latitude": self.base_lat - 0.05,
                "longitude": self.base_lon - 0.05
            }
        }
        self.threats.append(Threat(extra_threat_data, self.base_lat, self.base_lon))
        
        self.timestep = 0
        self.engagement_log = []

    def run_step(self):
        """Run one timestep of the simulation"""
        self.timestep += 1
        print(f"\n=== Timestep {self.timestep} ===")
        
        # 1. Update sensor states
        for sensor in self.sensors:
            try:
                sensor.safe_scan()
                if sensor.current_state == sensor.scanning:
                    if random.random() < 0.3:  # Detection chance
                        sensor.detect_threat()
                if sensor.current_state == sensor.updating:
                    sensor.finish_update()
            except Exception as e:
                print(f"Sensor {sensor.sensor_id} error: {str(e)}")
        
        # 2. Update threats (move closer to base)
        for threat in self.threats:
            if not threat.destroyed:
                threat.distance = max(0, threat.distance - random.randint(1, 5))
        
        # 3. Update SAM states
        for sam in self.sams:
            if sam.current_state == sam.reloading:
                sam.reload_time -= 1
                if sam.reload_time <= 0:
                    sam.reset()
                    sam.ammo = 3
        
        # 4. Display status of both sides
        self.display_status()
        
        # 5. Get user action
        self.handle_user_input()
        
        # 6. Log state
        self.log_engagement()
    
    def display_status(self):
        print("\n[Allied Assets]")
        print(f"Base at ({self.base_lat:.4f}, {self.base_lon:.4f})")
        print("Sensors:")
        for sensor in self.sensors:
            print(f"  Sensor {sensor.sensor_id}: {sensor.current_state}")
        print("SAMs:")
        for sam in self.sams:
            print(f"  SAM {sam.sam_id}: {sam.current_state} | Ammo: {sam.ammo}")
        
        print("\n[Enemy Threats]")
        for t in self.threats:
            status = "DESTROYED" if t.destroyed else f"{t.distance:.1f} km"
            print(f"  {t.label} ({t.id}): {status}")

    def handle_user_input(self):
        print("\nChoose action:")
        print("1. Engage threat")
        print("2. Reload SAM")
        print("3. Skip")
        choice = input("> ")
        
        if choice == '1':
            target = next((t for t in self.threats if not t.destroyed), None)
            if target:
                for sam in self.sams:
                    sam.safe_prepare()
                    if sam.current_state == sam.preparing:
                        sam.ready_up()
                        if sam.safe_fire(target):
                            # A 70% chance the threat is destroyed
                            target.destroyed = random.random() < 0.7
                            break
        elif choice == '2':
            for sam in self.sams:
                if sam.current_state == sam.idle:
                    sam.reload()
                    sam.reload_time = 2

    def log_engagement(self):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "threats": [t.__dict__ for t in self.threats],
            "sam_states": [s.current_state.value for s in self.sams]
        }
        self.engagement_log.append(log_entry)

    def check_simulation_over(self):
        """Stop the simulation if all enemy threats are destroyed or an enemy reaches the base."""
        enemy_remaining = [t for t in self.threats if not t.destroyed]
        if len(enemy_remaining) == 0:
            print("\nAll enemy threats are destroyed. You have neutralized the enemy artillery. Victory!")
            return True
        # Check if any threat is at zero distance (or very close)
        if any(t.distance <= 0 for t in enemy_remaining):
            print("\nAn enemy threat has reached the base. The enemy has overrun your defenses. Defeat!")
            return True
        return False
#endregion

#region Execution
if __name__ == "__main__":
    sim = AFSIMSimulator("simulations.json")
    
    while True:
        try:
            sim.run_step()
            if sim.check_simulation_over():
                break
            cont = input("\nContinue? (y/n): ")
            if cont.lower() != 'y':
                break
        except Exception as e:
            print(f"Simulation error: {str(e)}")
            break
    
    print("\n=== Simulation Complete ===")
    print(f"Total steps: {sim.timestep}")
    print("Engagement log saved to engagement_log.json")
    with open("engagement_log.json", 'w') as f:
        json.dump(sim.engagement_log, f)
#endregion
