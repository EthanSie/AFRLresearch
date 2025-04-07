#!/usr/bin/env python3
"""
scenario_example2.py

Scenario:
  Friendly assets: one SAM site and one Laser system.
  Enemy targets: one "Large Slow Drone" and one "Small Fast Drone".

Tactical notes:
  - Laser can reliably neutralize a small fast drone but cannot neutralize a large slow drone.
  - SAM can reliably hit a large slow drone, but its chance to hit a small fast drone is lower.
  - If both fire, they won’t have another shot.
  - The decision should favor future rewards over immediate engagement.

This file logs state transitions and displays messages when targets are hit or missed.
"""

import json
import math
import random
from datetime import datetime
import pygame
from statemachine import StateMachine, State

# ---------------------- Constants ----------------------
ENGAGEMENT_RANGE = 2.0  # km

# ---------------------- Helper Functions ----------------------
def haversine(lat1, lon1, lat2, lon2):
    """Calculate geographic distance in kilometers."""
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon/2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_entities(json_file):
    """Load entities from JSON observation data."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['observation']['entities']

def deduplicate_entities(entities, key=lambda e: (e['id'], e['type'])):
    """Return a list of unique entities based on a key function."""
    seen = set()
    unique = []
    for e in entities:
        k = key(e)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique

def geo_offset_km(lat, lon, base_lat, base_lon):
    """Compute approximate x,y offset (in km) from base given lat/lon differences."""
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(base_lat))
    dx = (lon - base_lon) * km_per_deg_lon
    dy = (lat - base_lat) * km_per_deg_lat
    return dx, dy

def geo_to_screen(lat, lon, base_lat, base_lon, center, scale):
    """Convert geographic coordinates to screen coordinates."""
    dx, dy = geo_offset_km(lat, lon, base_lat, base_lon)
    screen_x = center[0] + dx * scale
    screen_y = center[1] - dy * scale  # invert y for screen coordinates
    return (int(screen_x), int(screen_y))

def km_to_deg_lat(km):
    return km / 111.32

def km_to_deg_lon(km, at_lat):
    return km / (111.32 * math.cos(math.radians(at_lat)))

# ---------------------- State Machines ----------------------
class SensorStateMachine(StateMachine):
    idle = State('Idle', initial=True)
    scanning = State('Scanning')
    updating = State('Updating')
    
    scan = idle.to(scanning) | updating.to(scanning)
    detect_threat = scanning.to(updating)
    finish_update = updating.to(idle)
    
    def __init__(self, sensor_entity):
        super().__init__()
        self.sensor_id = sensor_entity['id']
        self.pos = sensor_entity['location']
        self.detected_threats = []
    
    def safe_scan(self):
        if self.current_state in [self.idle, self.updating]:
            self.scan()

class SAMStateMachine(StateMachine):
    idle = State('Idle', initial=True)
    preparing = State('Preparing')
    ready = State('Ready')
    firing = State('Firing')
    reloading = State('Reloading')
    
    # Transition to complete firing.
    complete = firing.to(idle)
    
    prepare = idle.to(preparing)
    ready_up = preparing.to(ready)
    fire = ready.to(firing)
    reload = firing.to(reloading) | idle.to(reloading)
    reset = reloading.to(idle)
    
    def __init__(self, sam_entity, ammo=1):
        super().__init__()
        self.sam_id = sam_entity['id']
        self.pos = sam_entity['location']
        self.ammo = ammo
        self.target = None
        self.last_engaged_target = None
        self.reload_time = 0
    
    def safe_prepare(self):
        if self.current_state == self.idle and self.ammo > 0:
            self.prepare()
    
    def safe_fire(self, target):
        if self.current_state == self.ready and self.ammo > 0:
            self.target = target
            self.last_engaged_target = target
            self.fire()
            self.ammo -= 1
            return True
        return False

class LaserStateMachine(StateMachine):
    idle = State('Idle', initial=True)
    firing = State('Firing')
    
    # Transitions for Laser.
    fire = idle.to(firing)
    complete = firing.to(idle)
    
    def __init__(self, laser_entity, ammo=1):
        super().__init__()
        self.laser_id = laser_entity['id']
        self.pos = laser_entity['location']
        self.ammo = ammo
        self.target = None
        self.last_engaged_target = None
    
    def safe_fire(self, target):
        # Laser fires only if the target is a small fast drone.
        if self.current_state == self.idle and self.ammo > 0:
            if target.drone_type == "small_fast":
                self.target = target
                self.last_engaged_target = target
                self.fire()
                self.ammo -= 1
                return True
            else:
                return False
        return False

# ---------------------- Core Simulation Classes ----------------------
class Threat:
    def __init__(self, entity_data, base_lat, base_lon):
        self.id = entity_data['id']
        self.label = entity_data['label']
        self.lat = entity_data['location']['latitude']
        self.lon = entity_data['location']['longitude']
        self.drone_type = entity_data.get('drone_type', None)
        self.distance = haversine(base_lat, base_lon, self.lat, self.lon)
        self.initial_distance = self.distance if self.distance > 0 else 1
        self.offset = geo_offset_km(self.lat, self.lon, base_lat, base_lon)
        self.destroyed = False
    
    def get_current_position(self, base_lat, base_lon):
        fraction = self.distance / self.initial_distance
        current_lat = base_lat + fraction * (self.lat - base_lat)
        current_lon = base_lon + fraction * (self.lon - base_lon)
        return current_lat, current_lon
    
    def get_screen_pos(self, base_screen, scale):
        fraction = self.distance / self.initial_distance
        cur_dx = self.offset[0] * fraction
        cur_dy = self.offset[1] * fraction
        screen_x = base_screen[0] + cur_dx * scale
        screen_y = base_screen[1] - cur_dy * scale
        return (int(screen_x), int(screen_y))

# ---------------------- Simulator ----------------------
class AFSIMSimulator:
    def __init__(self, scenario):
        # Accept scenario as filename or dictionary.
        if isinstance(scenario, str):
            from_file = load_entities(scenario)
        else:
            from_file = scenario["observation"]["entities"]
        self.entities = deduplicate_entities(from_file, key=lambda e: (e['id'], e['type']))
        
        self.base = next(e for e in self.entities if e['type'] == 'C2')
        self.base_lat = self.base['location']['latitude']
        self.base_lon = self.base['location']['longitude']
        
        self.sensors = [SensorStateMachine(e) for e in self.entities if e['type'] == 'RADAR']
        # Expect one SAM and one Laser.
        self.sams = [SAMStateMachine(e, ammo=1) for e in self.entities if e['type'] == 'SAM']
        self.lasers = [LaserStateMachine(e, ammo=1) for e in self.entities if e['type'] == 'LASER']
        
        red_entities = deduplicate_entities([e for e in self.entities if e['side'] == 'red'],
                                             key=lambda e: (e['id'], e.get('label')))
        self.threats = [Threat(e, self.base_lat, self.base_lon) for e in red_entities]
        
        self.timestep = 0
        self.engagement_log = []
        self.messages = []
    
    def add_message(self, msg, duration=3000):
        expire = pygame.time.get_ticks() + duration
        self.messages.append({"text": msg, "expire": expire})
    
    def is_threat_visible(self, threat):
        current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
        friendly_units = self.sensors + self.sams + self.lasers
        for unit in friendly_units:
            d = haversine(unit.pos['latitude'], unit.pos['longitude'], current_lat, current_lon)
            if d <= ENGAGEMENT_RANGE:
                return True
        return False
    
    def fire_sam(self):
        if not self.sams:
            return
        sam = self.sams[0]
        if sam.current_state == sam.idle and sam.ammo > 0:
            target = None
            for t in self.threats:
                if not t.destroyed:
                    current_lat, current_lon = t.get_current_position(self.base_lat, self.base_lon)
                    dist = haversine(sam.pos['latitude'], sam.pos['longitude'], current_lat, current_lon)
                    if dist <= ENGAGEMENT_RANGE:
                        target = t
                        print(f"SAM firing at target {target.label} at distance {dist:.2f} km")
                        break
            if target is not None:
                sam.safe_prepare()
                if sam.current_state == sam.preparing:
                    sam.ready_up()
                    if sam.safe_fire(target):
                        hit = True if target.drone_type == "large_slow" else (random.random() < 0.5)
                        if hit:
                            target.destroyed = True
                            self.add_message(f"Target {target.label} destroyed by SAM.")
                        else:
                            self.add_message(f"SAM missed target {target.label}.")
                        sam.complete()
            else:
                print("SAM: No target in range.")
    
    def fire_laser(self):
        if not self.lasers:
            return
        laser = self.lasers[0]
        if laser.current_state == laser.idle and laser.ammo > 0:
            target = None
            for t in self.threats:
                if not t.destroyed:
                    current_lat, current_lon = t.get_current_position(self.base_lat, self.base_lon)
                    dist = haversine(laser.pos['latitude'], laser.pos['longitude'], current_lat, current_lon)
                    if dist <= ENGAGEMENT_RANGE and t.drone_type == "small_fast":
                        target = t
                        print(f"Laser firing at target {target.label} at distance {dist:.2f} km")
                        break
            if target is not None:
                laser.fire()
                target.destroyed = True
                self.add_message(f"Target {target.label} destroyed by Laser.")
                laser.complete()
            else:
                print("Laser: No suitable target in range.")
    
    def process_action(self, action):
        if action == 'fire_sam':
            self.fire_sam()
        elif action == 'fire_laser':
            self.fire_laser()
        elif action == 'reload':
            for sam in self.sams:
                if sam.current_state == sam.idle:
                    sam.reload()
                    sam.reload_time = 2
    
    def run_step(self, action=None):
        self.timestep += 1
        for sensor in self.sensors:
            try:
                sensor.safe_scan()
                if sensor.current_state == sensor.scanning and random.random() < 0.3:
                    sensor.detect_threat()
                if sensor.current_state == sensor.updating:
                    sensor.finish_update()
            except Exception as e:
                print(f"Sensor {sensor.sensor_id} error: {str(e)}")
        for threat in self.threats:
            if not threat.destroyed:
                threat.distance = max(0, threat.distance - random.uniform(0.1, 0.5))
        for sam in self.sams:
            if sam.current_state == sam.reloading:
                sam.reload_time -= 1
                if sam.reload_time <= 0:
                    sam.reset()
                    sam.ammo = 1
            if sam.ammo == 0 and sam.current_state == sam.idle:
                sam.reload()
                sam.reload_time = 2
        if action is not None:
            self.process_action(action)
        self.log_engagement()
    
    def log_engagement(self):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "threats": [t.__dict__ for t in self.threats],
            "sam_states": [s.current_state.value for s in self.sams],
            "laser_states": [l.current_state.value for l in self.lasers]
        }
        self.engagement_log.append(log_entry)
    
    def check_simulation_over(self):
        enemy_remaining = [t for t in self.threats if not t.destroyed]
        if len(enemy_remaining) == 0:
            print("\nVictory! All enemy threats have been neutralized.")
            return True
        if any(t.distance == 0 for t in enemy_remaining):
            print("\nDefeat! An enemy threat has reached the base.")
            return True
        return False

# ---------------------- Pygame Drawing & Main Loop ----------------------
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AFSIM Simulator – Decision Scenario")
clock = pygame.time.Clock()

center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
scale = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# Create a specific scenario dictionary.
scenario = {
    "observation": {
        "entities": [
            {"id": 1, "type": "C2", "label": "Base", "location": {"latitude": 34.9295, "longitude": -117.8849}, "side": "blue"},
            {"id": 2, "type": "SAM", "label": "SAM Site", "location": {"latitude": 34.93, "longitude": -117.884}, "side": "blue"},
            {"id": 3, "type": "LASER", "label": "Laser", "location": {"latitude": 34.9295, "longitude": -117.8855}, "side": "blue"},
            {"id": 100, "type": "UAS", "label": "Large Slow Drone", "drone_type": "large_slow", "location": {"latitude": 34.935, "longitude": -117.882}, "side": "red"},
            {"id": 101, "type": "UAS", "label": "Small Fast Drone", "drone_type": "small_fast", "location": {"latitude": 34.923, "longitude": -117.887}, "side": "red"}
        ]
    }
}

sim = AFSIMSimulator(scenario)
current_action = None

SIM_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SIM_EVENT, 1000)

font = pygame.font.SysFont(None, 20)
sam_font = pygame.font.SysFont(None, 24)
legend_font = pygame.font.SysFont(None, 18)
message_font = pygame.font.SysFont(None, 22)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Keys: 1 to fire SAM, 2 to fire Laser, R to reload, S to skip.
            if event.key == pygame.K_1:
                current_action = 'fire_sam'
            elif event.key == pygame.K_2:
                current_action = 'fire_laser'
            elif event.key == pygame.K_r:
                current_action = 'reload'
            elif event.key == pygame.K_s:
                current_action = 'skip'
        elif event.type == SIM_EVENT:
            sim.run_step(action=current_action)
            current_action = None
            if sim.check_simulation_over():
                running = False
    
    screen.fill(WHITE)
    pygame.draw.circle(screen, BLACK, center, 8)
    
    # Draw friendly assets.
    for unit in sim.sams:
        pos = unit.pos
        unit_screen = geo_to_screen(pos['latitude'], pos['longitude'], sim.base_lat, sim.base_lon, center, scale)
        pygame.draw.circle(screen, GREEN, unit_screen, 8)
        pygame.draw.circle(screen, GRAY, unit_screen, int(scale * ENGAGEMENT_RANGE), 1)
    for unit in sim.lasers:
        pos = unit.pos
        unit_screen = geo_to_screen(pos['latitude'], pos['longitude'], sim.base_lat, sim.base_lon, center, scale)
        pygame.draw.circle(screen, GREEN, unit_screen, 8)
        pygame.draw.circle(screen, GRAY, unit_screen, int(scale * ENGAGEMENT_RANGE), 1)
    
    # Draw enemy threats only if visible.
    for threat in sim.threats:
        if not sim.is_threat_visible(threat):
            continue
        threat_screen = threat.get_screen_pos(center, scale)
        color = ORANGE if threat.label == "Extra Threat" else RED
        pygame.draw.circle(screen, color, threat_screen, 6)
        text = font.render(f"{threat.label} {threat.distance:.1f}km", True, BLACK)
        screen.blit(text, (threat_screen[0] + 10, threat_screen[1]))
    
    legend_lines = [
        "Legend:",
        "Base: Black circle",
        "SAM: Green circle (Gray outline = 2km range)",
        "Laser: Green circle (Gray outline = 2km range)",
        "Large Slow Drone: Red circle",
        "Small Fast Drone: Red circle (target for Laser)",
        "Keys: 1-Fire SAM, 2-Fire Laser, R-Reload, S-Skip"
    ]
    for i, line in enumerate(legend_lines):
        legend_text = legend_font.render(line, True, BLACK)
        screen.blit(legend_text, (10, 10 + i * 20))
    
    sam_title = sam_font.render("SAM Status", True, BLACK)
    screen.blit(sam_title, (WINDOW_WIDTH - 200, 10))
    for i, unit in enumerate(sim.sams):
        status_text = f"SAM {unit.sam_id}: Ammo {unit.ammo}"
        sam_text = sam_font.render(status_text, True, BLACK)
        screen.blit(sam_text, (WINDOW_WIDTH - 200, 40 + i * 30))
    laser_title = sam_font.render("Laser Status", True, BLACK)
    screen.blit(laser_title, (WINDOW_WIDTH - 200, 100))
    for i, unit in enumerate(sim.lasers):
        status_text = f"Laser {unit.laser_id}: Ammo {unit.ammo}"
        laser_text = sam_font.render(status_text, True, BLACK)
        screen.blit(laser_text, (WINDOW_WIDTH - 200, 130 + i * 30))
    
    current_ticks = pygame.time.get_ticks()
    sim.messages = [m for m in sim.messages if m["expire"] > current_ticks]
    for i, message in enumerate(sim.messages):
        msg_text = message_font.render(message["text"], True, BLACK)
        screen.blit(msg_text, (10, WINDOW_HEIGHT - 100 - i * 25))
    
    pygame.display.flip()
    clock.tick(60)

with open("engagement_log.json", 'w') as f:
    json.dump(sim.engagement_log, f)
pygame.quit()
