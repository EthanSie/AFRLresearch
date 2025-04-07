#!/usr/bin/env python3
"""
scenario_decision_missile2.py

Scenario:
  - The base has 10 hit points.
  - Friendly asset: one interceptor system with a random ammo count (2–4).
  - Enemy threats:
      * Big drones (each causes 5 damage if they reach the base).
         Their count is either equal to the interceptor’s ammo or one less.
      * Six small drones (each causes 1 damage if they hit the base).
      * One enemy missile (if it hits, it destroys the base).
      
Decision challenge:
  The agent must decide which interceptor rounds to fire:
    - Reserve at least one round for the enemy missile (a high‐priority target) if it’s imminent.
    - Engage big drones when extra rounds are available.
    - Optionally engage small drones only if enough rounds remain.
  
If enemy threats reach the base (distance nearly 0), they inflict damage.
The simulation ends with Victory if all threats are neutralized or Defeat if the base HP reaches 0.
"""

import json
import math
import random
from datetime import datetime
import pygame
from statemachine import StateMachine, State

# ---------------------- Constants ----------------------
ENGAGEMENT_RANGE = 2.0  # km for detection/engagement
BASE_HP_INITIAL = 10

# ---------------------- Helper Functions ----------------------
def haversine(lat1, lon1, lat2, lon2):
    """Calculate geographic distance in kilometers."""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon/2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def geo_offset_km(lat, lon, base_lat, base_lon):
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(base_lat))
    dx = (lon - base_lon) * km_per_deg_lon
    dy = (lat - base_lat) * km_per_deg_lat
    return dx, dy

def geo_to_screen(lat, lon, base_lat, base_lon, center, scale):
    dx, dy = geo_offset_km(lat, lon, base_lat, base_lon)
    screen_x = center[0] + dx * scale
    screen_y = center[1] - dy * scale
    return (int(screen_x), int(screen_y))

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

class InterceptorStateMachine(StateMachine):
    idle = State('Idle', initial=True)
    preparing = State('Preparing')
    ready = State('Ready')
    firing = State('Firing')
    reloading = State('Reloading')
    
    complete = firing.to(idle)
    prepare = idle.to(preparing)
    ready_up = preparing.to(ready)
    fire = ready.to(firing)
    reload = firing.to(reloading) | idle.to(reloading)
    reset = reloading.to(idle)
    
    def __init__(self, interceptor_entity, ammo=1):
        super().__init__()
        self.interceptor_id = interceptor_entity['id']
        self.pos = interceptor_entity['location']
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

# ---------------------- Core Simulation Classes ----------------------
class Threat:
    def __init__(self, entity_data, base_lat, base_lon):
        self.id = entity_data['id']
        self.label = entity_data['label']
        self.lat = entity_data['location']['latitude']
        self.lon = entity_data['location']['longitude']
        self.drone_type = entity_data.get('drone_type', None)  # "big", "small", or "missile"
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
        # Accept scenario as a dictionary.
        self.entities = scenario["observation"]["entities"]
        
        self.base = next(e for e in self.entities if e['type'] == 'C2')
        self.base_lat = self.base['location']['latitude']
        self.base_lon = self.base['location']['longitude']
        self.base_hp = BASE_HP_INITIAL
        
        # Create friendly interceptor asset (we assume one asset with a total ammo count).
        # Set its ammo from the scenario if provided; otherwise, we assign random ammo.
        interceptor_entity = next(e for e in self.entities if e['type'] == 'INTERCEPTOR')
        if 'ammo' in interceptor_entity:
            ammo = interceptor_entity['ammo']
        else:
            ammo = random.randint(2, 4)
            interceptor_entity['ammo'] = ammo
        self.interceptors = [InterceptorStateMachine(interceptor_entity, ammo=ammo)]
        
        # Enemy threats: filter for those with side "red".
        self.threats = [Threat(e, self.base_lat, self.base_lon) for e in self.entities if e['side'] == 'red']
        
        self.timestep = 0
        self.engagement_log = []
        self.messages = []
    
    def add_message(self, msg, duration=3000):
        expire = pygame.time.get_ticks() + duration
        self.messages.append({"text": msg, "expire": expire})
    
    def is_threat_visible(self, threat):
        # For simplicity, threats are visible if within ENGAGEMENT_RANGE of the interceptor.
        current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
        for unit in self.interceptors:
            d = haversine(unit.pos['latitude'], unit.pos['longitude'], current_lat, current_lon)
            if d <= ENGAGEMENT_RANGE:
                return True
        return False
    
    def fire_interceptor_for_missile(self):
        # Prioritize the enemy missile.
        enemy_missiles = [t for t in self.threats if (not t.destroyed) and t.drone_type == "missile"]
        if not enemy_missiles:
            return False
        missile = enemy_missiles[0]
        for interceptor in self.interceptors:
            if interceptor.current_state == interceptor.idle and interceptor.ammo > 0:
                current_lat, current_lon = missile.get_current_position(self.base_lat, self.base_lon)
                dist = haversine(interceptor.pos['latitude'], interceptor.pos['longitude'], current_lat, current_lon)
                if dist <= ENGAGEMENT_RANGE:
                    print(f"Interceptor {interceptor.interceptor_id} firing at enemy missile {missile.label} at {dist:.2f} km")
                    interceptor.safe_prepare()
                    if interceptor.current_state == interceptor.preparing:
                        interceptor.ready_up()
                        if interceptor.safe_fire(missile):
                            hit = random.random() < 0.9  # 90% hit chance for missile.
                            if hit:
                                missile.destroyed = True
                                self.add_message(f"Enemy missile {missile.label} destroyed by interceptor {interceptor.interceptor_id}.")
                            else:
                                self.add_message(f"Interceptor {interceptor.interceptor_id} missed enemy missile {missile.label}.")
                            interceptor.complete()
                            return True
        return False
    
    def fire_interceptor_for_big(self):
        # Engage big drones.
        big_drones = [t for t in self.threats if (not t.destroyed) and t.drone_type == "big"]
        if not big_drones:
            return False
        for threat in big_drones:
            for interceptor in self.interceptors:
                if interceptor.current_state == interceptor.idle and interceptor.ammo > 0:
                    current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
                    dist = haversine(interceptor.pos['latitude'], interceptor.pos['longitude'], current_lat, current_lon)
                    if dist <= ENGAGEMENT_RANGE:
                        print(f"Interceptor {interceptor.interceptor_id} firing at big drone {threat.label} at {dist:.2f} km")
                        interceptor.safe_prepare()
                        if interceptor.current_state == interceptor.preparing:
                            interceptor.ready_up()
                            if interceptor.safe_fire(threat):
                                hit = random.random() < 0.8  # 80% hit chance for big drone.
                                if hit:
                                    threat.destroyed = True
                                    self.add_message(f"Big drone {threat.label} destroyed by interceptor {interceptor.interceptor_id}.")
                                else:
                                    self.add_message(f"Interceptor {interceptor.interceptor_id} missed big drone {threat.label}.")
                                interceptor.complete()
                                return True
        return False
    
    def fire_interceptor_for_small(self):
        # Engage small drones only if more than one interceptor (or round) is available.
        small_drones = [t for t in self.threats if (not t.destroyed) and t.drone_type == "small"]
        if not small_drones:
            return False
        # Reserve one round for missile by only firing at small drones if interceptor ammo > 1.
        available = [i for i in self.interceptors if i.current_state == i.idle and i.ammo > 0]
        if len(available) < 2:
            return False
        for threat in small_drones:
            for interceptor in available:
                current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
                dist = haversine(interceptor.pos['latitude'], interceptor.pos['longitude'], current_lat, current_lon)
                if dist <= ENGAGEMENT_RANGE:
                    print(f"Interceptor {interceptor.interceptor_id} firing at small drone {threat.label} at {dist:.2f} km")
                    interceptor.safe_prepare()
                    if interceptor.current_state == interceptor.preparing:
                        interceptor.ready_up()
                        if interceptor.safe_fire(threat):
                            hit = random.random() < 0.4  # 40% hit chance for small drone.
                            if hit:
                                threat.destroyed = True
                                self.add_message(f"Small drone {threat.label} destroyed by interceptor {interceptor.interceptor_id}.")
                            else:
                                self.add_message(f"Interceptor {interceptor.interceptor_id} missed small drone {threat.label}.")
                            interceptor.complete()
                            return True
        return False
    
    def process_action(self, action):
        # Automated decision: prioritize enemy missile, then big drones, then small drones.
        if action == 'decide':
            missile_engaged = self.fire_interceptor_for_missile()
            if not missile_engaged:
                # Fire at big drones first.
                if not self.fire_interceptor_for_big():
                    # Then fire at small drones if interceptor ammo > 1 (reserve one for missile).
                    self.fire_interceptor_for_small()
        elif action == 'reload':
            for interceptor in self.interceptors:
                if interceptor.current_state == interceptor.idle:
                    interceptor.reload()
                    interceptor.reload_time = 2
    
    def run_step(self, action=None):
        self.timestep += 1
        
        # Update enemy threats.
        for threat in self.threats:
            if not threat.destroyed:
                if threat.drone_type == "missile":
                    # Missile decays slowly (simulate 1 minute to impact).
                    threat.distance = max(0, threat.distance - random.uniform(0.05, 0.1))
                elif threat.drone_type == "big":
                    threat.distance = max(0, threat.distance - random.uniform(0.2, 0.5))
                else:
                    threat.distance = max(0, threat.distance - random.uniform(0.1, 0.3))
        
        # Check if any threat has reached the base (distance nearly 0).
        for threat in self.threats:
            if not threat.destroyed and threat.distance <= 0.01:
                if threat.drone_type == "big":
                    damage = 5
                elif threat.drone_type == "small":
                    damage = 1
                elif threat.drone_type == "missile":
                    damage = 10  # Missile is catastrophic.
                self.base_hp -= damage
                threat.destroyed = True
                self.add_message(f"{threat.label} hit the base causing {damage} damage!")
                print(f"{threat.label} hit the base, causing {damage} damage. Base HP: {self.base_hp}")
        
        # Update interceptors (reload if needed).
        for interceptor in self.interceptors:
            if interceptor.current_state == interceptor.reloading:
                interceptor.reload_time -= 1
                if interceptor.reload_time <= 0:
                    interceptor.reset()
                    interceptor.ammo = random.randint(2, 4)  # Reload with a new random ammo count.
            if interceptor.ammo == 0 and interceptor.current_state == interceptor.idle:
                interceptor.reload()
                interceptor.reload_time = 2
        
        if action is not None:
            self.process_action(action)
        
        self.log_engagement()
    
    def log_engagement(self):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "base_hp": self.base_hp,
            "threats": [t.__dict__ for t in self.threats],
            "interceptor_states": [i.current_state.value for i in self.interceptors]
        }
        self.engagement_log.append(log_entry)
    
    def check_simulation_over(self):
        enemy_remaining = [t for t in self.threats if not t.destroyed]
        if self.base_hp <= 0:
            print("\nDefeat! The base has been destroyed.")
            return True
        if len(enemy_remaining) == 0:
            print("\nVictory! All enemy threats have been neutralized.")
            return True
        return False

# ---------------------- Pygame Drawing & Main Loop ----------------------
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AFSIM Simulator – Decision Scenario (Missile & Drones)")
clock = pygame.time.Clock()

center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
scale = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# Create a specific scenario dictionary.
# Generate random ammo count for interceptor between 2 and 4.
ammo_value = random.randint(2, 4)
# Set big drone count to either ammo_value or ammo_value - 1.
big_drone_count = ammo_value if random.random() < 0.5 else max(ammo_value - 1, 1)
scenario = {
    "observation": {
        "entities": [
            {"id": 1, "type": "C2", "label": "Base", "location": {"latitude": 34.9295, "longitude": -117.8849}, "side": "blue"},
            {"id": 2, "type": "INTERCEPTOR", "label": "Interceptor", "location": {"latitude": 34.93, "longitude": -117.884}, "side": "blue", "ammo": ammo_value},
            # Enemy threats:
            # Big drones.
            *[{"id": 100 + i, "type": "UAS", "label": f"Big Drone {i+1}", "drone_type": "big",
               "location": {"latitude": 34.94, "longitude": -117.8849 + 0.021*i},
               "side": "red"} for i in range(big_drone_count)],
            # Six small drones.
            *[{"id": 200 + i, "type": "UAS", "label": f"Small Drone {i+1}", "drone_type": "small",
               "location": {"latitude": 34.928 - 0.001*i, "longitude": -117.8855 - 0.001*i},
               "side": "red"} for i in range(6)],
            # One enemy missile, placed farther away.
            {"id": 300, "type": "MISSILE", "label": "Enemy Missile", "drone_type": "missile",
             "location": {"latitude": 34.95, "longitude": -117.8849},
             "side": "red"}
        ]
    }
}

sim = AFSIMSimulator(scenario)
# Automated decision each timestep.
current_action = "decide"

SIM_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SIM_EVENT, 1000)

font = pygame.font.SysFont(None, 20)
interceptor_font = pygame.font.SysFont(None, 24)
legend_font = pygame.font.SysFont(None, 18)
message_font = pygame.font.SysFont(None, 22)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == SIM_EVENT:
            sim.run_step(action=current_action)
            current_action = "decide"
            if sim.check_simulation_over():
                running = False
    
    screen.fill(WHITE)
    pygame.draw.circle(screen, BLACK, center, 8)
    
    # Draw interceptor.
    for unit in sim.interceptors:
        pos = unit.pos
        unit_screen = geo_to_screen(pos['latitude'], pos['longitude'], sim.base_lat, sim.base_lon, center, scale)
        pygame.draw.circle(screen, GREEN, unit_screen, 8)
        pygame.draw.circle(screen, GRAY, unit_screen, int(scale * ENGAGEMENT_RANGE), 1)
    
    # Draw enemy threats if visible.
    for threat in sim.threats:
        if not sim.is_threat_visible(threat):
            continue
        threat_screen = threat.get_screen_pos(center, scale)
        # Big drones in RED, small drones in ORANGE, missile in RED.
        if threat.drone_type == "big":
            color = RED
        elif threat.drone_type == "small":
            color = ORANGE
        elif threat.drone_type == "missile":
            color = RED
        else:
            color = RED
        pygame.draw.circle(screen, color, threat_screen, 6)
        text = font.render(f"{threat.label} {threat.distance:.1f}km", True, BLACK)
        screen.blit(text, (threat_screen[0] + 10, threat_screen[1]))
    
    legend_lines = [
        "Legend:",
        "Base: Black circle",
        "Interceptor: Green circle (Gray = 2km range)",
        f"Big Drone: Red circle (5 damage)  [Count: {big_drone_count}]",
        "Small Drone: Orange circle (1 damage) [Count: 6]",
        "Enemy Missile: Red circle (destroys base)",
        "Decision: Reserve interceptor for missile if imminent",
        "Automated Decision each timestep"
    ]
    for i, line in enumerate(legend_lines):
        legend_text = legend_font.render(line, True, BLACK)
        screen.blit(legend_text, (10, 10 + i * 20))
    
    status_lines = []
    for i, unit in enumerate(sim.interceptors):
        status_lines.append(f"Interceptor {unit.interceptor_id}: Ammo {unit.ammo}")
    for i, line in enumerate(status_lines):
        status_text = interceptor_font.render(line, True, BLACK)
        screen.blit(status_text, (WINDOW_WIDTH - 220, 10 + i * 30))
    
    hp_text = interceptor_font.render(f"Base HP: {sim.base_hp}", True, BLACK)
    screen.blit(hp_text, (WINDOW_WIDTH - 220, 150))
    
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
