import json, math, random, pygame, os, copy
from datetime import datetime
from statemachine import StateMachine, State

# ---------------- Constants ----------------
ENGAGEMENT_RANGE = 2.0    # km for detection/engagement
BASE_HP_INITIAL = 10      # Base hit points
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600

# ---------------- Helper Functions ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dLon/2)**2)
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

# ---------------- Q–Table Overlay Helpers ----------------
def softmax(q_values):
    exp_values = {action: math.exp(q) for action, q in q_values.items()}
    sum_exp = sum(exp_values.values())
    return {action: exp_val / sum_exp for action, exp_val in exp_values.items()}

def draw_q_table_overlay(screen, q_table, state_key, position, font):
    """Draw the Q–values and their softmax probabilities for the current state."""
    if state_key not in q_table:
        text = font.render("No Q–values for state", True, (0, 0, 0))
        screen.blit(text, position)
        return
    q_values = q_table[state_key]
    probabilities = softmax(q_values)
    x, y = position
    header = font.render("Action   Q-value   Prob", True, (0, 0, 0))
    screen.blit(header, (x, y))
    y += 25
    for action, q_val in q_values.items():
        prob = probabilities.get(action, 0)
        line = f"{action:<15}{q_val:>8.2f}{prob:>8.2f}"
        text = font.render(line, True, (0, 0, 0))
        screen.blit(text, (x, y))
        y += 25

def draw_success_probabilities_overlay(screen, success_probs, position, font):
    """Overlay a table of estimated success probabilities for each action."""
    x, y = position
    header = font.render("Action         Success Prob", True, (0, 0, 0))
    screen.blit(header, (x, y))
    y += 25
    for action, prob in success_probs.items():
        line = f"{action:<15}{prob:>6.2f}"
        text = font.render(line, True, (0, 0, 0))
        screen.blit(text, (x, y))
        y += 25

# ---------------- Simulation State Machines ----------------
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

# ---------------- Core Simulation Classes ----------------
class Threat:
    def __init__(self, entity_data, base_lat, base_lon):
        self.id = entity_data['id']
        self.label = entity_data['label']
        self.lat = entity_data['location']['latitude']
        self.lon = entity_data['location']['longitude']
        self.drone_type = entity_data.get('drone_type', None)  # "big", "small", "missile"
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

class AFSIMSimulator:
    def __init__(self, scenario, verbose=True):
        self.entities = scenario["observation"]["entities"]
        self.base = next(e for e in self.entities if e['type'] == 'C2')
        self.base_lat = self.base['location']['latitude']
        self.base_lon = self.base['location']['longitude']
        self.base_hp = BASE_HP_INITIAL
        self.verbose = verbose

        # Create interceptor asset.
        interceptor_entity = next(e for e in self.entities if e['type'] == 'INTERCEPTOR')
        if 'ammo' in interceptor_entity:
            ammo = interceptor_entity['ammo']
        else:
            ammo = random.randint(2, 4)
            interceptor_entity['ammo'] = ammo
        self.interceptors = [InterceptorStateMachine(interceptor_entity, ammo=ammo)]

        # Enemy threats.
        self.threats = [Threat(e, self.base_lat, self.base_lon) for e in self.entities if e['side'] == 'red']

        self.timestep = 0
        self.engagement_log = []
        self.messages = []
    
    def add_message(self, msg, duration=3000):
        expire = pygame.time.get_ticks() + duration
        self.messages.append({"text": msg, "expire": expire})
    
    def is_threat_visible(self, threat):
        current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
        for unit in self.interceptors:
            d = haversine(unit.pos['latitude'], unit.pos['longitude'], current_lat, current_lon)
            if d <= ENGAGEMENT_RANGE:
                return True
        return False
    
    # New process_action using the expanded action set.
    def process_action(self, action):
        if action == "fire_at_missile":
            self.fire_interceptor_for_missile()
        elif action == "fire_at_big":
            self.fire_interceptor_for_big()
        elif action == "fire_at_small":
            self.fire_interceptor_for_small()
        elif action == "reload":
            for interceptor in self.interceptors:
                if interceptor.current_state == interceptor.idle:
                    interceptor.reload()
                    interceptor.reload_time = 2
        elif action == "do_nothing":
            pass  # No action taken.
    
    def fire_interceptor_for_missile(self):
        enemy_missiles = [t for t in self.threats if (not t.destroyed) and t.drone_type == "missile"]
        if not enemy_missiles:
            return False
        missile = enemy_missiles[0]
        for interceptor in self.interceptors:
            if interceptor.current_state == interceptor.idle and interceptor.ammo > 0:
                current_lat, current_lon = missile.get_current_position(self.base_lat, self.base_lon)
                dist = haversine(interceptor.pos['latitude'], interceptor.pos['longitude'], current_lat, current_lon)
                if dist <= ENGAGEMENT_RANGE:
                    interceptor.safe_prepare()
                    if interceptor.current_state == interceptor.preparing:
                        interceptor.ready_up()
                        if interceptor.safe_fire(missile):
                            hit = random.random() < 0.9
                            if hit:
                                missile.destroyed = True
                                self.add_message(f"Enemy missile {missile.label} destroyed!")
                            else:
                                self.add_message(f"Interceptor missed enemy missile {missile.label}.")
                            interceptor.complete()
                            return True
        return False
    
    def fire_interceptor_for_big(self):
        big_drones = [t for t in self.threats if (not t.destroyed) and t.drone_type == "big"]
        if not big_drones:
            return False
        for threat in big_drones:
            for interceptor in self.interceptors:
                if interceptor.current_state == interceptor.idle and interceptor.ammo > 0:
                    current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
                    dist = haversine(interceptor.pos['latitude'], interceptor.pos['longitude'], current_lat, current_lon)
                    if dist <= ENGAGEMENT_RANGE:
                        interceptor.safe_prepare()
                        if interceptor.current_state == interceptor.preparing:
                            interceptor.ready_up()
                            if interceptor.safe_fire(threat):
                                hit = random.random() < 0.8
                                if hit:
                                    threat.destroyed = True
                                    self.add_message(f"Big drone {threat.label} destroyed!")
                                else:
                                    self.add_message(f"Interceptor missed big drone {threat.label}.")
                                interceptor.complete()
                                return True
        return False
    
    def fire_interceptor_for_small(self):
        small_drones = [t for t in self.threats if (not t.destroyed) and t.drone_type == "small"]
        if not small_drones:
            return False
        available = [i for i in self.interceptors if i.current_state == i.idle and i.ammo > 0]
        if len(available) < 1:
            return False
        for threat in small_drones:
            for interceptor in available:
                current_lat, current_lon = threat.get_current_position(self.base_lat, self.base_lon)
                dist = haversine(interceptor.pos['latitude'], interceptor.pos['longitude'], current_lat, current_lon)
                if dist <= ENGAGEMENT_RANGE:
                    interceptor.safe_prepare()
                    if interceptor.current_state == interceptor.preparing:
                        interceptor.ready_up()
                        if interceptor.safe_fire(threat):
                            hit = random.random() < 0.4
                            if hit:
                                threat.destroyed = True
                                self.add_message(f"Small drone {threat.label} destroyed!")
                            else:
                                self.add_message(f"Interceptor missed small drone {threat.label}.")
                            interceptor.complete()
                            return True
        return False
    
    def run_step(self, action="do_nothing"):
        self.timestep += 1

        # Update threat distances.
        for threat in self.threats:
            if not threat.destroyed:
                if threat.drone_type == "missile":
                    threat.distance = max(0, threat.distance - random.uniform(0.05, 0.1))
                elif threat.drone_type == "big":
                    threat.distance = max(0, threat.distance - random.uniform(0.2, 0.5))
                else:
                    threat.distance = max(0, threat.distance - random.uniform(0.1, 0.3))

        # Check if any threat hits the base.
        for threat in self.threats:
            if not threat.destroyed and threat.distance <= 0.01:
                if threat.drone_type == "big":
                    damage = 5
                elif threat.drone_type == "small":
                    damage = 1
                elif threat.drone_type == "missile":
                    damage = 10
                self.base_hp -= damage
                threat.destroyed = True
                self.add_message(f"{threat.label} hit the base! Damage: {damage}")

        # Manage interceptor reloads.
        for interceptor in self.interceptors:
            if interceptor.current_state == interceptor.reloading:
                interceptor.reload_time -= 1
                if interceptor.reload_time <= 0:
                    interceptor.reset()
                    interceptor.ammo = random.randint(2, 4)
            if interceptor.ammo == 0 and interceptor.current_state == interceptor.idle:
                interceptor.reload()
                interceptor.reload_time = 2

        self.process_action(action)
        self.log_engagement()
    
    def log_engagement(self):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "base_hp": self.base_hp,
            "threats": [t.__dict__ for t in self.threats],
            "interceptor_states": [i.current_state.value for i in self.interceptors],
            "timestep": self.timestep
        }
        self.engagement_log.append(log_entry)
        if self.verbose:
            print(f"Time Step {self.timestep} | Base HP: {self.base_hp}")

    def check_simulation_over(self):
        enemy_remaining = [t for t in self.threats if not t.destroyed]
        if self.base_hp <= 0:
            if self.verbose:
                print("Defeat! The base has been destroyed.")
            return True
        if len(enemy_remaining) == 0:
            if self.verbose:
                print("Victory! All enemy threats have been neutralized.")
            return True
        return False


# ---------------- Monte Carlo Rollout Policy ----------------
def rollout_policy(sim, q_table, actions, epsilon=0.1):
    state = (sim.base_hp, sum(1 for t in sim.threats if not t.destroyed))
    state_key = str(state)
    if state_key in q_table and random.random() > epsilon:
        return max(q_table[state_key], key=q_table[state_key].get)
    else:
        return random.choice(actions)

def estimate_success_probability(sim, candidate_action, trials=10, q_table=None, actions=None):
    wins = 0
    for _ in range(trials):
        sim_copy = copy.deepcopy(sim)
        sim_copy.verbose = False  # Disable logging for the rollout copy.
        sim_copy.run_step(action=candidate_action)
        while not sim_copy.check_simulation_over():
            if q_table and actions:
                action = rollout_policy(sim_copy, q_table, actions)
            else:
                action = "do_nothing"
            sim_copy.run_step(action=action)
        enemy_remaining = [t for t in sim_copy.threats if not t.destroyed]
        if sim_copy.base_hp > 0 and len(enemy_remaining) == 0:
            wins += 1
    return wins / trials

# ---------------- Q–Learning Agent Training ----------------
def train_mode():
    total_episodes = 50000
    alpha = 0.1        # Learning rate
    gamma = 0.9        # Discount factor
    epsilon = 1.0      # Initial exploration rate
    epsilon_min = 0.1
    epsilon_decay = 0.995
    actions = ["fire_at_missile", "fire_at_big", "fire_at_small", "do_nothing", "reload"]
    q_table = {}
    rewards_all = []

    for episode in range(total_episodes):
        ammo_value = random.randint(2, 4)
        big_drone_count = ammo_value if random.random() < 0.5 else max(ammo_value - 1, 1)
        scenario = {
            "observation": {
                "entities": [
                    {"id": 1, "type": "C2", "label": "Base",
                     "location": {"latitude": 34.9295, "longitude": -117.8849}, "side": "blue"},
                    {"id": 2, "type": "INTERCEPTOR", "label": "Interceptor",
                     "location": {"latitude": 34.93, "longitude": -117.884}, "side": "blue", "ammo": ammo_value},
                    *[{"id": 100 + i, "type": "UAS", "label": f"Big Drone {i+1}", "drone_type": "big",
                        "location": {"latitude": 34.94, "longitude": -117.8849 + 0.021 * i},
                        "side": "red"} for i in range(big_drone_count)],
                    *[{"id": 200 + i, "type": "UAS", "label": f"Small Drone {i+1}", "drone_type": "small",
                        "location": {"latitude": 34.928 - 0.001 * i, "longitude": -117.8855 - 0.001 * i},
                        "side": "red"} for i in range(6)],
                    {"id": 300, "type": "MISSILE", "label": "Enemy Missile", "drone_type": "missile",
                     "location": {"latitude": 34.95, "longitude": -117.8849}, "side": "red"}
                ]
            }
        }
        sim = AFSIMSimulator(scenario)
        transitions = []  # To store transitions for the episode.
        total_reward_episode = 0

        # Run simulation until it ends.
        while not sim.check_simulation_over():
            state = (sim.base_hp, sum(1 for t in sim.threats if not t.destroyed))
            state_key = str(state)
            if state_key not in q_table:
                q_table[state_key] = {a: 0.0 for a in actions}

            # Epsilon-greedy action selection.
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(q_table[state_key], key=q_table[state_key].get)

            sim.run_step(action)

            next_state = (sim.base_hp, sum(1 for t in sim.threats if not t.destroyed))
            next_state_key = str(next_state)
            if next_state_key not in q_table:
                q_table[next_state_key] = {a: 0.0 for a in actions}

            # For intermediate steps, apply a small time penalty.
            reward = -1  
            transitions.append((state_key, action, next_state_key, reward))
            total_reward_episode += reward

        # Episode finished—compute final reward.
        if sim.base_hp <= 0:
            final_reward = -20  # Base destroyed.
        else:
            big_destroyed = sum(1 for t in sim.threats if t.drone_type == "big" and t.destroyed)
            final_reward = 2 * sim.base_hp + 3 * big_destroyed + 10  # Victory reward.
        total_reward_episode += final_reward

        # Update Q–values for all transitions using backward update.
        G = final_reward
        for state_key, action, next_state_key, reward in reversed(transitions):
            G = reward + gamma * G
            old_value = q_table[state_key][action]
            q_table[state_key][action] = old_value + alpha * (G - old_value)

        # Update Q–value for the last transition using the final reward.
        if transitions:
            last_state_key, last_action, last_next_state_key, _ = transitions[-1]
            old_value = q_table[last_state_key][last_action]
            # For terminal state, next_max is 0.
            new_value = old_value + alpha * (final_reward - old_value)
            q_table[last_state_key][last_action] = new_value

        rewards_all.append(total_reward_episode)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{total_episodes} Total Reward: {total_reward_episode}")

    with open("q_table.json", "w") as f:
        json.dump(q_table, f)
    with open("training_rewards.json", "w") as f:
        json.dump(rewards_all, f)

# ---------------- Demo Mode with Paused Actions and Overlays ----------------
def demo_mode():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("AFSIM Demo – Step-by-Step")
    clock = pygame.time.Clock()
    center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    scale = 50
    font = pygame.font.SysFont(None, 20)

    if os.path.exists("q_table.json"):
        with open("q_table.json", "r") as f:
            q_table = json.load(f)
    else:
        q_table = {}

    actions = ["fire_at_missile", "fire_at_big", "fire_at_small", "do_nothing", "reload"]

    ammo_value = random.randint(3, 4)
    big_drone_count = ammo_value if random.random() < 0.2 else max(ammo_value - 2, 2)
    scenario = {
    "observation": {
        "entities": [
            {"id": 1, "type": "C2", "label": "Base",
             "location": {"latitude": 34.9295, "longitude": -117.8849}, "side": "blue"},
            {"id": 2, "type": "INTERCEPTOR", "label": "Interceptor",
             "location": {"latitude": 34.93, "longitude": -117.884}, "side": "blue", "ammo": ammo_value},
            *[{"id": 100 + i, "type": "UAS", "label": f"Big Drone {i+1}", "drone_type": "big",
                "location": {"latitude": 34.94, "longitude": -117.8849 + 0.021 * i},
                "side": "red"} for i in range(big_drone_count)],
            *[{"id": 200 + i, "type": "UAS", "label": f"Small Drone {i+1}", "drone_type": "small",
                "location": {"latitude": 34.928 - 0.001 * i, "longitude": -117.8855 - 0.001 * i},
                "side": "red"} for i in range(6)],
            {"id": 300, "type": "MISSILE", "label": "Enemy Missile", "drone_type": "missile",
             "location": {"latitude": 34.95, "longitude": -117.8849}, "side": "red"}
        ]
    }
}
    sim = AFSIMSimulator(scenario)

    running = True
    instructions = font.render("Press SPACE to advance one time step. Press ESC to quit.", True, (0, 0, 0))
    success_probs = {}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Use the Q–table to select the best action based on the current state.
                    current_state = (sim.base_hp, sum(1 for t in sim.threats if not t.destroyed))
                    state_key = str(current_state)
                    if state_key in q_table:
                        best_action = max(q_table[state_key], key=q_table[state_key].get)
                    else:
                        best_action = "do_nothing"  # Default action if no Q–values exist.
                    
                    sim.run_step(action=best_action)
                    if sim.check_simulation_over():
                        running = False
                    for act in actions:
                        success_probs[act] = estimate_success_probability(sim, act, trials=5, q_table=q_table, actions=actions)

        screen.fill((255, 255, 255))
        screen.blit(instructions, (10, 10))
        pygame.draw.circle(screen, (0, 0, 0), center, 8)

        for unit in sim.interceptors:
            pos = unit.pos
            unit_screen = geo_to_screen(pos['latitude'], pos['longitude'], sim.base_lat, sim.base_lon, center, scale)
            pygame.draw.circle(screen, (0, 255, 0), unit_screen, 8)
            pygame.draw.circle(screen, (128, 128, 128), unit_screen, int(scale * ENGAGEMENT_RANGE), 1)

        for threat in sim.threats:
            if not sim.is_threat_visible(threat):
                continue
            threat_screen = threat.get_screen_pos(center, scale)
            color = (255, 0, 0) if threat.drone_type in ["big", "missile"] else (255, 165, 0)
            pygame.draw.circle(screen, color, threat_screen, 6)
            label_text = font.render(f"{threat.label} {threat.distance:.1f}km", True, (0, 0, 0))
            screen.blit(label_text, (threat_screen[0] + 10, threat_screen[1]))

        hp_text = font.render(f"Base HP: {sim.base_hp}", True, (0, 0, 0))
        ts_text = font.render(f"Time Step: {sim.timestep}", True, (0, 0, 0))
        screen.blit(hp_text, (WINDOW_WIDTH - 150, 10))
        screen.blit(ts_text, (WINDOW_WIDTH - 150, 30))

        current_state = (sim.base_hp, sum(1 for t in sim.threats if not t.destroyed))
        state_key = str(current_state)
        draw_q_table_overlay(screen, q_table, state_key, (WINDOW_WIDTH - 250, 50), font)
        draw_success_probabilities_overlay(screen, success_probs, (WINDOW_WIDTH - 250, 200), font)

        current_ticks = pygame.time.get_ticks()
        sim.messages = [m for m in sim.messages if m["expire"] > current_ticks]
        for i, message in enumerate(sim.messages):
            msg_text = font.render(message["text"], True, (0, 0, 0))
            screen.blit(msg_text, (10, WINDOW_HEIGHT - 100 - i * 25))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

# ---------------- Main Program ----------------
if __name__ == "__main__":
    mode = input("Enter 'train' to train the Q–learning agent or 'demo' to run a demo episode: ").strip().lower()
    if mode == "train":
        train_mode()
    elif mode == "demo":
        demo_mode()
    else:
        print("Invalid mode. Please choose 'train' or 'demo'.")
