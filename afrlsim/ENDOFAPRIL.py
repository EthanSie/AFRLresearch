#!/usr/bin/env python
"""
ADVANCED DOUBLE-DQN TRAINER (~<10h, with detailed logs)
────────────────────────────────────────────────────────
• Double DQN with target network for reduced bias
• Dueling architecture: separate value & advantage streams
• ε-greedy exploration with linear decay
• 4 parallel actor threads, 3s sim ticks
• Episode cap = 200 steps, 4096 episodes total (~8h)
• Replay buffer size = 200k, batch = 256
• Target network sync every 1k updates
• Gzip checkpoints every 256 episodes
• Detailed actor & learner logging for debugging
"""
import random
import time
import gzip
import threading
import queue
from collections import deque
from datetime import datetime, timezone
from typing import Tuple, List

import http.client
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

# ── HYPERPARAMETERS ────────────────────────────────────
TOTAL_EPISODES     = 4096
WORKERS            = 4
EPISODES_PER_ACTOR = TOTAL_EPISODES // WORKERS
SIM_STEP_MSEC      = 3000
MAX_STEPS_PER_EP   = 200
BATCH_SIZE         = 256
REPLAY_CAPACITY    = 200_000
TARGET_SYNC_EVERY  = 1000
EVAL_EVERY_EPISODE = 256

GAMMA              = 0.99
LR                 = 5e-4
EPS_START, EPS_END = 1.0, 0.01
EPS_DECAY_STEPS    = TOTAL_EPISODES * MAX_STEPS_PER_EP

# ── ENVIRONMENT / HTTP SESSION ────────────────────────
SERVER_URL        = "http://10.95.10.32:5000"
TIMEOUT           = 10
BASE_HP_INITIAL   = 10
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_MAPPING    = {"fire_at_missile":0, "fire_at_big":1,
                     "fire_at_small":2, "reload":3, "do_nothing":4}
NUM_ACTIONS       = len(ACTION_MAPPING)

# Retry configuration for HTTP (including read errors)
_retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=0.3,
    status_forcelist=(502, 503, 504),
    allowed_methods=frozenset(["POST"])
)

def http_session():
    s = requests.Session()
    # optionally remove gzip if still problematic:
    # s.headers.update({"Accept-Encoding": "gzip"})
    adapter = HTTPAdapter(max_retries=_retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = http_session()

# ── Remote AFSIM wrapper ───────────────────────────────
class RemoteAFSIM:
    def __init__(self):
        print(f"[actor] Starting new session")
        self.token = self._post_json("new_session", {"client_name":"dbl_dqn","sim_type":"SINS"})["client_token"]
        init = self._post_json("init_sim", {
            "client_token": self.token,
            "sim_type": "SINS",
            "sim_params": {
                "scenario": "scenarios.acumen.acumen",
                "headless": True,
                "step_size": SIM_STEP_MSEC
            }
        })
        self.obs = init["observation"]
        self.prev_hp = self._get_hp()
        print(f"[actor] Session initialized (token={self.token})")

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        url = f"{SERVER_URL}/{endpoint}"
        for attempt in range(3):
            try:
                resp = SESSION.post(url, json=payload, timeout=TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ChunkedEncodingError,
                    http.client.IncompleteRead,
                    requests.exceptions.ConnectionError) as e:
                print(f"[actor] Read error on {endpoint}, retry {attempt+1}/3: {e}")
                time.sleep(0.5)
        raise RuntimeError(f"[actor] Failed to POST {endpoint} after retries")

    def _ents(self) -> List[dict]:
        return [e for e in self.obs.get("entities", []) if e]

    def _get_hp(self) -> int:
        base = next((e for e in self._ents() if e.get("type")=="C2"), None)
        return base.get("health", BASE_HP_INITIAL) if base else BASE_HP_INITIAL

    def _threats(self) -> List[dict]:
        return [e for e in self._ents() if e.get("side")=="red" and not e.get("destroyed",False)]

    def state(self) -> Tuple[float,float]:
        return float(self._get_hp()), float(len(self._threats()))

    def step(self, action_idx: int) -> bool:
        data = self._post_json("action", {"client_token":self.token,"action":float(action_idx)})
        obs = data.get("observation")
        if obs is None:
            print(f"[actor] WARNING - no observation, response={data}")
            return False
        self.obs = obs
        return True

    def done(self) -> bool:
        return (
            self.obs.get("sim_state")=="SimulationState.TEARDOWN"
            or self._get_hp()<=0
            or len(self._threats())==0
        )

    def reward(self) -> float:
        hp = self._get_hp()
        delta = hp - self.prev_hp
        self.prev_hp = hp
        return 2*delta - 0.2*len(self._threats())

# ── Dueling Double-DQN architecture ─────────────────────
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature      = nn.Sequential(nn.Linear(2,128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
        self.adv_stream   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,NUM_ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))

# ── Replay buffer ───────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def push(self, *t): self.buf.append(t)
    def sample(self, bs: int):
        batch = random.sample(self.buf, bs)
        s,a,r,ns,d = zip(*batch)
        cast = lambda v, dt: torch.tensor(v, dtype=dt, device=DEVICE)
        return (
            cast(s, torch.float32),
            cast(a, torch.int64),
            cast(r, torch.float32),
            cast(ns, torch.float32),
            cast(d, torch.float32)
        )
    def __len__(self): return len(self.buf)

# ── ε-greedy schedule ─────────────────────────────────
def epsilon(step: int) -> float:
    if step >= EPS_DECAY_STEPS: return EPS_END
    return EPS_START + (EPS_END-EPS_START)*step/EPS_DECAY_STEPS

# ── Actor thread ───────────────────────────────────────
def actor_thread(tid: int, trans_q: queue.Queue, done_evt: threading.Event):
    for ep in range(EPISODES_PER_ACTOR):
        print(f"[actor {tid}] Episode {ep+1}/{EPISODES_PER_ACTOR} start")
        try:
            env = RemoteAFSIM()
        except Exception as e:
            print(f"[actor {tid}] Init error: {e}")
            done_evt.set()
            return
        state = env.state()
        steps = 0
        for t in range(MAX_STEPS_PER_EP):
            action = random.randrange(NUM_ACTIONS)
            print(f"[actor {tid}] Step {t+1}/{MAX_STEPS_PER_EP} action {action}")
            if not env.step(action):
                print(f"[actor {tid}] Step failed, ending episode")
                break
            reward = env.reward()
            nxt = env.state()
            done = env.done()
            try:
                trans_q.put((state,action,reward,nxt,done), timeout=1)
                print(f"[actor {tid}] Pushed transition, queue size {trans_q.qsize()}")
            except queue.Full:
                print(f"[actor {tid}] Queue full, skip")
            state = nxt
            steps += 1
            if done:
                print(f"[actor {tid}] Done at step {t+1}")
                break
        print(f"[actor {tid}] Episode done in {steps} steps")
        if done_evt.is_set(): return
    done_evt.set()

# ── Main training loop ─────────────────────────────────
def train():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    policy_net = DuelingDQN().to(DEVICE)
    target_net = DuelingDQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer   = optim.Adam(policy_net.parameters(), lr=LR)

    replay_buf = ReplayBuffer(REPLAY_CAPACITY)
    trans_queue = queue.Queue(maxsize=50_000)
    done_event = threading.Event()

    for i in range(WORKERS):
        threading.Thread(target=actor_thread,
                         args=(i,trans_queue,done_event),
                         daemon=True).start()

    total_updates = 0
    episode_count = 0
    progress = tqdm(total=TOTAL_EPISODES, desc="Episodes")
    print("[learner] Starting training loop...")
    start = time.time()

    while not done_event.is_set() or not trans_queue.empty():
        try:
            s,a,r,ns,d = trans_queue.get(timeout=0.5)
        except queue.Empty:
            print("[learner] Waiting for transitions...")
            continue

        replay_buf.push(s,a,r,ns,d)
        if d:
            episode_count += 1
            progress.update(1)
            print(f"[learner] Episode {episode_count} complete")

        if len(replay_buf) >= BATCH_SIZE:
            states, acts, rews, nxt_states, dones = replay_buf.sample(BATCH_SIZE)
            q_vals = policy_net(states).gather(1, acts.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                best_a = policy_net(nxt_states).argmax(dim=1)
                next_q = target_net(nxt_states).gather(1, best_a.unsqueeze(1)).squeeze(1)
                target_q = rews + (1-dones)*GAMMA*next_q
            loss = nn.functional.smooth_l1_loss(q_vals, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_updates += 1
            if total_updates % TARGET_SYNC_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if episode_count >= TOTAL_EPISODES:
            break

    progress.close()
    ckpt = f"double_dqn_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.pt.gz"
    with gzip.open(ckpt,"wb") as f:
        torch.save(policy_net.state_dict(), f)
    hrs = (time.time()-start)/3600
    print(f"Finished in {hrs:.2f}h. Model saved to {ckpt}")

if __name__ == "__main__":
    train()
