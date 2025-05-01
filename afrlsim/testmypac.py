#!/usr/bin/env python
"""
Local DQN throughput benchmark (no nvml, no network).

•  Mocks the RemoteAFSIM environment with random state / reward.
•  Uses the same DQN, replay buffer, mini-batch size, and target-sync cadence.
•  Prints steps-per-second and estimated wall-clock for a 500-episode run
   *if* the only bottleneck were the GPU / Python side.

Expect ≈8-12 k steps/sec on a RTX 4060 Laptop @ 6-7 TFLOPS.
"""
import random, time, torch, torch.nn as nn, torch.optim as optim
from collections import deque

# ---------------- hyper-params (match trainer) -----------------
GAMMA, LR = 0.9, 1e-3
BATCH_SIZE, REPLAY_CAP = 256, 50_000
TARGET_SYNC_EVERY = 1_000
TOTAL_EPISODES, STEPS_PER_EP = 2000, 200  # run long for stable avg
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
print("PyTorch CUDA available? ", torch.cuda.is_available())
print("CUDA devices count:   ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU!")
# ---------------- tiny DQN (same shape) ------------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 5))
    def forward(self,x): return self.net(x)

# ---------------- dummy env ------------------------------------
class DummyEnv:
    """Returns (base_hp, threats) and reward each step."""
    def __init__(self): self.reset()
    def reset(self):
        self.base_hp = 10
        self.threats = random.randint(3,10)
        return (self.base_hp, self.threats)
    def step(self,act):
        # random dynamics
        self.threats = max(0, self.threats - random.randint(0,2))
        self.base_hp = max(0, self.base_hp - random.randint(0,1))
        done = self.threats==0 or self.base_hp==0
        reward = 2*random.randint(-1,1) - .2*self.threats
        return (self.base_hp, self.threats), reward, done

# ---------------- replay ---------------------------------------
class Replay:
    def __init__(self,cap): self.buf=deque(maxlen=cap)
    def push(self,*t): self.buf.append(t)
    def sample(self,n):
        s,a,r,ns,d=zip(*random.sample(self.buf,n))
        to = lambda x, dt: torch.tensor(x, dtype=dt, device=DEVICE)
        return (to(s,torch.float32),to(a,torch.int64),to(r,torch.float32),
                to(ns,torch.float32),to(d,torch.float32))
    def __len__(self): return len(self.buf)

# ---------------- benchmark loop --------------------------------
def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    q, tgt = DQN().to(DEVICE), DQN().to(DEVICE)
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=LR)
    rb  = Replay(REPLAY_CAP)

    env = DummyEnv()
    global_step = 0
    t_start = time.perf_counter()

    for ep in range(TOTAL_EPISODES):
        s = env.reset()
        for _ in range(STEPS_PER_EP):
            a = random.randrange(5) if random.random()<0.1 else \
                q(torch.tensor(s,dtype=torch.float32,device=DEVICE)).argmax().item()
            ns, r, done = env.step(a)
            rb.push(s,a,r,ns,float(done)); s=ns; global_step+=1

            if len(rb)>=BATCH_SIZE:
                states,acts,rews,nstates,dones = rb.sample(BATCH_SIZE)
                qv = q(states).gather(1,acts.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next = tgt(nstates).max(1)[0]
                    target_q = rews + (1-dones)*GAMMA*max_next
                loss = nn.functional.smooth_l1_loss(qv,target_q)
                opt.zero_grad(); loss.backward(); opt.step()
                if global_step%TARGET_SYNC_EVERY==0:
                    tgt.load_state_dict(q.state_dict())
            if done: break

    torch.cuda.synchronize() if DEVICE.type=="cuda" else None
    elapsed = time.perf_counter()-t_start
    sps = global_step/elapsed
    print(f"Total env steps: {global_step:,}")
    print(f"Elapsed: {elapsed:.1f}s  ->  {sps:,.0f} steps / sec")

    # convert to ETA for real training (assume 200 steps / ep)
    est_wall = (500*200)/sps
    print(f"\nIf GPU were the only bottleneck, 500 eps ≈ {est_wall/60:.1f} min.")

if __name__=="__main__":
    main()
