import requests
import json
import random
import time
import os
from datetime import datetime

def main():
    parent_log_dir = "random_agent_logs"
    os.makedirs(parent_log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M")
    log_dir = os.path.join(parent_log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    summary_log = open(os.path.join(log_dir, "readable_summary.txt"), 'w')
    
    url = "http://10.95.10.32:5000"
    
    new_session = {
        "client_name": "random_agent",
        "sim_type": "SINS"
    }
    
    with open(os.path.join(log_dir, "new_session.json"), 'w') as f:
        json.dump(new_session, f, indent=2)

    response = requests.post(f"{url}/new_session", json=new_session)
    with open(os.path.join(log_dir, "new_session_response.json"), 'w') as f:
        json.dump(response.json(), f, indent=2)
    client_token = response.json()["client_token"]
    
    init_sim = {
        "client_token": client_token,
        "sim_type": "SINS",
        "sim_params": {
            "scenario": "scenarios.acumen.acumen",
            "headless": False,
            "step_size": 3000
        }
    }

    with open(os.path.join(log_dir, "init_sim.json"), 'w') as f:
        json.dump(init_sim, f, indent=2)

    response = requests.post(f"{url}/init_sim", json=init_sim)
    with open(os.path.join(log_dir, "init_sim_response.json"), 'w') as f:
        json.dump(response.json(), f, indent=2)


    action_space = response.json()["observation"]["action_space"]
    
    round_counter = 0
    while True:
        # Create round separator for readable summary
        round_summary = f"\n{'='*50}\n"
        round_summary += f"Round {round_counter} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        round_summary += f"{'='*50}\n"
        
        action_space = response.json()["observation"]["action_space"]
        action_ids = list(action_space.keys())
        
        # Add available actions to summary
        round_summary += "\nAvailable actions:\n"
        for action_id in action_ids:
            round_summary += f"Action {action_id}: {action_space[action_id]}\n"

        '''
        This is where the agent logic would go. For now, we will just select a random action.
        '''
        action_str = random.choice(action_ids) 

        action = float(action_str)
        
        # Add selected actions to summary
        selected_action = action_space[action_str]
        round_summary += f"\nSelected action: {selected_action}\n"
        round_summary += f"Action ID: {selected_action["action_id"]}\n"

        action_request = {
            "client_token": client_token,
            "action": action
        }
        response = requests.post(f"{url}/action", json=action_request)        

        with open(os.path.join(log_dir, f"round_{round_counter}_action.json"), 'w') as f:
            json.dump(action_request, f, indent=2)
        with open(os.path.join(log_dir, f"round_{round_counter}_result.json"), 'w') as f:
            json.dump(response.json(), f, indent=2)
        
        ''' 
        This is all information for the readable summary. It is not required.
        '''
        entities = response.json()["observation"].get("entities", [])
        if entities:
            drones_found = False
            
            for entity in entities:   
                if not isinstance(entity, dict):
                    continue

                entity_type = entity.get("type")
                if entity_type == "DroneEntity":
                    drones_found = True
                    drone_status = f"\nDrone Status | {entity.get('label', 'Unknown')} (ID: {entity.get('id', 'Unknown')}):\n"
                    drone_status += f"  Side: {entity.get('side', 'N/A')}\n"
                    location = entity.get('location', {})
                    if isinstance(location, dict):
                        drone_status += f"  Position: lat={location.get('latitude', 'N/A'):.5f}, "
                        drone_status += f"lon={location.get('longitude', 'N/A'):.5f}, "
                        drone_status += f"alt={location.get('altitude', 'N/A'):.5f}\n"
                    drone_status += f"  Health: {entity.get('health', 'N/A')}\n"
                    drone_status += f"  Operational State: {entity.get('operational_state', 'N/A')}\n"
                    round_summary += drone_status
                
                if entity_type == "MissileEntity":
                    cm_status = f"\nCountermeasure Status | Missile {entity.get('label', 'Unknown')} (ID: {entity.get('id', 'Unknown')}):\n"
                    cm_status += f"  Ammo: {entity.get('remaining_ammo', 'N/A')}\n"
                    cm_status += f"  Status: {entity.get('state', 'N/A')}\n"
                    cm_status += f"  Operational State: {entity.get('operational_state', 'N/A')}\n"
                    round_summary += cm_status
                
                elif entity_type == "GunEntity":
                    cm_status = f"\nCountermeasure Status | Gun {entity.get('label', 'Unknown')} (ID: {entity.get('id', 'Unknown')}):\n"
                    cm_status += f"  Ammo: {entity.get('ammo', 'N/A')}\n"
                    cm_status += f"  Operational State: {entity.get('operational_state', 'N/A')}\n"
                    round_summary += cm_status
                
                elif entity_type == "JammerEntity":
                    cm_status = f"\nCountermeasure Status | Jammer {entity.get('label', 'Unknown')} (ID: {entity.get('id', 'Unknown')}):\n"
                    cm_status += f"  Battery: {entity.get('battery', 'N/A')}\n"
                    cm_status += f"  Buzzer Active: {entity.get('buzzer_active', 'N/A')}\n"
                    cm_status += f"  Operational State: {entity.get('operational_state', 'N/A')}\n"
                    round_summary += cm_status
                
                elif entity_type == "AlarmEntity":
                    cm_status = f"\nCountermeasure Status | Alarm {entity.get('label', 'Unknown')} (ID: {entity.get('id', 'Unknown')}):\n"
                    cm_status += f"  Active: {entity.get('active', 'N/A')}\n"
                    cm_status += f"  Operational State: {entity.get('operational_state', 'N/A')}\n"
                    round_summary += cm_status
        
        summary_log.write(round_summary)
        summary_log.flush()

        round_counter += 1


        '''
        At the end of the sim, we can pull metrics (need to figure out what metrics to pull)
        '''
        if response.json()["observation"]["sim_state"] == "SimulationState.TEARDOWN":
            summary_log.close()
            with open(os.path.join(log_dir, "metrics.json"), 'w') as f:
                json.dump({
                    "total_rounds": round_counter,
                    "final_state": response.json()["observation"]["sim_state"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            break
        
        
        time.sleep(0.1) # Sleep for a bit so we don't overwhelm the server

if __name__ == "__main__":
    main()
