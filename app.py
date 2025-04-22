import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import math # For ceiling function
import argparse # Import argparse for command-line arguments
import numpy as np # Import NumPy

# --- Configuration Loading ---
# CONFIG_FILENAME = 'config.json' # Default filename (no longer primary way)

def load_config(filename): # Now takes filename as required argument
    """Loads the simulation configuration from a JSON file."""
    # Check if the provided filename exists directly
    if not os.path.exists(filename):
         # If not, check in the script's directory as a fallback
         script_dir = os.path.dirname(os.path.abspath(__file__))
         config_path = os.path.join(script_dir, filename)
         if not os.path.exists(config_path):
              print(f"ERROR: Configuration file '{filename}' not found directly or in script directory.")
              raise FileNotFoundError(f"Config file '{filename}' not found.")
    else:
        config_path = filename # Use the provided path directly

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print(f"Configuration loaded successfully from {config_path}")
        # Basic validation (can be expanded)
        if "simulation" not in config_data or "economy" not in config_data or "personas" not in config_data:
             raise ValueError("Config file missing required top-level keys: simulation, economy, personas.")
        if "resources" not in config_data["economy"] or not config_data["economy"]["resources"]:
             raise ValueError("Config file must define at least one resource in economy.resources.")
        if "actions" not in config_data["economy"]:
            print("WARNING: No actions defined in economy.actions.")
        return config_data
    except FileNotFoundError: # Should be caught above, but good practice
        print(f"ERROR: Configuration file '{filename}' not found.")
        raise
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR loading configuration from '{config_path}': {e}")
        raise


# --- Player State Class (Generic) ---
class Player:
    def __init__(self, env, player_id, persona_config, economy_config, sim_config):
        self.env = env
        self.player_id = player_id
        self.persona_config = persona_config
        self.economy_config = economy_config
        self.sim_config = sim_config
        self.name = persona_config.get("name", "Unnamed Persona")

        # Initialize resource balances based on config
        self.balances = {}
        self.resource_caps = {}
        self.resource_regen_rates = {}
        for resource in self.economy_config.get("resources", []):
            res_id = resource.get("id")
            if res_id:
                self.balances[res_id] = resource.get("initial_balance", 0)
                self.resource_caps[res_id] = resource.get("cap") # Can be None
                self.resource_regen_rates[res_id] = resource.get("regen_rate_per_day", 0)

        # Tracking stats
        self.action_counts = {action.get("id"): 0 for action in self.economy_config.get("actions", []) if action.get("id")}
        self.iap_spend_usd = 0.0
        self.log = []

        # Time series data for all resources
        self.balance_over_time = {res_id: [(0, balance)] for res_id, balance in self.balances.items()}

        # Start behavior processes
        self.action_process = env.process(self.perform_actions())
        if any(rate > 0 for rate in self.resource_regen_rates.values()):
            self.regen_process = env.process(self.regenerate_resources())

    def log_event(self, message):
        timestamp = self.env.now
        log_entry = f"Day {timestamp:.2f}: {self.name} [{self.player_id}] - {message}"
        self.log.append(log_entry)
        if self.sim_config.get("print_logs", False):
             print(log_entry)

    def record_balances(self):
        timestamp = self.env.now
        for res_id, balance in self.balances.items():
            # Ensure timestamp is float for consistency before appending
            self.balance_over_time[res_id].append((float(timestamp), balance))

    def change_resource(self, resource_id, amount, source_action=""):
        """Changes a resource balance, respecting caps if applicable."""
        if resource_id not in self.balances:
            self.log_event(f"Warning: Attempted to change unknown resource '{resource_id}'")
            return

        original_balance = self.balances[resource_id]
        new_balance = original_balance + amount

        # Apply cap if resource has one and amount is positive
        cap = self.resource_caps.get(resource_id)
        if cap is not None and amount > 0:
            new_balance = min(new_balance, cap)

        # Prevent balance from going below zero (usually - check game rules)
        new_balance = max(0, new_balance)

        # Update balance if it actually changed
        # Use a small tolerance for float comparison if necessary, but direct comparison is usually fine here
        if abs(new_balance - original_balance) > 1e-9 :
            self.balances[resource_id] = new_balance
            change = new_balance - original_balance
            verb = "Gained" if change > 0 else "Lost"
            # Format balance to avoid excessive decimals in logs
            current_bal_str = f"{self.balances[resource_id]:.2f}" if isinstance(self.balances[resource_id], float) else str(self.balances[resource_id])
            self.log_event(f"{verb} {abs(change):.2f} {resource_id} from '{source_action}'. New balance: {current_bal_str}")


    def can_afford(self, costs):
        """Checks if the player has enough resources for the costs."""
        if not costs: return True # No cost means affordable
        for cost in costs:
            res_id = cost.get("resource_id")
            amount = cost.get("amount", 0)
            if res_id not in self.balances or self.balances[res_id] < amount:
                return False
        return True

    def apply_costs(self, costs, action_id):
        """Deducts resources based on costs list."""
        if not costs: return
        for cost in costs:
            res_id = cost.get("resource_id")
            amount = cost.get("amount", 0)
            if res_id and amount > 0:
                 self.change_resource(res_id, -amount, action_id)

    def apply_outputs(self, outputs, action_id):
        """Adds resources based on outputs list, handling random amounts."""
        if not outputs: return
        for output in outputs:
            res_id = output.get("resource_id")
            if res_id:
                min_val = output.get("amount_min")
                max_val = output.get("amount_max")
                fixed_val = output.get("amount")

                amount = 0
                if fixed_val is not None:
                    amount = fixed_val
                elif min_val is not None and max_val is not None:
                    # Ensure min <= max before generating random int
                    if min_val <= max_val:
                        amount = random.randint(min_val, max_val)
                    else:
                        self.log_event(f"Warning: amount_min > amount_max for action '{action_id}', resource '{res_id}'. Using min_val.")
                        amount = min_val
                elif min_val is not None: # Only min specified
                    amount = min_val
                elif max_val is not None: # Only max specified (treat as fixed?)
                     amount = max_val

                if amount != 0: # Allow for potential negative outputs if designed
                    # Check if this output is linked to an IAP tier for tracking cost
                    is_iap = False
                    iap_cost = 0.0
                    # Find the action config efficiently
                    action_config = next((a for a in self.economy_config.get("actions", []) if a.get("id") == action_id), None)
                    if action_config and action_config.get("type") == "iap":
                         iap_tier_id = action_config.get("iap_tier_id")
                         iap_tier_info = self.economy_config.get("iap_tiers", {}).get(iap_tier_id)
                         if iap_tier_info:
                             iap_cost = iap_tier_info.get("cost_usd", 0.0)
                             is_iap = True # Mark as IAP source for potential different tracking later

                    self.change_resource(res_id, amount, action_id)
                    if is_iap and iap_cost > 0:
                        self.iap_spend_usd += iap_cost
                        self.log_event(f"Tracked ${iap_cost:.2f} spend for IAP action '{action_id}'")


    def perform_actions(self):
        """SimPy process that triggers player actions based on persona probabilities."""
        self.log_event("Action process started.")
        # Get all possible action IDs from the config
        all_action_configs = {a.get("id"): a for a in self.economy_config.get("actions", []) if a.get("id")}
        persona_action_probs = self.persona_config.get("action_probabilities_per_day", {})

        while True:
            day_start_time = self.env.now
            # self.log_event(f"--- Start of Day {math.ceil(day_start_time + 1e-9)} ---") # Add small epsilon for ceiling

            # Iterate through actions defined for the persona
            actions_to_attempt = list(persona_action_probs.items())
            random.shuffle(actions_to_attempt) # Randomize action order each day

            for action_id, daily_prob in actions_to_attempt:
                if action_id not in all_action_configs:
                    self.log_event(f"Warning: Action '{action_id}' in persona not found in economy actions.")
                    continue

                action_config = all_action_configs[action_id]
                costs = action_config.get("costs")
                outputs = action_config.get("outputs")
                frequency_type = action_config.get("frequency_type") # e.g., "daily"

                # Determine how many times to attempt the action today
                attempts_today = 0
                if frequency_type == "daily":
                    # Ensure daily actions only trigger once per day transition
                    # Check if current time is very close to an integer day value
                    if abs(day_start_time - round(day_start_time)) < 1e-9 :
                         attempts_today = 1
                elif daily_prob > 0:
                    # Use Poisson distribution for rates (more accurate for avg events/day)
                    attempts_today = np.random.poisson(daily_prob)


                # Perform the attempts
                for _ in range(attempts_today):
                    if self.can_afford(costs):
                        self.apply_costs(costs, action_id)
                        self.apply_outputs(outputs, action_id)
                        self.action_counts[action_id] = self.action_counts.get(action_id, 0) + 1 # Increment count
                    else:
                        pass # Decide if other logic needed (e.g., trigger IAP consideration)

            # Record balances AFTER potentially performing actions for the current time step
            self.record_balances()
            # Wait until the start of the next day
            yield self.env.timeout(1) # Simulate 1 day passing

    def regenerate_resources(self):
        """SimPy process for regenerating resources over time."""
        self.log_event("Regeneration process started.")
        while True:
            # Wait first, then regenerate. This ensures regeneration happens *before* actions on a given day.
            yield self.env.timeout(1) # Check regeneration daily
            for res_id, rate in self.resource_regen_rates.items():
                if rate > 0:
                    # Pass the specific source for clarity in logs
                    self.change_resource(res_id, rate, f"{res_id} Regeneration")
            # No need to record balance here, perform_actions records at end of day


# --- Simulation Setup Function ---
def run_simulation(config):
    """Sets up and runs the SimPy simulation based on config."""
    print("--- Starting Generic Economy Simulation ---")
    sim_config = config.get("simulation", {})
    # Use NumPy's random seed generator for consistency if needed elsewhere
    seed = sim_config.get("random_seed", None)
    np.random.seed(seed)
    # Also seed Python's random module if used elsewhere
    random.seed(seed)
    env = simpy.Environment()

    players = []
    player_id_counter = 1
    personas_config = config.get("personas", {})
    economy_config = config.get("economy", {})

    if not personas_config: print("WARNING: No personas found in configuration.")
    if not economy_config: print("WARNING: No economy parameters found in configuration.")

    for persona_key, persona_conf in personas_config.items():
        player = Player(env,
                        player_id=f"P{player_id_counter:03d}_{persona_key}",
                        persona_config=persona_conf,
                        economy_config=economy_config,
                        sim_config=sim_config)
        players.append(player)
        player_id_counter += 1

    sim_duration = sim_config.get("duration_days", 90)
    env.run(until=sim_duration)
    print(f"\n--- Simulation Finished ({sim_duration} days) ---")
    return players

# --- Results Analysis Function ---
def analyze_results(players, config):
    """Analyzes and prints summary results, generates plots."""
    results_summary = []
    all_balance_data = {} # {resource_id: {persona_name: pd.Series}}

    if not players:
        print("No players were simulated.")
        return

    # Initialize resource data structure
    for resource in config.get("economy", {}).get("resources", []):
        res_id = resource.get("id")
        if res_id:
            all_balance_data[res_id] = {}

    # Process player data
    for player in players:
        summary = {
            "Player ID": player.player_id,
            "Persona": player.name,
            "IAP Spend ($)": f"{player.iap_spend_usd:.2f}"
        }
        # Add final balances for each resource
        for res_id, balance in player.balances.items():
             # Format final balance for readability
             summary[f"Final {res_id}"] = f"{balance:.2f}" if isinstance(balance, float) else str(balance)

        results_summary.append(summary)

        # Prepare data for plotting each resource
        for res_id, balance_log in player.balance_over_time.items():
             if balance_log:
                 # Create a DataFrame from the log
                 log_df = pd.DataFrame(balance_log, columns=['day', 'balance'])
                 # Drop potential duplicate timestamps, keeping the last entry for that time
                 log_df = log_df.drop_duplicates(subset=['day'], keep='last')

                 # Set 'day' as index (should be unique now)
                 # Convert SimPy float time to Timedelta for proper time-series indexing
                 log_df['day'] = pd.to_timedelta(log_df['day'], unit='D')
                 log_df = log_df.set_index('day')

                 # Ensure index covers the full simulation duration for reindexing
                 sim_duration = config.get("simulation", {}).get("duration_days", 90)
                 # Create index from day 0 up to *and including* the final day
                 full_index = pd.to_timedelta(range(sim_duration + 1), unit='D')

                 # Reindex and forward fill
                 # Use the unique index from log_df
                 resampled_series = log_df['balance'].reindex(full_index).ffill()

                 # Fill initial NaN if day 0 wasn't explicitly logged
                 resampled_series = resampled_series.fillna(method='bfill').fillna(0) # Backfill then fill remaining NaN with 0

                 # Convert TimedeltaIndex back to float days for plotting
                 resampled_series.index = resampled_series.index.total_seconds() / (24 * 3600)

                 if res_id in all_balance_data:
                      all_balance_data[res_id][player.name] = resampled_series
             else:
                 print(f"Warning: No balance data for resource '{res_id}' for player {player.player_id}")


    summary_df = pd.DataFrame(results_summary)
    print("\n--- Final Player Summary ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_df.to_string(index=False))

    # --- Plotting ---
    if config.get("simulation", {}).get("plot_results", False) and any(all_balance_data.values()):
        print("\n--- Generating Plots ---")
        num_resources = len(all_balance_data)
        # Adjust subplot creation for single resource case
        if num_resources == 1:
             fig, axes = plt.subplots(1, 1, figsize=(12, 5), squeeze=False)
        else:
             # Ensure enough vertical space per plot
             fig_height = max(5, 4 * num_resources) # Min height 5, scale otherwise
             fig, axes = plt.subplots(num_resources, 1, figsize=(12, fig_height), sharex=True, squeeze=False) # Ensure axes is always 2D array

        resource_keys = list(all_balance_data.keys())

        for i, res_id in enumerate(resource_keys):
            ax = axes[i, 0] # Access subplot correctly
            persona_balances = all_balance_data[res_id]
            if not persona_balances:
                 ax.set_title(f"'{res_id}' Balance Over Time (No Data)")
                 continue

            # Check if dictionary is empty before iterating
            if not persona_balances:
                 print(f"Warning: No balance series data found for resource '{res_id}'")
                 ax.set_title(f"'{res_id}' Balance Over Time (No Data)")
                 continue

            for name, balance_series in persona_balances.items():
                if not balance_series.empty:
                     # Plot integer days if possible for cleaner axis
                     plot_index = balance_series.index
                     plot_values = balance_series.values
                     ax.plot(plot_index, plot_values, label=f"{name}", marker='.', linestyle='-', markersize=3, alpha=0.8)
                else:
                     print(f"Warning: Empty balance series for {name} - {res_id}")


            ax.set_ylabel(f"{res_id} Balance")
            ax.set_title(f"'{res_id}' Balance Over Time")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            # Optionally set y-axis limit to start at 0 if appropriate
            ax.set_ylim(bottom=0)

        axes[-1, 0].set_xlabel("Simulation Day") # Label only the bottom axis
        sim_duration = config.get("simulation", {}).get("duration_days", 90)
        fig.suptitle(f"Resource Balances Over {sim_duration} Days", fontsize=16, y=1.0) # Adjust title position
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout
        plt.show()

    elif config.get("simulation", {}).get("plot_results", False):
         print("Plotting enabled, but no balance data was available to plot.")


# --- Main Execution ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a generic F2P economy simulation.")
    parser.add_argument(
        "-c", "--config",
        default="config.json", # Default config filename
        help="Path to the JSON configuration file (default: config.json)"
    )
    args = parser.parse_args()

    try:
        # Load config from the specified file
        current_config = load_config(args.config)

        # Run simulation
        simulated_players = run_simulation(current_config)

        # Analyze and display results
        analyze_results(simulated_players, current_config)

    except FileNotFoundError:
         # Error already printed in load_config
         print("Simulation aborted due to missing config file.")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        print("Simulation aborted.")

