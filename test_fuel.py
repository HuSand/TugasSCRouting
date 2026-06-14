# test_fuel.py
from settings import Settings
from src.routing.base import Vehicle
from src.routing.fuel import inject_fuel_stops, extract_spbu_nodes

cfg     = Settings()
vehicle = Vehicle.from_settings(cfg, "motor")

print(f"Vehicle : {vehicle.label}")
print(f"Range   : {vehicle.range_km} km")
print(f"Tank    : {vehicle.tank_liters} liter")
print(f"Threshold: {vehicle.refill_threshold_km} km")
print("Import OK!")