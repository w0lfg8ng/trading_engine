#!/usr/bin/env python3
"""
Emergency stop script for trading engine
"""

import psutil
import sqlite3
import json
from datetime import datetime
from pathlib import Path

def emergency_stop():
    print("🚨 EMERGENCY STOP INITIATED")
    
    # Stop trading engine
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('6.26.25_live_trading_engine_v6.py' in arg for arg in proc.info['cmdline']):
                proc.terminate()
                print(f"✅ Terminated trading engine (PID: {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Create emergency stop flag
    emergency_file = Path("../data/emergency_stop.flag")
    with open(emergency_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'reason': 'Manual emergency stop'
        }, f)
    
    print("🛑 Emergency stop completed. Remove ../data/emergency_stop.flag to resume.")

if __name__ == "__main__":
    emergency_stop()