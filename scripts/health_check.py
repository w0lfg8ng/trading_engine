#!/usr/bin/env python3
"""
Health check script for the automated trading pipeline
"""

import os
import sys
import sqlite3
import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path

def check_trading_engine():
    """Check if trading engine is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('6.26.25_live_trading_engine_v6.py' in arg for arg in proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def check_data_freshness():
    """Check if data is fresh"""
    try:
        status_file = Path("../data/pipeline_status.json")
        if not status_file.exists():
            return False
        
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        last_update = datetime.fromisoformat(status['last_update'].replace('Z', '+00:00'))
        hours_old = (datetime.now() - last_update.replace(tzinfo=None)).total_seconds() / 3600
        
        return hours_old < 6  # Data should be less than 6 hours old
        
    except Exception:
        return False

def main():
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'trading_engine_running': check_trading_engine(),
        'data_fresh': check_data_freshness(),
        'overall_healthy': False
    }
    
    health_status['overall_healthy'] = (
        health_status['trading_engine_running'] and 
        health_status['data_fresh']
    )
    
    if not health_status['overall_healthy']:
        print(f"❌ HEALTH CHECK FAILED: {health_status}")
        sys.exit(1)
    else:
        print(f"✅ System healthy: {health_status}")
        sys.exit(0)

if __name__ == "__main__":
    main()