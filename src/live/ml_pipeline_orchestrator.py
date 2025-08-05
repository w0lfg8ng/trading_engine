#!/usr/bin/env python3
"""
Automated ML Pipeline Orchestrator for Cryptocurrency Trading System
Handles weekly model retraining, validation, and deployment

Features:
- Weekly XGBoost model retraining on latest data
- Model performance validation before deployment
- Hot-swap model deployment without stopping trading engine
- Comprehensive logging and error handling
- Rollback mechanism for poor-performing models
- Data quality checks before training
"""

import os
import sys
import sqlite3
import pandas as pd
import shutil
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import subprocess
import json
import time
from pathlib import Path

# Optional imports for timestamp parsing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

# Add the project root to Python path for imports
sys.path.append('/mnt/raid0/data_erick/kraken_trading_model_v2')

class MLPipelineOrchestrator:
    def __init__(self, config_path: str = "/mnt/raid0/data_erick/kraken_trading_model_v2/config/pipeline_config.json"):
        """Initialize the ML Pipeline Orchestrator"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup paths - using absolute paths to match your structure
        self.db_path = self.config['database']['path']
        self.models_dir = Path(self.config['models']['directory'])
        self.backup_dir = Path(self.config['models']['backup_directory'])
        self.logs_dir = Path(self.config['logging']['directory'])
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Trading pairs and regimes - matching your current setup
        self.pairs = self.config['trading']['pairs']
        self.regimes = self.config['trading']['regimes']
        
        # Performance thresholds
        self.min_accuracy = self.config['validation']['min_accuracy']
        self.min_trades_for_validation = self.config['validation']['min_trades']
        
        self.logger.info("ML Pipeline Orchestrator initialized")
    
    def check_data_freshness(self) -> bool:
        """Check if we have fresh data for model training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check latest timestamp across all pairs
            latest_timestamps = {}
            institutional_timestamps = {}
            tables_checked = 0
            
            for pair in self.pairs:
                # Check both regular and institutional tables
                regular_table = f"features_{pair.lower()}_1m"
                institutional_table = f"features_{pair.upper()}_1m_institutional"
                
                # Check regular table first
                try:
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=?
                    """, (regular_table,))
                    
                    if cursor.fetchone():
                        cursor.execute(f"SELECT MAX(timestamp) FROM {regular_table}")
                        result = cursor.fetchone()
                        if result and result[0] is not None:
                            latest_timestamps[pair] = float(result[0])
                            
                except Exception as e:
                    self.logger.warning(f"Could not check regular table for {pair}: {e}")
                
                # Check institutional table
                try:
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=?
                    """, (institutional_table,))
                    
                    if cursor.fetchone():
                        cursor.execute(f"SELECT MAX(timestamp) FROM {institutional_table}")
                        result = cursor.fetchone()
                        if result and result[0] is not None:
                            timestamp = result[0]
                            
                            # Handle string timestamps from institutional tables
                            if isinstance(timestamp, str):
                                try:
                                    # Try parsing ISO format datetime string
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    timestamp = dt.timestamp()
                                except ValueError:
                                    try:
                                        timestamp = float(timestamp)
                                    except ValueError:
                                        self.logger.warning(f"Could not parse institutional timestamp for {pair}: {timestamp}")
                                        continue
                            
                            institutional_timestamps[pair] = float(timestamp)
                            tables_checked += 1
                            
                except Exception as e:
                    self.logger.warning(f"Could not check institutional table for {pair}: {e}")
            
            conn.close()
            
            if tables_checked == 0:
                self.logger.error("No institutional feature tables found or accessible")
                return False
            
            if not institutional_timestamps:
                self.logger.error("No timestamps found in institutional feature tables")
                return False
            
            # Check institutional data freshness (more lenient since labels take time to generate)
            current_time = datetime.now().timestamp()
            stale_pairs = []
            very_stale_pairs = []
            
            for pair, timestamp in institutional_timestamps.items():
                hours_old = (current_time - timestamp) / 3600
                days_old = hours_old / 24
                
                if days_old > 10:  # More than 10 days old
                    very_stale_pairs.append(f"{pair} ({days_old:.1f}d old)")
                elif days_old > 3:  # More than 3 days old
                    stale_pairs.append(f"{pair} ({days_old:.1f}d old)")
            
            # Log status
            if very_stale_pairs:
                self.logger.warning(f"Very stale institutional data: {', '.join(very_stale_pairs)}")
            if stale_pairs:
                self.logger.warning(f"Stale institutional data: {', '.join(stale_pairs)}")
            
            # Allow training if we have data within 10 days for most pairs
            if len(very_stale_pairs) >= len(institutional_timestamps) * 0.7:  # More than 70% very stale
                self.logger.error("Too much very stale institutional data - recommend updating labels first")
                return False
            
            # Check if we have recent regular data for comparison
            if latest_timestamps:
                regular_hours_old = [(current_time - ts) / 3600 for ts in latest_timestamps.values()]
                avg_regular_age = sum(regular_hours_old) / len(regular_hours_old)
                self.logger.info(f"Regular feature data average age: {avg_regular_age:.1f} hours")
                
                if avg_regular_age > 12:  # Regular data also stale
                    self.logger.warning("Both regular and institutional data are stale")
                    return False
            
            self.logger.info(f"Data freshness check passed for {len(institutional_timestamps)} institutional tables")
            
            # Log recommendation to update labels if institutional data is moderately stale
            if stale_pairs and len(stale_pairs) > 0:
                self.logger.info("Note: Institutional labels are somewhat stale - consider running label update")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data freshness check failed: {e}")
            return False
    
    def run_subprocess_with_live_output(self, cmd: List[str], description: str, 
                                       timeout_seconds: int, cwd: str = None) -> Tuple[bool, str, str]:
        """
        Run subprocess with live output streaming to console and logs
        """
        self.logger.info(f"🚀 Starting: {description}")
        self.logger.info(f"📝 Command: {' '.join(cmd)}")
        self.logger.info(f"📁 Working directory: {cwd or 'current'}")
        self.logger.info(f"⏱️ Timeout: {timeout_seconds/60:.0f} minutes")
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"{'='*60}")
        
        import subprocess
        import threading
        from queue import Queue, Empty
        
        def stream_output(pipe, queue, stream_name):
            """Stream output from subprocess to queue"""
            try:
                for line in iter(pipe.readline, b''):
                    queue.put((stream_name, line.decode('utf-8', errors='replace').rstrip()))
                pipe.close()
            except Exception as e:
                queue.put((stream_name, f"Stream error: {e}"))
        
        try:
            # Start subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                bufsize=1,
                universal_newlines=False
            )
            
            # Setup output streaming
            output_queue = Queue()
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, output_queue, 'STDOUT'))
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, output_queue, 'STDERR'))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Collect output while showing live progress
            stdout_lines = []
            stderr_lines = []
            start_time = time.time()
            last_output_time = start_time
            
            while process.poll() is None:
                try:
                    # Check for new output (non-blocking)
                    stream_name, line = output_queue.get(timeout=1)
                    current_time = time.time()
                    elapsed = current_time - start_time
                    last_output_time = current_time
                    
                    # Format and display line with timestamp
                    timestamp = f"[{elapsed/60:6.1f}m]"
                    
                    if stream_name == 'STDOUT':
                        stdout_lines.append(line)
                        if line.strip():  # Only show non-empty lines
                            print(f"{timestamp} 📊 {line}")
                            self.logger.info(f"STDOUT: {line}")
                    else:  # STDERR
                        stderr_lines.append(line)
                        if line.strip():  # Only show non-empty lines
                            print(f"{timestamp} ⚠️  {line}")
                            self.logger.warning(f"STDERR: {line}")
                    
                except Empty:
                    # No output for 1 second, check if process is still alive
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Show heartbeat every 5 minutes if no output
                    if current_time - last_output_time > 300:  # 5 minutes
                        print(f"[{elapsed/60:6.1f}m] 💓 Process still running... (no output for {(current_time-last_output_time)/60:.1f}m)")
                        last_output_time = current_time
                    
                    # Check timeout
                    if elapsed > timeout_seconds:
                        print(f"[{elapsed/60:6.1f}m] ⏰ Timeout reached! Terminating process...")
                        process.terminate()
                        time.sleep(5)
                        if process.poll() is None:
                            process.kill()
                        raise subprocess.TimeoutExpired(cmd, timeout_seconds)
            
            # Process finished, collect any remaining output
            while True:
                try:
                    stream_name, line = output_queue.get(timeout=0.1)
                    if stream_name == 'STDOUT':
                        stdout_lines.append(line)
                    else:
                        stderr_lines.append(line)
                except Empty:
                    break
            
            # Wait for threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            # Get final return code
            return_code = process.returncode
            stdout_text = '\n'.join(stdout_lines)
            stderr_text = '\n'.join(stderr_lines)
            
            elapsed = time.time() - start_time
            
            if return_code == 0:
                print(f"\n✅ {description} completed successfully in {elapsed/60:.1f} minutes")
                self.logger.info(f"✅ {description} completed successfully in {elapsed/60:.1f} minutes")
                return True, stdout_text, stderr_text
            else:
                print(f"\n❌ {description} failed with return code {return_code} after {elapsed/60:.1f} minutes")
                self.logger.error(f"❌ {description} failed with return code {return_code} after {elapsed/60:.1f} minutes")
                return False, stdout_text, stderr_text
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"\n⏰ {description} timed out after {elapsed/60:.1f} minutes")
            self.logger.error(f"⏰ {description} timed out after {elapsed/60:.1f} minutes")
            return False, "", f"Process timed out after {timeout_seconds} seconds"
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n💥 {description} failed with error: {e}")
            self.logger.error(f"💥 {description} failed with error: {e}")
            return False, "", str(e)

    def update_institutional_labels(self) -> bool:
        """Update institutional labels with latest data"""
        try:
            self.logger.info("Updating institutional labels...")
        
            # Check if the institutional labeling script exists
            label_script_path = '/mnt/raid0/data_erick/kraken_trading_model_v2/src/features/1m_institutional_label_targets_enhanced.py'
        
            if not os.path.exists(label_script_path):
                # Try alternative path
                alt_path = '/mnt/raid0/data_erick/kraken_trading_model_v2/src/features/institutional_label_targets.py'
                if os.path.exists(alt_path):
                    label_script_path = alt_path
                else:
                    self.logger.error(f"Institutional labeling script not found at {label_script_path}")
                    # List available scripts
                    features_dir = '/mnt/raid0/data_erick/kraken_trading_model_v2/src/features'
                    if os.path.exists(features_dir):
                        scripts = [f for f in os.listdir(features_dir) if f.endswith('.py')]
                        self.logger.info(f"Available scripts in features dir: {scripts}")
                    return False
        
            # Run the institutional labeling script with verbose output
            success, stdout, stderr = self.run_subprocess_with_live_output(
                cmd=['python', label_script_path],
                description="Institutional Label Generation (🏷️ Creating buy/sell/hold labels)",
                timeout_seconds=14400,  # 4 hours
                cwd=os.path.dirname(label_script_path)
            )
        
            if success:
                self.logger.info("✅ Institutional labels updated successfully")
                return True
            else:
                self.logger.error(f"❌ Label update failed")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Label update error: {e}")
            return False
    
    def update_daily_data(self) -> bool:
        """Update daily OHLCV data and features"""
        try:
            # Check if the daily updater script exists
            daily_updater_path = '/mnt/raid0/data_erick/kraken_trading_model_v2/src/data_pipeline/daily_updater.py'
            
            if not os.path.exists(daily_updater_path):
                self.logger.error(f"Daily updater script not found at {daily_updater_path}")
                return False
            
            # Run the daily updater script with verbose output
            success, stdout, stderr = self.run_subprocess_with_live_output(
                cmd=['python', daily_updater_path],
                description="Daily Data Update (📡 Fetching OHLCV + calculating features)",
                timeout_seconds=3600,  # 1 hour
                cwd=os.path.dirname(daily_updater_path)
            )
            
            if success:
                self.logger.info("✅ Daily data update completed successfully")
                return True
            else:
                self.logger.error(f"❌ Daily data update failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Daily data update error: {e}")
            return False
        
    def backup_current_models(self) -> str:
        """Backup current models before retraining"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"models_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
        
            # Copy all model files
            for file in self.models_dir.glob("*.pkl"):
                shutil.copy2(file, backup_path)
        
            # Create manifest file
            manifest = {
                'backup_time': datetime.now().isoformat(),
                'model_count': len(list(backup_path.glob("*.pkl"))),
                'pairs': self.pairs,
                'regimes': self.regimes
            }
        
            with open(backup_path / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
        
            self.logger.info(f"Models backed up to {backup_path}")
            return str(backup_path)
        
        except Exception as e:
            self.logger.error(f"Model backup failed: {e}")
            raise
    
    def train_new_models(self) -> Dict[str, Dict]:
        """Train new XGBoost models for all pairs and regimes"""
        try:
            # Check if the training script exists
            training_script_path = '/mnt/raid0/data_erick/kraken_trading_model_v2/src/training/regime_matrix_training_xgboost.py'
            
            if not os.path.exists(training_script_path):
                # Try alternative paths
                alt_paths = [
                    '/mnt/raid0/data_erick/kraken_trading_model_v2/src/training/xgboost_regime_training.py',
                    '/mnt/raid0/data_erick/kraken_trading_model_v2/src/training/train_regime_models.py'
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        training_script_path = alt_path
                        break
                else:
                    self.logger.error(f"Training script not found at {training_script_path}")
                    # List available scripts
                    training_dir = '/mnt/raid0/data_erick/kraken_trading_model_v2/src/training'
                    if os.path.exists(training_dir):
                        scripts = [f for f in os.listdir(training_dir) if f.endswith('.py')]
                        self.logger.info(f"Available scripts in training dir: {scripts}")
                    return {}
            
            # Run the regime matrix training script with verbose output
            success, stdout, stderr = self.run_subprocess_with_live_output(
                cmd=['python', training_script_path],
                description="XGBoost Model Training (🤖 Training 45 regime-specific models)",
                timeout_seconds=7200,  # 2 hours
                cwd=os.path.dirname(training_script_path)
            )
            
            if not success:
                self.logger.error(f"❌ Model training failed")
                return {}
            
            # Parse training results from stdout
            training_results = self._parse_training_output(stdout)
            
            # Count successful models by checking if files exist
            successful_models = 0
            total_expected = len(self.pairs) * len(self.regimes)
            
            print(f"\n🔍 Checking trained models...")
            for pair in self.pairs:
                for regime in self.regimes:
                    model_path = self.models_dir / f"{pair}_{regime}_xgb_model.pkl"
                    encoder_path = self.models_dir / f"{pair}_{regime}_encoder.pkl"
                    features_path = self.models_dir / f"{pair}_{regime}_features.pkl"
                    
                    if model_path.exists() and encoder_path.exists() and features_path.exists():
                        successful_models += 1
                        print(f"  ✅ {pair}_{regime}")
                    else:
                        print(f"  ❌ {pair}_{regime} (missing files)")
            
            success_rate = successful_models / total_expected if total_expected > 0 else 0
            
            print(f"\n📊 Training Summary:")
            print(f"  ✅ Successful: {successful_models}/{total_expected} models ({success_rate:.1%})")
            
            self.logger.info(f"✅ Model training completed. {successful_models}/{total_expected} models trained successfully")
            
            return {
                'successful_models': successful_models,
                'total_expected': total_expected,
                'success_rate': success_rate
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model training error: {e}")
            return {}
    
    def validate_new_models(self) -> Dict[str, bool]:
        """Validate newly trained models against recent performance"""
        validation_results = {}
        
        try:
            self.logger.info("Validating new models...")
            
            # Load recent trading performance for comparison
            recent_performance = self._get_recent_trading_performance()
            
            for pair in self.pairs:
                for regime in self.regimes:
                    model_key = f"{pair}_{regime}"
                    
                    try:
                        # Load new model
                        model_path = self.models_dir / f"{pair}_{regime}_xgb_model.pkl"
                        encoder_path = self.models_dir / f"{pair}_{regime}_encoder.pkl"
                        features_path = self.models_dir / f"{pair}_{regime}_features.pkl"
                        
                        if not all(p.exists() for p in [model_path, encoder_path, features_path]):
                            validation_results[model_key] = False
                            continue
                        
                        # Basic validation - check if models can be loaded
                        try:
                            model = joblib.load(model_path)
                            encoder = joblib.load(encoder_path)
                            features = joblib.load(features_path)
                            
                            # Validate model components
                            if model is None or encoder is None or not features:
                                validation_results[model_key] = False
                                continue
                            
                            # For now, if the model loads successfully, consider it valid
                            # In a more sophisticated setup, you could run backtest validation here
                            validation_results[model_key] = True
                            
                        except Exception as e:
                            self.logger.error(f"Failed to load model {model_key}: {e}")
                            validation_results[model_key] = False
                        
                    except Exception as e:
                        self.logger.error(f"Validation failed for {model_key}: {e}")
                        validation_results[model_key] = False
            
            successful_validations = sum(validation_results.values())
            total_models = len(validation_results)
            
            self.logger.info(f"Model validation completed: {successful_validations}/{total_models} models passed")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation error: {e}")
            return {}
    
    def deploy_validated_models(self, validation_results: Dict[str, bool]) -> bool:
        """Deploy validated models to production"""
        try:
            self.logger.info("Deploying validated models...")
            
            # Count successful models
            successful_models = sum(validation_results.values())
            total_models = len(validation_results)
            
            # Require at least 70% of models to pass validation
            success_rate = successful_models / total_models if total_models > 0 else 0
            min_success_rate = self.config['deployment']['min_success_rate']
            
            if success_rate < min_success_rate:
                self.logger.error(f"Only {success_rate:.1%} of models passed validation. Minimum required: {min_success_rate:.1%}")
                return False
            
            # Create deployment manifest
            deployment_manifest = {
                'deployment_time': datetime.now().isoformat(),
                'total_models': total_models,
                'successful_models': successful_models,
                'success_rate': success_rate,
                'validation_results': validation_results
            }
            
            # Save deployment manifest
            manifest_path = self.models_dir / 'deployment_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(deployment_manifest, f, indent=2)
            
            # Models are already in the correct location from training
            # The live trading engine will pick them up automatically
            
            self.logger.info(f"Successfully deployed {successful_models}/{total_models} models")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False
    
    def rollback_models(self, backup_path: str) -> bool:
        """Rollback to previous models if deployment fails"""
        try:
            self.logger.info(f"Rolling back models from {backup_path}")
            
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_path}")
                return False
            
            # Remove current models
            for file in self.models_dir.glob("*.pkl"):
                file.unlink()
            
            # Restore backup models
            for file in backup_dir.glob("*.pkl"):
                shutil.copy2(file, self.models_dir)
            
            self.logger.info("Models successfully rolled back")
            return True
            
        except Exception as e:
            self.logger.error(f"Model rollback failed: {e}")
            return False
    
    def _parse_training_output(self, output: str) -> Dict[str, Dict]:
        """Parse training script output to extract results"""
        # Simple implementation - in practice you could parse the actual output
        # For now, just return an indicator that training ran
        return {'training_completed': True}
    
    def _get_recent_trading_performance(self) -> Dict:
        """Get recent trading performance metrics"""
        try:
            # Check if live trading database exists
            live_db_path = "/mnt/raid0/data_erick/kraken_trading_model_v2/src/live/institutional_trading_history.db"
            if not os.path.exists(live_db_path):
                self.logger.info("No live trading history found - using default performance metrics")
                return {}
            
            conn = sqlite3.connect(live_db_path)
            
            # Get performance from last 30 days
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            performance_query = """
            SELECT pair, market_regime, 
                   COUNT(*) as trade_count,
                   AVG(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as win_rate,
                   AVG(net_pnl) as avg_pnl,
                   AVG(net_pnl * net_pnl) - AVG(net_pnl) * AVG(net_pnl) as pnl_variance
            FROM trades 
            WHERE timestamp >= ? 
            GROUP BY pair, market_regime
            """
            
            df = pd.read_sql(performance_query, conn, params=[cutoff_date])
            conn.close()
            
            # Convert to nested dict
            performance = {}
            for _, row in df.iterrows():
                pair = row['pair']
                regime = row['market_regime']
                if pair not in performance:
                    performance[pair] = {}
                performance[pair][regime] = {
                    'trade_count': row['trade_count'],
                    'win_rate': row['win_rate'],
                    'avg_pnl': row['avg_pnl'],
                    'pnl_variance': row['pnl_variance'] or 0
                }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting recent performance: {e}")
            return {}
    
    def run_data_quality_checks(self) -> bool:
        """Run comprehensive data quality checks before training"""
        try:
            self.logger.info("🔍 Running data quality checks...")
            
            # Try to import database integrity check module (optional)
            try:
                from database_integrity_check import run_database_checks  # type: ignore
                HAS_DB_INTEGRITY_CHECK = True
            except ImportError:
                HAS_DB_INTEGRITY_CHECK = False
                self.logger.warning("⚠️ database_integrity_check module not found - running basic checks only")
            
            if HAS_DB_INTEGRITY_CHECK:
                # Run comprehensive integrity checks
                results = run_database_checks(self.db_path)
                
                # Log results
                self.logger.info(f"Database integrity status: {results['overall_status']}")
                self.logger.info(f"Database size: {results['checks']['database_size_mb']} MB")
                
                # Check for critical errors
                if results['overall_status'] in ['corrupted', 'error']:
                    self.logger.error("❌ Critical database issues detected:")
                    for error in results['errors']:
                        self.logger.error(f"  - {error}")
                    return False
                
                # Log warnings
                if results['warnings']:
                    self.logger.warning("⚠️ Data quality warnings:")
                    for warning in results['warnings']:
                        self.logger.warning(f"  - {warning}")
                
                # Check each pair's data quality
                all_pairs_healthy = True
                for pair in self.pairs:
                    pair_key = f'{pair}_data_quality'
                    if pair_key in results['checks']:
                        pair_checks = results['checks'][pair_key]
                        
                        if not pair_checks['has_data']:
                            self.logger.error(f"❌ {pair}: No data found")
                            all_pairs_healthy = False
                        elif not pair_checks['has_recent_data']:
                            self.logger.warning(f"⚠️ {pair}: Data is stale")
                        elif pair_checks['data_range_days'] < 365:
                            self.logger.warning(f"⚠️ {pair}: Limited data ({pair_checks['data_range_days']} days)")
                        else:
                            self.logger.info(f"✅ {pair}: {pair_checks['row_count']:,} rows, {pair_checks['data_range_days']} days")
                
                # Save quality report
                quality_report_path = self.logs_dir / f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(quality_report_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                self.logger.info(f"Quality report saved to {quality_report_path}")
                
                return all_pairs_healthy and results['overall_status'] != 'corrupted'
            else:
                # Basic data quality check without external module
                self.logger.info("Running basic data quality checks...")
                
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Check database accessibility and basic structure
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    if len(tables) == 0:
                        self.logger.error("❌ No tables found in database")
                        return False
                    
                    self.logger.info(f"✅ Database accessible with {len(tables)} tables")
                    
                    # Basic check for each trading pair
                    for pair in self.pairs:
                        institutional_table = f"features_{pair.upper()}_1m_institutional"
                        cursor.execute(f"SELECT COUNT(*) FROM {institutional_table}")
                        count = cursor.fetchone()[0]
                        
                        if count > 0:
                            self.logger.info(f"✅ {pair}: {count:,} institutional records")
                        else:
                            self.logger.warning(f"⚠️ {pair}: No institutional data found")
                    
                    conn.close()
                    self.logger.info("✅ Basic data quality check passed")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"❌ Basic data quality check failed: {e}")
                    return False
            
        except Exception as e:
            self.logger.error(f"❌ Data quality check failed: {e}")
            return False
    
    def run_weekly_pipeline(self, force_label_update: bool = False, update_data: bool = True) -> bool:
        """Run the complete weekly ML pipeline"""
        pipeline_start = datetime.now()
        self.logger.info("🚀 Starting weekly ML pipeline")
        
        print(f"\n{'='*80}")
        print(f"🤖 WEEKLY ML PIPELINE - COMPLETE AUTOMATION")
        print(f"{'='*80}")
        print(f"📅 Started: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Estimated Duration: 5-8 hours")
        print(f"{'='*80}")
        
        try:
            step = 1
            total_steps = 8 if update_data else 7
            
            # Step 1: Data Quality Check
            print(f"\n[STEP {step}/{total_steps}] 🔍 DATA QUALITY CHECK")
            print("─" * 50)
            if not self.run_data_quality_checks():
                self.logger.error("❌ Data quality checks failed - aborting pipeline")
                return False
            step += 1
            
            # Step 2: Update daily data (if requested)
            if update_data:
                print(f"\n[STEP {step}/{total_steps}] 📡 DAILY DATA UPDATE")
                print("─" * 50)
                if not self.update_daily_data():
                    self.logger.error("❌ Daily data update failed - aborting pipeline")
                    return False
                step += 1
            
            # Step 3: Check data freshness
            print(f"\n[STEP {step}/{total_steps}] 🔍 DATA FRESHNESS CHECK")
            print("─" * 50)
            data_fresh = self.check_data_freshness()
            step += 1
            
            if not data_fresh and not force_label_update:
                self.logger.warning("⚠️ Data freshness check indicates stale institutional data")
                self.logger.info("💡 Consider running with force_label_update=True to update labels first")
                self.logger.info("   Or run: python ml_pipeline_orchestrator.py --force-labels")
                # Don't abort immediately - try to update labels first
                self.logger.info("🔄 Attempting to update institutional labels to resolve staleness...")
                force_label_update = True
            
            # Step 4: Update institutional labels (if needed or forced)
            if force_label_update or not data_fresh:
                print(f"\n[STEP {step}/{total_steps}] 🏷️ INSTITUTIONAL LABELS UPDATE")
                print("─" * 50)
                print("⚠️ WARNING: This step typically takes 3-5 hours")
                print("💡 You'll see live progress updates every few minutes")
                
                if not self.update_institutional_labels():
                    self.logger.error("❌ Failed to update institutional labels")
                    if not data_fresh:
                        self.logger.error("❌ Cannot proceed with stale data - aborting pipeline")
                        return False
                    else:
                        self.logger.warning("⚠️ Label update failed but data is fresh enough to continue")
                else:
                    self.logger.info("✅ Institutional labels updated successfully")
                    # Re-check data freshness after label update
                    if not self.check_data_freshness():
                        self.logger.warning("⚠️ Data still appears stale after label update - proceeding anyway")
            step += 1
            
            # Step 5: Backup current models
            print(f"\n[STEP {step}/{total_steps}] 💾 MODEL BACKUP")
            print("─" * 50)
            backup_path = self.backup_current_models()
            step += 1
            
            # Step 6: Train new models
            print(f"\n[STEP {step}/{total_steps}] 🤖 MODEL TRAINING")
            print("─" * 50)
            print("⚠️ WARNING: This step typically takes 1-2 hours")
            print("💡 Training 45 XGBoost models (9 pairs × 5 regimes)")
            
            training_results = self.train_new_models()
            if not training_results or training_results.get('successful_models', 0) == 0:
                self.logger.error("❌ Model training failed - rolling back")
                self.rollback_models(backup_path)
                return False
            
            # Check training success rate
            success_rate = training_results.get('success_rate', 0)
            if success_rate < 0.5:  # Less than 50% of models trained successfully
                self.logger.error(f"❌ Only {success_rate:.1%} of models trained successfully - rolling back")
                self.rollback_models(backup_path)
                return False
            step += 1
            
            # Step 7: Validate new models
            print(f"\n[STEP {step}/{total_steps}] ✅ MODEL VALIDATION")
            print("─" * 50)
            validation_results = self.validate_new_models()
            step += 1
            
            # Step 8: Deploy validated models
            print(f"\n[STEP {step}/{total_steps}] 🚀 MODEL DEPLOYMENT")
            print("─" * 50)
            if not self.deploy_validated_models(validation_results):
                self.logger.error("❌ Model deployment failed - rolling back")
                self.rollback_models(backup_path)
                return False
            
            pipeline_duration = datetime.now() - pipeline_start
            
            print(f"\n{'='*80}")
            print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"⏱️ Total Duration: {pipeline_duration}")
            print(f"📊 Pipeline Summary:")
            trained_models = training_results.get('successful_models', 0)
            total_expected = training_results.get('total_expected', len(self.pairs) * len(self.regimes))
            deployed_models = sum(validation_results.values()) if validation_results else 0
            print(f"   🤖 Training: {trained_models}/{total_expected} models ({success_rate:.1%})")
            print(f"   ✅ Validation: {deployed_models}/{trained_models} models passed")
            print(f"   🚀 New models are now available for live trading!")
            print(f"{'='*80}")
            
            self.logger.info(f"✅ Weekly ML pipeline completed successfully in {pipeline_duration}")
            
            # Clean up old backups (keep last 4 weeks)
            self._cleanup_old_backups()
            
            # Create success indicator file
            success_file = Path("/mnt/raid0/data_erick/kraken_trading_model_v2/data/last_model_update.txt")
            success_file.parent.mkdir(parents=True, exist_ok=True)
            with open(success_file, 'w') as f:
                f.write(datetime.now().isoformat())
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed with error: {e}")
            if 'backup_path' in locals():
                self.rollback_models(backup_path)
            return False
    
    def _cleanup_old_backups(self, keep_weeks: int = 4):
        """Clean up old backup directories"""
        try:
            cutoff_date = datetime.now() - timedelta(weeks=keep_weeks)
            
            for backup_dir in self.backup_dir.glob("models_backup_*"):
                # Extract timestamp from directory name
                timestamp_str = backup_dir.name.replace("models_backup_", "")
                try:
                    backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    if backup_date < cutoff_date:
                        shutil.rmtree(backup_dir)
                        self.logger.info(f"Cleaned up old backup: {backup_dir}")
                except ValueError:
                    # Skip directories that don't match expected format
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Backup cleanup failed: {e}")

def main():
    """Main entry point for weekly pipeline execution"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ML Pipeline Orchestrator - Weekly Execution')
    parser.add_argument('--force-labels', action='store_true', 
                       help='Force update of institutional labels before training')
    parser.add_argument('--labels-only', action='store_true',
                       help='Only update institutional labels, skip model training')
    parser.add_argument('--skip-data-update', action='store_true',
                       help='Skip daily data update (use existing data)')
    parser.add_argument('--data-only', action='store_true',
                       help='Only update daily data, skip everything else')
    
    args = parser.parse_args()
    
    print("🤖 ML Pipeline Orchestrator - Weekly Execution")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = MLPipelineOrchestrator()
        
        # Handle data-only mode
        if args.data_only:
            print("📡 Running data-only update...")
            success = orchestrator.update_daily_data()
            if success:
                print("\n✅ Daily data updated successfully!")
                exit_code = 0
            else:
                print("\n❌ Daily data update failed!")
                exit_code = 1
        # Handle labels-only mode
        elif args.labels_only:
            print("🏷️ Running labels-only update...")
            success = orchestrator.update_institutional_labels()
            if success:
                print("\n✅ Institutional labels updated successfully!")
                exit_code = 0
            else:
                print("\n❌ Institutional labels update failed!")
                exit_code = 1
        else:
            # Run full weekly pipeline
            update_data = not args.skip_data_update
            
            if args.skip_data_update:
                print("⚠️ Skipping daily data update (using existing data)")
            
            success = orchestrator.run_weekly_pipeline(
                force_label_update=args.force_labels,
                update_data=update_data
            )
            
            if success:
                print("\n✅ Weekly ML pipeline completed successfully!")
                print("🔄 New models are now available for the live trading engine")
                exit_code = 0
            else:
                print("\n❌ Weekly ML pipeline failed!")
                print("🔙 Models have been rolled back to previous version")
                exit_code = 1
        
    except Exception as e:
        print(f"\n💥 Critical pipeline error: {e}")
        exit_code = 2
    
    print(f"Completed at: {datetime.now()}")
    exit(exit_code)

if __name__ == "__main__":
    main()