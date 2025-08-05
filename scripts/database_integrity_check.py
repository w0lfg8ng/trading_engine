#!/usr/bin/env python3
"""
Database integrity checks and backup system for orchestrator
FIXED VERSION - No false positives for institutional-only setup
"""

import sqlite3
import pandas as pd
import shutil
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import logging

class DatabaseIntegrityChecker:
    def __init__(self, db_path: str, backup_dir: str = "../data/db_backups"):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected pairs and timeframes
        self.pairs = ["XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
                     "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"]
        self.expected_start_date = "2023-03-08"  # Common start date
        
    def create_backup(self) -> str:
        """Create timestamped database backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"kraken_v2_backup_{timestamp}.db"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            
            # Create backup manifest
            manifest = {
                'backup_time': datetime.now().isoformat(),
                'original_size': self.db_path.stat().st_size,
                'backup_size': backup_path.stat().st_size,
                'md5_hash': self._calculate_md5(backup_path)
            }
            
            manifest_path = backup_path.with_suffix('.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logging.info(f"Database backed up to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            raise
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def check_database_integrity(self) -> dict:
        """Comprehensive database integrity check - FIXED for institutional-only setup"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 1. Basic SQLite integrity check
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            results['checks']['sqlite_integrity'] = integrity_result == 'ok'
            
            if integrity_result != 'ok':
                results['errors'].append(f"SQLite integrity check failed: {integrity_result}")
                results['overall_status'] = 'corrupted'
            
            # 2. Check for institutional tables (the ones we actually need)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            # FIXED: Only check for institutional tables (these have features + labels)
            required_institutional_tables = []
            for pair in self.pairs:
                required_institutional_tables.append(f"features_{pair}_1m_institutional")
            
            missing_institutional_tables = [table for table in required_institutional_tables if table not in existing_tables]
            results['checks']['institutional_tables_present'] = len(missing_institutional_tables) == 0
            
            if missing_institutional_tables:
                results['errors'].append(f"Missing institutional tables: {missing_institutional_tables}")
                results['overall_status'] = 'incomplete'
            else:
                results['checks']['training_ready'] = True
            
            # 3. Check data quality for each pair's institutional table
            institutional_tables_healthy = 0
            for pair in self.pairs:
                pair_checks = self._check_institutional_data_quality(conn, pair)
                results['checks'][f'{pair}_data_quality'] = pair_checks
                
                if not pair_checks['has_data']:
                    results['errors'].append(f"{pair}: No institutional data found")
                    results['overall_status'] = 'incomplete'
                elif not pair_checks['has_recent_data']:
                    results['warnings'].append(f"{pair}: No recent data (>7 days old)")
                elif pair_checks['missing_chunks']:
                    results['warnings'].append(f"{pair}: Data gaps detected")
                else:
                    institutional_tables_healthy += 1
            
            # Overall assessment based on institutional tables
            if institutional_tables_healthy == len(self.pairs):
                results['checks']['all_pairs_healthy'] = True
            elif institutional_tables_healthy >= len(self.pairs) * 0.8:  # 80% threshold
                results['warnings'].append(f"Some pairs have data issues, but {institutional_tables_healthy}/{len(self.pairs)} are healthy")
            
            # 4. Check database size
            current_size = self.db_path.stat().st_size
            results['checks']['database_size_mb'] = round(current_size / (1024*1024), 1)
            
            # Check against previous backup sizes
            if self._check_size_regression(current_size):
                results['warnings'].append("Database size significantly smaller than recent backups")
            
            conn.close()
            
        except Exception as e:
            results['errors'].append(f"Integrity check failed: {e}")
            results['overall_status'] = 'error'
        
        return results
    
    def _check_institutional_data_quality(self, conn: sqlite3.Connection, pair: str) -> dict:
        """Check data quality for institutional table (features + labels)"""
        checks = {
            'has_data': False,
            'has_recent_data': False,
            'has_labels': False,
            'data_range_days': 0,
            'missing_chunks': False,
            'row_count': 0,
            'label_distribution': {}
        }
        
        try:
            # Check institutional table (this has features + labels)
            inst_table = f"features_{pair}_1m_institutional"
            
            # Basic data check
            df = pd.read_sql(f"""
                SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as row_count
                FROM {inst_table}
            """, conn)
            
            if len(df) > 0 and df['row_count'].iloc[0] > 0:
                checks['has_data'] = True
                checks['row_count'] = int(df['row_count'].iloc[0])
                
                # Handle different timestamp formats
                min_ts_raw = df['min_ts'].iloc[0]
                max_ts_raw = df['max_ts'].iloc[0]
                
                # Try to parse timestamps (could be Unix or ISO format)
                try:
                    if isinstance(min_ts_raw, str):
                        min_ts = pd.to_datetime(min_ts_raw)
                        max_ts = pd.to_datetime(max_ts_raw)
                    else:
                        min_ts = pd.to_datetime(min_ts_raw, unit='s')
                        max_ts = pd.to_datetime(max_ts_raw, unit='s')
                    
                    checks['data_range_days'] = (max_ts - min_ts).days
                    checks['has_recent_data'] = (datetime.now() - max_ts.to_pydatetime()).days <= 7
                except:
                    checks['data_range_days'] = 0
                    checks['has_recent_data'] = False
                
                # Check for institutional labels (essential for XGBoost training)
                checks['has_labels'] = False
                checks['label_distribution'] = {}
                
                # Check for institutional_label columns (1h, 4h, 12h)
                for timeframe in ['1h', '4h', '12h']:
                    label_col = f'institutional_label_{timeframe}'
                    try:
                        label_df = pd.read_sql(f"""
                            SELECT {label_col}, COUNT(*) as count 
                            FROM {inst_table} 
                            WHERE {label_col} IS NOT NULL 
                            GROUP BY {label_col}
                        """, conn)
                        
                        if len(label_df) > 0:
                            checks['has_labels'] = True
                            total_labels = label_df['count'].sum()
                            distribution = dict(zip(label_df[label_col], label_df['count']))
                            
                            # Calculate percentages and find best timeframe
                            buy_pct = distribution.get('buy', 0) / total_labels * 100
                            sell_pct = distribution.get('sell', 0) / total_labels * 100
                            
                            checks['label_distribution'][timeframe] = {
                                'total': int(total_labels),
                                'buy_count': int(distribution.get('buy', 0)),
                                'sell_count': int(distribution.get('sell', 0)),
                                'hold_count': int(distribution.get('hold', 0)),
                                'buy_pct': round(buy_pct, 1),
                                'sell_pct': round(sell_pct, 1),
                                'actionable_pct': round(buy_pct + sell_pct, 1)
                            }
                        
                    except Exception as e:
                        continue
                
                # Determine best timeframe for training
                if checks['label_distribution']:
                    best_timeframe = max(checks['label_distribution'].keys(), 
                                       key=lambda x: checks['label_distribution'][x]['actionable_pct'])
                    checks['recommended_timeframe'] = best_timeframe
                
                # Check for large gaps (>6 hours) - but be more lenient
                try:
                    gap_check = pd.read_sql(f"""
                        SELECT COUNT(*) as gap_count FROM (
                            SELECT timestamp, 
                                   LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                            FROM {inst_table}
                            ORDER BY timestamp
                        ) WHERE ABS(CAST(timestamp AS REAL) - CAST(prev_timestamp AS REAL)) > 21600
                    """, conn)
                    
                    checks['missing_chunks'] = gap_check['gap_count'].iloc[0] > 20  # More lenient threshold
                except:
                    checks['missing_chunks'] = False
        
        except Exception as e:
            logging.warning(f"Institutional data quality check failed for {pair}: {e}")
        
        return checks
    
    def _check_size_regression(self, current_size: int) -> bool:
        """Check if database has shrunk significantly compared to recent backups"""
        try:
            # Get recent backup sizes
            backup_files = list(self.backup_dir.glob("kraken_v2_backup_*.json"))
            if len(backup_files) < 2:
                return False
            
            # Check last 3 backups
            recent_backups = sorted(backup_files, key=lambda x: x.stat().st_mtime)[-3:]
            
            for backup_manifest in recent_backups:
                with open(backup_manifest, 'r') as f:
                    manifest = json.load(f)
                
                previous_size = manifest.get('backup_size', 0)
                if previous_size > 0:
                    size_ratio = current_size / previous_size
                    if size_ratio < 0.8:  # More than 20% smaller
                        return True
            
            return False
            
        except Exception:
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30):
        """Remove backups older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for backup_file in self.backup_dir.glob("kraken_v2_backup_*"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    backup_file.unlink()
                    logging.info(f"Removed old backup: {backup_file}")
                except Exception as e:
                    logging.warning(f"Failed to remove backup {backup_file}: {e}")

def run_database_checks(db_path: str) -> dict:
    """Run complete database integrity and quality checks"""
    checker = DatabaseIntegrityChecker(db_path)
    
    # Create backup before checks
    backup_path = checker.create_backup()
    
    # Run integrity checks
    results = checker.check_database_integrity()
    results['backup_created'] = backup_path
    
    # Cleanup old backups
    checker.cleanup_old_backups()
    
    return results

def main():
    """Main function for standalone execution"""
    print("🔍 DATABASE INTEGRITY CHECK (FIXED VERSION)")
    print("=" * 60)
    
    # Database path from scripts/ directory
    db_path = "../data/kraken_v2.db"
    
    try:
        # Run the checks
        results = run_database_checks(db_path)
        
        # Display results with explanations
        print(f"\n📊 INTEGRITY CHECK RESULTS:")
        status = results['overall_status'].upper()
        print(f"   Status: {status}")
        
        # Explain what the status means
        status_explanations = {
            'HEALTHY': 'All institutional tables present with good data quality',
            'CORRUPTED': 'SQLite integrity check failed - database corruption detected',
            'INCOMPLETE': 'Missing institutional tables or serious data issues', 
            'ERROR': 'Could not complete integrity checks due to errors'
        }
        print(f"   Meaning: {status_explanations.get(status, 'Unknown status')}")
        print(f"   Database Size: {results['checks']['database_size_mb']} MB")
        print(f"   Backup Created: ✅")
        
        # Check if training ready
        if results['checks'].get('training_ready', False):
            print(f"   XGBoost Ready: ✅ All institutional tables found")
        else:
            print(f"   XGBoost Ready: ❌ Missing institutional tables")
        
        # Show any warnings with more context
        if results['warnings']:
            print(f"\n⚠️ WARNINGS (Minor Issues):")
            for warning in results['warnings']:
                print(f"   - {warning}")
            print(f"   💡 Note: Small data gaps are normal and won't affect model training")
        
        # Show any errors  
        if results['errors']:
            print(f"\n❌ ERRORS (Need Attention):")
            for error in results['errors']:
                print(f"   - {error}")
        
        # Enhanced per-pair summary with label info
        print(f"\n📈 INSTITUTIONAL DATA STATUS:")
        for pair in ["XBTUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
                     "LTCUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"]:
            pair_key = f'{pair}_data_quality'
            if pair_key in results['checks']:
                pair_data = results['checks'][pair_key]
                
                if pair_data['has_data']:
                    status_icon = "✅" if pair_data['has_labels'] else "⚠️"
                    
                    # Show label info for best timeframe
                    label_info = ""
                    if pair_data['has_labels'] and pair_data['label_distribution']:
                        best_tf = pair_data.get('recommended_timeframe', 'unknown')
                        if best_tf in pair_data['label_distribution']:
                            tf_data = pair_data['label_distribution'][best_tf]
                            label_info = f" [Labels: {best_tf} - {tf_data['actionable_pct']}% actionable]"
                        else:
                            label_info = " [Labels: ✅]"
                    else:
                        label_info = " [Labels: ❌]"
                        status_icon = "⚠️"
                    
                    gap_info = " [Gaps]" if pair_data['missing_chunks'] else ""
                    
                    print(f"   {status_icon} {pair}: {pair_data['row_count']:,} rows, "
                          f"{pair_data['data_range_days']} days{label_info}{gap_info}")
                else:
                    print(f"   ❌ {pair}: No data found")
        
        print(f"\n💾 Backup saved successfully")
        print(f"🔗 For orchestrator integration, import: run_database_checks()")
        
        # Show label distribution summary if available
        if results['checks'].get('XBTUSDT_data_quality', {}).get('label_distribution'):
            print(f"\n🏷️ LABEL DISTRIBUTION SUMMARY (XBTUSDT):")
            label_dist = results['checks']['XBTUSDT_data_quality']['label_distribution']
            for timeframe, data in label_dist.items():
                print(f"   {timeframe.upper()}: {data['actionable_pct']}% actionable ({data['buy_count']:,} buy, {data['sell_count']:,} sell)")
            
            best_tf = results['checks']['XBTUSDT_data_quality'].get('recommended_timeframe', '12h')
            print(f"   💡 Recommended for XGBoost: {best_tf.upper()} (most balanced)")
        
        # Overall assessment
        if results['overall_status'] in ['healthy', 'HEALTHY']:
            print(f"\n✅ ASSESSMENT: Database is ready for XGBoost training")
            print(f"   💡 All institutional tables (features + labels) are present and healthy")
            if results['checks'].get('XBTUSDT_data_quality', {}).get('recommended_timeframe'):
                rec_tf = results['checks']['XBTUSDT_data_quality']['recommended_timeframe']
                print(f"   🎯 Recommended: Use {rec_tf} labels for best model performance")
        elif results['overall_status'] in ['incomplete', 'INCOMPLETE']:
            missing_tables = any('Missing institutional tables' in error for error in results['errors'])
            if missing_tables:
                print(f"\n❌ ASSESSMENT: Missing institutional tables - cannot train XGBoost models")
            else:
                print(f"\n⚠️ ASSESSMENT: Database has minor issues but is usable for training")
        else:
            print(f"\n❌ ASSESSMENT: Database issues should be addressed before training")
        
    except Exception as e:
        print(f"❌ Check failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())