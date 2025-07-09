# data_quality_check.py
"""
Quality check for enhanced feature tables
Validates data integrity before model training
"""

import pandas as pd
import sqlite3
import numpy as np
from typing import Dict, List
import logging

class FeatureQualityChecker:
    def __init__(self, db_path: str = "../../data/kraken_v2.db"):
        self.db_path = db_path
        self.conn = None
        
        # Expected pairs and timeframes
        self.pairs = [
            "ADAUSDT", "AVAXUSDT", "DOTUSDT", "ETHUSDT", 
            "LINKUSDT", "LTCUSDT", "SOLUSDT", "XBTUSDT", "XRPUSDT"
        ]
        self.timeframes = ['1m', '5m']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect_database(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.logger.info(f"Connected to {self.db_path}")
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a feature table exists"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        return cursor.fetchone() is not None
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get basic table information"""
        try:
            # Get row count
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get column count and names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Get date range
            cursor.execute(f"""
                SELECT MIN(datetime), MAX(datetime) 
                FROM {table_name}
            """)
            date_range = cursor.fetchone()
            
            return {
                'exists': True,
                'row_count': row_count,
                'column_count': len(column_names),
                'columns': column_names,
                'start_date': date_range[0],
                'end_date': date_range[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting info for {table_name}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def check_feature_quality(self, table_name: str) -> Dict:
        """Check quality of features in a table"""
        try:
            # Load a sample of data
            df = pd.read_sql(f"""
                SELECT * FROM {table_name} 
                ORDER BY timestamp 
                LIMIT 10000
            """, self.conn)
            
            if df.empty:
                return {'error': 'Table is empty'}
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            nan_columns = nan_counts[nan_counts > 0].to_dict()
            
            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_counts = {}
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
            
            # Check feature ranges (look for unrealistic values)
            feature_stats = {}
            key_features = ['rsi_14', 'macd', 'ema_12', 'bb_upper', 'volume_spike']
            
            for feature in key_features:
                if feature in df.columns:
                    feature_stats[feature] = {
                        'min': df[feature].min(),
                        'max': df[feature].max(),
                        'mean': df[feature].mean(),
                        'std': df[feature].std()
                    }
            
            # Check data continuity (no large time gaps)
            df['timestamp_diff'] = df['timestamp'].diff()
            large_gaps = (df['timestamp_diff'] > 300).sum()  # > 5 minutes gap
            
            return {
                'sample_size': len(df),
                'nan_columns': nan_columns,
                'inf_columns': inf_counts,
                'feature_stats': feature_stats,
                'large_time_gaps': large_gaps,
                'data_quality': 'Good' if len(nan_columns) == 0 and len(inf_counts) == 0 else 'Issues Found'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking quality for {table_name}: {e}")
            return {'error': str(e)}
    
    def verify_feature_calculations(self, table_name: str) -> Dict:
        """Verify that feature calculations make sense"""
        try:
            # Get recent data to verify calculations
            df = pd.read_sql(f"""
                SELECT timestamp, close, rsi_14, ema_12, ema_26, 
                       bb_upper, bb_lower, volume, volume_spike
                FROM {table_name} 
                ORDER BY timestamp DESC 
                LIMIT 100
            """, self.conn)
            
            issues = []
            
            # Check RSI is in valid range (0-100)
            if 'rsi_14' in df.columns:
                rsi_out_of_range = ((df['rsi_14'] < 0) | (df['rsi_14'] > 100)).sum()
                if rsi_out_of_range > 0:
                    issues.append(f"RSI out of range: {rsi_out_of_range} instances")
            
            # Check EMA ordering (EMA12 should respond faster than EMA26)
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                # In trending markets, EMAs should be relatively close to price
                ema12_deviation = abs((df['ema_12'] - df['close']) / df['close']).mean()
                ema26_deviation = abs((df['ema_26'] - df['close']) / df['close']).mean()
                
                if ema12_deviation > 0.1:  # EMA12 more than 10% away from price on average
                    issues.append(f"EMA12 deviation seems high: {ema12_deviation:.2%}")
            
            # Check Bollinger Bands logic (upper > lower)
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_inverted = (df['bb_upper'] <= df['bb_lower']).sum()
                if bb_inverted > 0:
                    issues.append(f"Bollinger Bands inverted: {bb_inverted} instances")
            
            # Check volume spike logic (should be binary 0/1)
            if 'volume_spike' in df.columns:
                non_binary = (~df['volume_spike'].isin([0, 1])).sum()
                if non_binary > 0:
                    issues.append(f"Volume spike not binary: {non_binary} instances")
            
            return {
                'issues_found': len(issues),
                'issues': issues,
                'validation_status': 'PASS' if len(issues) == 0 else 'ISSUES_DETECTED'
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying {table_name}: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_check(self):
        """Run comprehensive quality check on all feature tables"""
        
        self.logger.info("Starting comprehensive feature quality check...")
        
        results = {
            'table_status': {},
            'quality_summary': {},
            'validation_summary': {},
            'overall_status': 'UNKNOWN'
        }
        
        total_tables = len(self.pairs) * len(self.timeframes)
        successful_tables = 0
        
        for timeframe in self.timeframes:
            self.logger.info(f"\n🔍 CHECKING {timeframe.upper()} FEATURE TABLES...")
            
            for pair in self.pairs:
                table_name = f"features_{pair.lower()}_{timeframe}"
                
                # Check table exists and get basic info
                table_info = self.get_table_info(table_name)
                
                if table_info['exists']:
                    # Check feature quality
                    quality_check = self.check_feature_quality(table_name)
                    
                    # Verify calculations
                    validation_check = self.verify_feature_calculations(table_name)
                    
                    # Store results
                    results['table_status'][table_name] = table_info
                    results['quality_summary'][table_name] = quality_check
                    results['validation_summary'][table_name] = validation_check
                    
                    # Log results
                    status = "✅" if quality_check.get('data_quality') == 'Good' and validation_check.get('validation_status') == 'PASS' else "⚠️"
                    row_count = table_info['row_count']
                    col_count = table_info['column_count']
                    
                    self.logger.info(f"  {status} {pair}: {row_count:,} rows, {col_count} features")
                    
                    if status == "✅":
                        successful_tables += 1
                    else:
                        # Log issues
                        if 'issues' in validation_check:
                            for issue in validation_check['issues']:
                                self.logger.warning(f"    ⚠️ {issue}")
                
                else:
                    self.logger.error(f"  ❌ {pair}: Table not found")
                    results['table_status'][table_name] = table_info
        
        # Overall status
        if successful_tables == total_tables:
            results['overall_status'] = 'EXCELLENT'
        elif successful_tables >= total_tables * 0.8:
            results['overall_status'] = 'GOOD'
        else:
            results['overall_status'] = 'NEEDS_ATTENTION'
        
        # Summary report
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FEATURE QUALITY SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Tables checked: {total_tables}")
        self.logger.info(f"✅ Successful: {successful_tables}")
        self.logger.info(f"⚠️ Issues: {total_tables - successful_tables}")
        self.logger.info(f"🎯 Overall status: {results['overall_status']}")
        
        # Calculate total feature count
        total_rows = sum(info['row_count'] for info in results['table_status'].values() if info.get('exists'))
        avg_features = np.mean([info['column_count'] for info in results['table_status'].values() if info.get('exists')])
        
        self.logger.info(f"📈 Total feature rows: {total_rows:,}")
        self.logger.info(f"🔢 Average features per table: {avg_features:.0f}")
        
        return results
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("FEATURE QUALITY CHECKER")
    print("=" * 60)
    
    checker = FeatureQualityChecker()
    
    try:
        checker.connect_database()
        results = checker.run_comprehensive_check()
        
        if results['overall_status'] in ['EXCELLENT', 'GOOD']:
            print("\n🎉 Data is ready for model training!")
        else:
            print("\n⚠️ Some issues found - review before training")
        
    except Exception as e:
        print(f"\n❌ Quality check failed: {e}")
        
    finally:
        checker.close_connection()

if __name__ == "__main__":
    main()