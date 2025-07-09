#!/bin/bash
echo "🤖 Setting up automated ML pipeline cron jobs..."

PROJECT_DIR="/mnt/raid0/data_erick/kraken_trading_model_v2"
ORCHESTRATOR_SCRIPT="$PROJECT_DIR/src/live/ml_pipeline_orchestrator.py"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "🔍 Found existing cron job. Analyzing..."
echo "📋 Current crontab:"
crontab -l
echo ""

echo "🔧 Issues found in existing job:"
echo "  - Syntax error (missing space)"
echo "  - Uses old script instead of orchestrator"
echo "  - No model training automation"
echo ""

echo "💡 Recommended: Replace with orchestrator system"
read -p "🤔 Replace existing job with new orchestrator system? (y/N): " confirm

case $confirm in
    [Yy]*)
        # Create backup
        crontab -l > "crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
        echo "💾 Backup created: crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
        
        # Remove existing ML-related jobs and add new ones
        CRON_JOBS="# Edit this file to introduce tasks to be run by cron.

# ========================================
# AUTOMATED ML PIPELINE CRON JOBS
# ========================================

# Daily data update (every 4 hours) - UPGRADED VERSION
# Uses orchestrator for better logging and error handling
0 */4 * * * cd $PROJECT_DIR && /usr/bin/python3 $ORCHESTRATOR_SCRIPT --data-only >> $LOG_DIR/daily_data_update.log 2>&1

# Weekly full pipeline (Sundays at 2 AM)
# Complete model retraining and deployment
0 2 * * 0 cd $PROJECT_DIR && /usr/bin/python3 $ORCHESTRATOR_SCRIPT --force-labels >> $LOG_DIR/weekly_pipeline.log 2>&1

# Monthly institutional labels update (1st of month at 4 AM)
# Ensures labels stay current with changing market conditions
0 4 1 * * cd $PROJECT_DIR && /usr/bin/python3 $ORCHESTRATOR_SCRIPT --labels-only >> $LOG_DIR/monthly_labels.log 2>&1
# ========================================

# 
# Each task to run has to be defined through a single line
# indicating with different fields when the task will be run
# and what command to run for the task
# 
# To define the time you can provide concrete values for
# minute (m), hour (h), day of month (dom), month (mon),
# and day of week (dow) or use '*' in these fields (for 'any').
# 
# Notice that tasks will be started based on the cron's system
# daemon's notion of time and timezones.
# 
# Output of the crontab jobs (including errors) is sent through
# email to the user the crontab file belongs to (unless redirected).
# 
# For example, you can run a backup of all your user accounts
# at 5 a.m every week with:
# 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
# 
# For more information see the manual pages of crontab(5) and cron(8)
# 
# m h  dom mon dow   command"

        echo "$CRON_JOBS" | crontab -
        
        echo "✅ Cron jobs updated successfully!"
        echo ""
        echo "📊 NEW Schedule:"
        echo "  🔄 Data Updates: Every 4 hours (UPGRADED to orchestrator)"
        echo "  🤖 Full Pipeline: Weekly (Sundays 2 AM) - NEW!"
        echo "  🏷️ Labels Update: Monthly (1st at 4 AM) - NEW!"
        ;;
    *)
        echo "🚫 Cancelled by user"
        exit 0
        ;;
esac

echo ""
echo "📝 View current crontab: crontab -l"
echo "📋 Log locations:"
echo "  Daily Data: $LOG_DIR/daily_data_update.log (NEW improved logging)"
echo "  Weekly Pipeline: $LOG_DIR/weekly_pipeline.log (NEW)"
echo "  Monthly Labels: $LOG_DIR/monthly_labels.log (NEW)"
echo ""
echo "🚀 Your ML pipeline is now fully automated!"
