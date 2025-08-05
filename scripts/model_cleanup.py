#!/usr/bin/env python3
"""
Clean up incomplete model sets and verify model consistency
"""

import os
from datetime import datetime
from pathlib import Path

def cleanup_incomplete_models(model_dir="models/xgboost_regime_specific/"):
    """Remove any incomplete model sets and verify consistency"""
    
    model_dir = Path(model_dir)
    
    # Find all model files
    model_files = list(model_dir.glob("*_xgb_model.pkl"))
    
    print(f"🔍 Checking {len(model_files)} model sets...")
    
    complete_sets = []
    incomplete_sets = []
    files_to_remove = []
    
    for model_file in model_files:
        base_name = model_file.name.replace('_xgb_model.pkl', '')
        encoder_file = model_dir / f"{base_name}_encoder.pkl"
        features_file = model_dir / f"{base_name}_features.pkl"
        
        # Check if all 3 files exist
        if encoder_file.exists() and features_file.exists():
            complete_sets.append(base_name)
            print(f"✅ {base_name}")
        else:
            incomplete_sets.append(base_name)
            print(f"❌ {base_name} (incomplete)")
            
            # Mark incomplete files for removal
            if model_file.exists():
                files_to_remove.append(model_file)
            if encoder_file.exists():
                files_to_remove.append(encoder_file)
            if features_file.exists():
                files_to_remove.append(features_file)
    
    # Remove incomplete model sets
    if files_to_remove:
        print(f"\n🧹 Removing {len(files_to_remove)} files from incomplete model sets...")
        for file_path in files_to_remove:
            print(f"  Removing: {file_path.name}")
            file_path.unlink()
    
    # Check for any orphaned files (encoder/features without model)
    all_files = list(model_dir.glob("*.pkl"))
    orphaned_files = []
    
    for file_path in all_files:
        if file_path.name.endswith('_encoder.pkl') or file_path.name.endswith('_features.pkl'):
            if file_path.name.endswith('_encoder.pkl'):
                base_name = file_path.name.replace('_encoder.pkl', '')
            else:
                base_name = file_path.name.replace('_features.pkl', '')
            
            model_file = model_dir / f"{base_name}_xgb_model.pkl"
            if not model_file.exists():
                orphaned_files.append(file_path)
    
    if orphaned_files:
        print(f"\n🧹 Removing {len(orphaned_files)} orphaned files...")
        for file_path in orphaned_files:
            print(f"  Removing: {file_path.name}")
            file_path.unlink()
    
    # Final count
    final_model_files = list(model_dir.glob("*_xgb_model.pkl"))
    final_complete_sets = []
    
    for model_file in final_model_files:
        base_name = model_file.name.replace('_xgb_model.pkl', '')
        encoder_file = model_dir / f"{base_name}_encoder.pkl"
        features_file = model_dir / f"{base_name}_features.pkl"
        
        if encoder_file.exists() and features_file.exists():
            final_complete_sets.append(base_name)
    
    print(f"\n📊 Final Results:")
    print(f"✅ Complete model sets: {len(final_complete_sets)}")
    print(f"📁 Total files: {len(list(model_dir.glob('*.pkl')))}")
    print(f"🎯 Expected files: {len(final_complete_sets) * 3}")
    
    if len(list(model_dir.glob('*.pkl'))) == len(final_complete_sets) * 3:
        print("🎉 All models are complete and clean!")
    else:
        print("⚠️  File count mismatch - manual review needed")
    
    # Create a manifest of clean models
    manifest = {
        'cleanup_time': datetime.now().isoformat(),
        'complete_model_sets': len(final_complete_sets),
        'total_files': len(list(model_dir.glob('*.pkl'))),
        'model_list': sorted(final_complete_sets)
    }
    
    manifest_path = model_dir / 'clean_models_manifest.json'
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📋 Clean models manifest saved: {manifest_path}")
    
    return final_complete_sets

if __name__ == "__main__":
    complete_models = cleanup_incomplete_models()
    
    print(f"\n🚀 Ready for backtesting with {len(complete_models)} verified models!")
    print("\nModel breakdown:")
    
    # Analyze by regime
    regime_counts = {}
    for model in complete_models:
        parts = model.split('_')
        if len(parts) >= 3:
            regime = '_'.join(parts[1:])  # Everything after pair name
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime}: {count} models")