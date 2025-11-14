#!/usr/bin/env python3
"""
backup_manager.py - Backup and Recovery System

Provides backup and recovery capabilities including:
- Automatic periodic backups
- On-demand backup creation
- Backup rotation and cleanup
- Recovery from backups
- Data integrity verification
"""

import os
import json
import shutil
import hashlib
import logging
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
import time


@dataclass
class BackupConfig:
    """Backup configuration settings."""
    backup_dir: str = "backups"
    data_dir: str = "outputs"
    auto_backup_enabled: bool = True
    backup_interval_minutes: int = 60
    max_backups: int = 10
    max_backup_age_days: int = 30
    include_logs: bool = True
    compress_backups: bool = True
    verify_integrity: bool = True


class BackupManager:
    """Manages backup and recovery operations."""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Create backup directory
        self.backup_path = Path(self.config.backup_dir)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        self.data_path = Path(self.config.data_dir)
        
        # Auto-backup thread
        self.auto_backup_thread = None
        self.stop_auto_backup = threading.Event()
        
        if self.config.auto_backup_enabled:
            self.start_auto_backup()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure backup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - BACKUP - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a backup of all data."""
        try:
            # Generate backup name if not provided
            if not backup_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"backup_{timestamp}"
            
            backup_dir = self.backup_path / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Creating backup: {backup_name}")
            
            # Backup data directory
            files_backed_up = 0
            if self.data_path.exists():
                data_backup = backup_dir / "data"
                shutil.copytree(self.data_path, data_backup, dirs_exist_ok=True)
                files_backed_up += sum(1 for _ in data_backup.rglob('*') if _.is_file())
            
            # Backup logs if enabled
            if self.config.include_logs:
                logs_path = Path("logs")
                if logs_path.exists():
                    logs_backup = backup_dir / "logs"
                    shutil.copytree(logs_path, logs_backup, dirs_exist_ok=True)
            
            # Backup configuration files
            config_files = ["config/config.ini", ".env"]
            for config_file in config_files:
                if Path(config_file).exists():
                    dest = backup_dir / Path(config_file).name
                    shutil.copy2(config_file, dest)
            
            # Create manifest
            manifest = {
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'files_backed_up': files_backed_up,
                'backup_size_bytes': self._get_dir_size(backup_dir),
                'includes_logs': self.config.include_logs,
                'verified': False
            }
            
            # Compress if enabled
            if self.config.compress_backups:
                self.logger.info(f"Compressing backup: {backup_name}")
                archive_path = self._compress_backup(backup_dir)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_dir)
                
                manifest['compressed'] = True
                manifest['archive_path'] = str(archive_path)
                manifest['compressed_size_bytes'] = archive_path.stat().st_size
            else:
                manifest['compressed'] = False
            
            # Verify integrity if enabled
            if self.config.verify_integrity:
                checksum = self._calculate_checksum(backup_dir if not self.config.compress_backups else archive_path)
                manifest['checksum'] = checksum
                manifest['verified'] = True
            
            # Save manifest
            manifest_path = self.backup_path / f"{backup_name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Backup created successfully: {backup_name}")
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            return {
                'success': True,
                'backup_name': backup_name,
                'manifest': manifest
            }
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def restore_backup(self, backup_name: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """Restore from a backup."""
        try:
            self.logger.info(f"Restoring backup: {backup_name}")
            
            # Load manifest
            manifest_path = self.backup_path / f"{backup_name}_manifest.json"
            if not manifest_path.exists():
                return {
                    'success': False,
                    'error': f"Backup manifest not found: {backup_name}"
                }
            
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Verify integrity if checksum exists
            if manifest.get('verified') and manifest.get('checksum'):
                backup_path = Path(manifest.get('archive_path', self.backup_path / backup_name))
                current_checksum = self._calculate_checksum(backup_path)
                
                if current_checksum != manifest['checksum']:
                    self.logger.error(f"Backup integrity check failed for {backup_name}")
                    return {
                        'success': False,
                        'error': 'Backup integrity verification failed'
                    }
            
            # Determine source and target
            if manifest.get('compressed'):
                archive_path = Path(manifest['archive_path'])
                temp_extract = self.backup_path / f"{backup_name}_temp"
                self._extract_backup(archive_path, temp_extract)
                source_dir = temp_extract
            else:
                source_dir = self.backup_path / backup_name
            
            target_dir = Path(target_dir) if target_dir else self.data_path
            
            # Restore data
            if (source_dir / "data").exists():
                if target_dir.exists():
                    # Create a safety backup before restoring
                    safety_backup = self.backup_path / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copytree(target_dir, safety_backup, dirs_exist_ok=True)
                    self.logger.info(f"Created safety backup before restore: {safety_backup.name}")
                
                # Clear target and restore
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(source_dir / "data", target_dir)
            
            # Clean up temp extraction
            if manifest.get('compressed') and source_dir.exists():
                shutil.rmtree(source_dir)
            
            self.logger.info(f"Backup restored successfully: {backup_name}")
            
            return {
                'success': True,
                'backup_name': backup_name,
                'files_restored': manifest.get('files_backed_up', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for manifest_file in self.backup_path.glob("*_manifest.json"):
            try:
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    backups.append(manifest)
            except Exception as e:
                self.logger.warning(f"Could not read manifest {manifest_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return backups
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        try:
            backups = self.list_backups()
            
            # Remove backups exceeding max count
            if len(backups) > self.config.max_backups:
                backups_to_remove = backups[self.config.max_backups:]
                for backup in backups_to_remove:
                    self._delete_backup(backup['backup_name'])
                    self.logger.info(f"Removed backup (max count): {backup['backup_name']}")
            
            # Remove backups older than max age
            cutoff_date = datetime.now() - timedelta(days=self.config.max_backup_age_days)
            
            for backup in backups:
                backup_date = datetime.fromisoformat(backup['timestamp'])
                if backup_date < cutoff_date:
                    self._delete_backup(backup['backup_name'])
                    self.logger.info(f"Removed backup (too old): {backup['backup_name']}")
                    
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def _delete_backup(self, backup_name: str):
        """Delete a specific backup."""
        # Delete manifest
        manifest_path = self.backup_path / f"{backup_name}_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        
        # Delete backup data (compressed or uncompressed)
        backup_dir = self.backup_path / backup_name
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        archive_path = self.backup_path / f"{backup_name}.zip"
        if archive_path.exists():
            archive_path.unlink()
    
    def _compress_backup(self, backup_dir: Path) -> Path:
        """Compress backup directory into a zip file."""
        archive_path = self.backup_path / f"{backup_dir.name}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in backup_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(backup_dir)
                    zipf.write(file_path, arcname)
        
        return archive_path
    
    def _extract_backup(self, archive_path: Path, extract_dir: Path):
        """Extract a compressed backup."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(extract_dir)
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA-256 checksum of a file or directory."""
        hasher = hashlib.sha256()
        
        if path.is_file():
            with open(path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
        else:
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    def start_auto_backup(self):
        """Start automatic backup thread."""
        if self.auto_backup_thread and self.auto_backup_thread.is_alive():
            self.logger.warning("Auto-backup already running")
            return
        
        self.stop_auto_backup.clear()
        self.auto_backup_thread = threading.Thread(target=self._auto_backup_loop, daemon=True)
        self.auto_backup_thread.start()
        self.logger.info(f"Auto-backup started (interval: {self.config.backup_interval_minutes} minutes)")
    
    def stop_auto_backup_thread(self):
        """Stop automatic backup thread."""
        if self.auto_backup_thread:
            self.stop_auto_backup.set()
            self.auto_backup_thread.join(timeout=5)
            self.logger.info("Auto-backup stopped")
    
    def _auto_backup_loop(self):
        """Background thread for automatic backups."""
        while not self.stop_auto_backup.is_set():
            try:
                # Wait for interval
                if self.stop_auto_backup.wait(timeout=self.config.backup_interval_minutes * 60):
                    break  # Stop signal received
                
                # Create backup
                self.logger.info("Starting automatic backup")
                result = self.create_backup()
                
                if result['success']:
                    self.logger.info("Automatic backup completed successfully")
                else:
                    self.logger.error(f"Automatic backup failed: {result.get('error')}")
                    
            except Exception as e:
                self.logger.error(f"Auto-backup loop error: {e}")
    
    def get_backup_metrics(self) -> Dict[str, Any]:
        """Get backup metrics for monitoring."""
        backups = self.list_backups()
        
        total_size = sum(
            backup.get('compressed_size_bytes', backup.get('backup_size_bytes', 0))
            for backup in backups
        )
        
        return {
            'total_backups': len(backups),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_backup': backups[-1]['timestamp'] if backups else None,
            'newest_backup': backups[0]['timestamp'] if backups else None,
            'auto_backup_enabled': self.config.auto_backup_enabled,
        }
