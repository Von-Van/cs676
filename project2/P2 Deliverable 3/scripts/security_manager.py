#!/usr/bin/env python3
"""
security_manager.py - Security, Authentication, and Access Control

Provides security features including:
- User authentication with session management
- Rate limiting and abuse prevention
- Input validation and sanitization
- API key management
- Audit logging
"""

import os
import hashlib
import secrets
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_authentication: bool = False
    session_timeout: int = 3600  # 1 hour
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    rate_limit_per_minute: int = 60
    min_password_length: int = 8
    require_strong_password: bool = True
    enable_audit_log: bool = True
    audit_log_path: str = "logs/audit.log"


class SecurityManager:
    """Manages security, authentication, and access control."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limit_tracker = defaultdict(list)
        
        # Failed login tracking
        self.failed_attempts = defaultdict(int)
        self.lockout_until = {}
        
        # Audit log
        if self.config.enable_audit_log:
            self._setup_audit_log()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure security logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_audit_log(self):
        """Set up audit logging to file."""
        log_path = Path(self.config.audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        audit_handler = logging.FileHandler(log_path)
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        ))
        self.logger.addHandler(audit_handler)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{pwd_hash}"
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, pwd_hash = stored_hash.split(':')
            test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return test_hash == pwd_hash
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password meets security requirements."""
        if len(password) < self.config.min_password_length:
            return False, f"Password must be at least {self.config.min_password_length} characters"
        
        if self.config.require_strong_password:
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
            
            if not (has_upper and has_lower and has_digit and has_special):
                return False, "Password must contain uppercase, lowercase, number, and special character"
        
        return True, "Password is strong"
    
    def authenticate(self, username: str, password: str, stored_hash: str) -> Optional[str]:
        """Authenticate user and return session token if successful."""
        # Check if account is locked out
        if username in self.lockout_until:
            if time.time() < self.lockout_until[username]:
                remaining = int(self.lockout_until[username] - time.time())
                self.logger.warning(f"Login attempt for locked account: {username}")
                self._audit_log(f"LOGIN_FAILED_LOCKED: {username}")
                return None
            else:
                # Lockout expired
                del self.lockout_until[username]
                self.failed_attempts[username] = 0
        
        # Verify password
        if self.verify_password(password, stored_hash):
            # Success - reset failed attempts and create session
            self.failed_attempts[username] = 0
            session_token = self.create_session(username)
            self.logger.info(f"Successful login: {username}")
            self._audit_log(f"LOGIN_SUCCESS: {username}")
            return session_token
        else:
            # Failed attempt
            self.failed_attempts[username] += 1
            self.logger.warning(f"Failed login attempt for {username} (attempt {self.failed_attempts[username]})")
            self._audit_log(f"LOGIN_FAILED: {username}")
            
            # Check if should lock out
            if self.failed_attempts[username] >= self.config.max_failed_attempts:
                self.lockout_until[username] = time.time() + self.config.lockout_duration
                self.logger.error(f"Account locked due to failed attempts: {username}")
                self._audit_log(f"ACCOUNT_LOCKED: {username}")
            
            return None
    
    def create_session(self, username: str) -> str:
        """Create a new session and return session token."""
        session_token = secrets.token_urlsafe(32)
        self.active_sessions[session_token] = {
            'username': username,
            'created_at': time.time(),
            'last_activity': time.time(),
            'ip_address': None  # Can be set by caller
        }
        return session_token
    
    def validate_session(self, session_token: str) -> bool:
        """Check if session token is valid and not expired."""
        if session_token not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_token]
        
        # Check if session has expired
        if time.time() - session['last_activity'] > self.config.session_timeout:
            self.logger.info(f"Session expired for {session['username']}")
            del self.active_sessions[session_token]
            return False
        
        # Update last activity
        session['last_activity'] = time.time()
        return True
    
    def invalidate_session(self, session_token: str):
        """Invalidate a session (logout)."""
        if session_token in self.active_sessions:
            username = self.active_sessions[session_token]['username']
            del self.active_sessions[session_token]
            self.logger.info(f"Session invalidated for {username}")
            self._audit_log(f"LOGOUT: {username}")
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier (user/IP) has exceeded rate limit."""
        current_time = time.time()
        
        # Clean up old entries
        self.rate_limit_tracker[identifier] = [
            t for t in self.rate_limit_tracker[identifier]
            if current_time - t < 60  # Keep last minute
        ]
        
        # Check rate limit
        if len(self.rate_limit_tracker[identifier]) >= self.config.rate_limit_per_minute:
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            self._audit_log(f"RATE_LIMIT_EXCEEDED: {identifier}")
            return False
        
        # Record this request
        self.rate_limit_tracker[identifier].append(current_time)
        return True
    
    def sanitize_input(self, user_input: str, max_length: int = 10000) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Truncate to max length
        sanitized = user_input[:max_length]
        
        # Remove potentially dangerous characters/patterns
        # Note: This is basic sanitization. Consider using libraries like bleach for HTML
        dangerous_patterns = ['<script>', 'javascript:', 'onerror=', 'onclick=']
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format (basic check)."""
        if not api_key:
            return False
        
        # OpenAI API keys start with 'sk-' and are 51 chars
        if api_key.startswith('sk-') and len(api_key) >= 40:
            return True
        
        return False
    
    def _audit_log(self, message: str):
        """Write to audit log."""
        if self.config.enable_audit_log:
            self.logger.info(message)
    
    def get_session_info(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        return self.active_sessions.get(session_token)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired = [
            token for token, session in self.active_sessions.items()
            if current_time - session['last_activity'] > self.config.session_timeout
        ]
        
        for token in expired:
            username = self.active_sessions[token]['username']
            del self.active_sessions[token]
            self.logger.info(f"Cleaned up expired session for {username}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        return {
            'active_sessions': len(self.active_sessions),
            'locked_accounts': len(self.lockout_until),
            'rate_limited_ips': len(self.rate_limit_tracker),
            'failed_login_attempts': sum(self.failed_attempts.values()),
        }
