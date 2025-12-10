# cython: language_level=3

"""
License Management for LAM (Compiled)
======================================

Handles license validation for Pro/Enterprise tiers.
Cross-platform support for Windows, Linux, and macOS.
Compiled for performance and security.
"""

import json
import os
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class LicenseManager:
    """Manages license validation and tier detection."""
    
    LICENSE_ENV_VAR = "LAM_LICENSE_KEY"
    DEFAULT_LIMIT = 0x2000       # 8K for free tier
    PRO_LIMIT = 0x1E8480         # 2M for pro/enterprise (2,000,000 tokens) - infinite scaling
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize license manager."""
        self.model_path = model_path
        self.license_data: Optional[Dict[str, Any]] = None
        self.tier: str = "free"
        self.max_length: int = self.DEFAULT_LIMIT
        self.license_path: Optional[Path] = None
        self._load_license()
    
    def _get_license_locations(self) -> List[Path]:
        """Get license file location - ONLY checks lam/lam_license.json in package directory."""
        locations = []
        
        # ONLY check the package directory: lam/lam_license.json
        try:
            import lam
            if hasattr(lam, '__file__'):
                lam_module_path = Path(lam.__file__).parent
                # Only check lam/lam_license.json in the package directory
                locations.append(lam_module_path / "lam_license.json")
        except:
            pass
        
        return locations
    
    def _load_license(self) -> None:
        """Load and validate license - ONLY from lam/lam_license.json in package directory."""
        license_locations = self._get_license_locations()
        
        # Only check the package directory location
        for license_path in license_locations:
            if license_path.exists():
                if self._validate_license_file(license_path):
                    self.license_path = license_path
                    return
        
        # No license found - use free tier
        self.license_data = None
        self.tier = "free"
        self.max_length = self.DEFAULT_LIMIT
        self.license_path = None
    
    def _validate_license_file(self, license_path: Path) -> bool:
        """Validate a license file."""
        try:
            with open(license_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                return False
            
            if "license_key" not in data or "tier" not in data:
                return False
            
            license_key = data["license_key"]
            if not self._validate_license_key(license_key):
                return False
            
            if "expires_at" in data:
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.now() > expires_at:
                    return False
            
            self.license_data = data
            self.tier = data.get("tier", "pro").lower()
            
            if self.tier == "enterprise":
                self.max_length = self.PRO_LIMIT
            elif self.tier == "pro":
                self.max_length = self.PRO_LIMIT
            else:
                self.max_length = self.DEFAULT_LIMIT
            
            return True
            
        except Exception:
            return False
    
    def _validate_license_key(self, license_key: str) -> bool:
        """
        Validate a license key.
        
        Supports industry-standard formats:
        - lam_dev_<base64> (developer/master keys) - 51 chars
        - lam_pro_<base64> (production keys)
        - lam_ent_<base64> (enterprise keys)
        - LAM-<format> (legacy format)
        """
        if not license_key or len(license_key) < 20:
            return False
        
        # Support multiple key formats:
        # - lam_dev_<base64> (developer/master keys) - 51 chars
        # - lam_pro_<base64> (production keys)
        # - lam_ent_<base64> (enterprise keys)
        if license_key.startswith("lam_"):
            parts = license_key.split("_")
            if len(parts) >= 3:
                # Validate base64-like suffix (at least 32 chars for security)
                suffix = parts[-1]
                if len(suffix) >= 32 and len(license_key) >= 45:
                    return True
        
        # Legacy format support
        if license_key.startswith("LAM-"):
            parts = license_key.split("-")
            if len(parts) == 5:
                return True
        
        return False
    
    def is_licensed(self) -> bool:
        """Check if a valid license is present."""
        return self.license_data is not None
    
    def get_tier(self) -> str:
        """Get current license tier."""
        return self.tier
    
    def get_max_length(self) -> int:
        """Get maximum sequence length for current tier."""
        return self.max_length
    
    def get_license_info(self) -> Optional[Dict[str, Any]]:
        """Get license information (without sensitive data)."""
        if not self.license_data:
            return None
        
        return {
            "tier": self.tier,
            "max_length": self.max_length,
            "expires_at": self.license_data.get("expires_at"),
            "customer": self.license_data.get("customer", "Unknown"),
        }


def get_license_locations_info() -> Dict[str, Any]:
    """Get information about where license files are checked."""
    manager = LicenseManager()
    locations = manager._get_license_locations()
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "license_locations": [str(loc) for loc in locations],
        "environment_variable": LicenseManager.LICENSE_ENV_VAR,
    }
