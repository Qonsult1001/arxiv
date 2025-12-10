# LAM License System Guide

## Overview

LAM uses a license-based tier system:
- **Free Tier**: Up to 4,096 tokens (no license required)
- **Pro Tier**: Up to 32,768 tokens (requires license)
- **Enterprise Tier**: Up to 32,768 tokens + support (requires license)

## How Users Obtain a License

### Step 1: Purchase License

Users purchase a license through:
- **Website**: https://saidhome.ai
- **API**: Contact sales for enterprise licenses
- **Email**: support@saidhome.ai

### Step 2: Receive License File

After purchase, users receive a `license.json` file:

```json
{
  "license_key": "LAM-PRO-XXXX-XXXX-XXXX",
  "tier": "pro",
  "customer": "Company Name",
  "issued_at": "2024-01-15T10:00:00",
  "expires_at": "2025-01-15T10:00:00"
}
```

### Step 3: Install License

Users can place the license file in any of these locations (checked in order):

#### All Platforms:

1. **Model Directory** (Recommended):
   ```bash
   LAM-base-v1/license.json
   ```

2. **Current Working Directory**:
   ```bash
   ./lam_license.json
   ```

#### Windows:

3. **User AppData** (Recommended for Windows):
   ```powershell
   %APPDATA%\lam\license.json
   # Example: C:\Users\YourName\AppData\Roaming\lam\license.json
   ```

4. **User LocalAppData**:
   ```powershell
   %LOCALAPPDATA%\lam\license.json
   # Example: C:\Users\YourName\AppData\Local\lam\license.json
   ```

5. **System-Wide** (Windows):
   ```powershell
   %ProgramData%\lam\license.json
   # Example: C:\ProgramData\lam\license.json
   ```

#### Linux/macOS:

3. **User Home Directory**:
   ```bash
   ~/.lam/license.json
   ```

4. **System-Wide** (Linux):
   ```bash
   /etc/lam/license.json
   ```

5. **System-Wide** (macOS):
   ```bash
   /Library/Application Support/lam/license.json
   ```

#### All Platforms (Environment Variable):

6. **Environment Variable**:
   ```bash
   # Windows (PowerShell):
   $env:LAM_LICENSE_KEY="LAM-PRO-XXXX-XXXX-XXXX"
   
   # Windows (CMD):
   set LAM_LICENSE_KEY=LAM-PRO-XXXX-XXXX-XXXX
   
   # Linux/macOS:
   export LAM_LICENSE_KEY="LAM-PRO-XXXX-XXXX-XXXX"
   ```

### Step 4: Verify License

The license is automatically detected when you initialize the model:

```python
from lam import LAM

# License is checked automatically during initialization
model = LAM('LAM-base-v1')

# Check license status
if hasattr(model, '_license_manager'):
    license_info = model._license_manager.get_license_info()
    if license_info:
        print(f"License Tier: {license_info['tier']}")
        print(f"Max Length: {license_info['max_length']} tokens")
    else:
        print("Free Tier: 4,096 token limit")
```

## License File Format

### Pro License Example

```json
{
  "license_key": "LAM-PRO-XXXX-XXXX-XXXX",
  "tier": "pro",
  "customer": "Your Company",
  "issued_at": "2024-01-15T10:00:00",
  "expires_at": "2025-01-15T10:00:00"
}
```

### Enterprise License Example

```json
{
  "license_key": "LAM-ENTERPRISE-XXXX-XXXX-XXXX",
  "tier": "enterprise",
  "customer": "Enterprise Corp",
  "issued_at": "2024-01-15T10:00:00",
  "expires_at": "2025-01-15T10:00:00",
  "support_email": "support@saidhome.ai"
}
```

## Usage Examples

### Free Tier (No License)

```python
from lam import LAM

model = LAM('LAM-base-v1')

# Works up to 4,096 tokens
embeddings = model.encode(["Your text here..."], max_length=4096)

# This will raise an error:
# embeddings = model.encode(["Very long text..."], max_length=5000)  # ❌ Error
```

### Pro/Enterprise Tier (With License)

```python
from lam import LAM

# License is automatically detected from license.json
model = LAM('LAM-base-v1')

# Works up to 32,768 tokens
embeddings = model.encode(["Very long document..."], max_length=32768)  # ✅ Works
```

## License Validation

The license system validates:

1. **License Key Format**: Must match `LAM-XXXX-XXXX-XXXX-XXXX` pattern
2. **Expiration Date**: License must not be expired
3. **Tier**: Must be `pro` or `enterprise` for 32k support

## Troubleshooting

### License Not Detected

If your license isn't being detected:

1. **Check File Location**: Ensure `license.json` is in one of the checked locations
2. **Check File Format**: Ensure JSON is valid
3. **Check Expiration**: Ensure license hasn't expired
4. **Check Environment Variable**: If using `LAM_LICENSE_KEY`, ensure it's set correctly

### Testing License System

Create a test license:

```python
from lam._license import create_sample_license
from pathlib import Path

# Create test license in model directory
create_sample_license(Path("LAM-base-v1/license.json"), tier="pro")
```

## Security Notes

- License keys are validated but not cryptographically signed in this implementation
- For production, implement:
  - RSA signature verification
  - License server validation
  - Hardware fingerprinting (optional)
  - Rate limiting

## Support

For license issues:
- **Email**: support@saidhome.ai
- **Website**: https://saidhome.ai
- **Documentation**: https://github.com/said-research/lam

