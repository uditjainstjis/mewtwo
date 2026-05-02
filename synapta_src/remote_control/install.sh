#!/bin/bash
# ═══════════════════════════════════════════════════════
# Mewtwo Remote Control — Installation Script
# ═══════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="mewtwo-remote"

echo "🚀 Installing Mewtwo Remote Control..."

# 1. Disable screen blanking and auto-suspend
echo "[1/5] Disabling sleep/suspend/screensaver..."
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || true

# Disable GNOME screen lock and auto-suspend
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' 2>/dev/null || true

# 2. Enable Wake-on-LAN
echo "[2/5] Enabling Wake-on-LAN..."
IFACE=$(ip route get 1 | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
sudo ethtool -s "$IFACE" wol g 2>/dev/null || echo "  ⚠️  WoL may not be supported by this NIC"

# 3. Configure passwordless sudo for nvidia-smi and power commands
echo "[3/5] Configuring sudo rules..."
sudo tee /etc/sudoers.d/mewtwo-remote > /dev/null << 'EOF'
# Allow learner to run GPU and power commands without password
learner ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
learner ALL=(ALL) NOPASSWD: /usr/sbin/rtcwake
learner ALL=(ALL) NOPASSWD: /bin/systemctl suspend
learner ALL=(ALL) NOPASSWD: /bin/systemctl mask *
learner ALL=(ALL) NOPASSWD: /bin/systemctl unmask *
learner ALL=(ALL) NOPASSWD: /sbin/shutdown
learner ALL=(ALL) NOPASSWD: /sbin/reboot
EOF
sudo chmod 440 /etc/sudoers.d/mewtwo-remote

# 4. Create systemd service
echo "[4/5] Creating systemd service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Mewtwo GPU Remote Control Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=learner
WorkingDirectory=${PROJECT_DIR}
ExecStart=${SCRIPT_DIR}/start_server.sh
Restart=always
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/learner/.Xauthority
Environment=RC_HOST=0.0.0.0
Environment=RC_PORT=7777

[Install]
WantedBy=multi-user.target
EOF

chmod +x "${SCRIPT_DIR}/start_server.sh"

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

# 5. Get access info
echo "[5/5] Getting network info..."
LOCAL_IP=$(hostname -I | awk '{print $1}')
MAC=$(ip link show "$IFACE" 2>/dev/null | grep ether | awk '{print $2}')

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ Mewtwo Remote Control — INSTALLED                ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  🌐 Dashboard:  http://${LOCAL_IP}:7777               "
echo "║  🔑 SSH:        ssh learner@${LOCAL_IP}               "
echo "║  📡 MAC:        ${MAC}                               "
echo "║                                                      ║"
echo "║  The dashboard starts automatically on boot.         ║"
echo "║  Sleep/suspend is permanently disabled.              ║"
echo "║                                                      ║"
echo "║  From your laptop browser, open:                     ║"
echo "║    http://${LOCAL_IP}:7777                            "
echo "║                                                      ║"
echo "║  Controls available:                                 ║"
echo "║    • GPU monitoring (live)                           ║"
echo "║    • Run commands remotely                           ║"
echo "║    • Turn monitor on/off                             ║"
echo "║    • Schedule suspend + auto-wake                    ║"
echo "║    • Kill GPU processes                              ║"
echo "║    • Edit API keys (.env)                            ║"
echo "║    • Launch training jobs                            ║"
echo "╚══════════════════════════════════════════════════════╝"
