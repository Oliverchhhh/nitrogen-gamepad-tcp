#!/bin/bash
# WSL 代理配置脚本（基于你的 Clash Verge 设置）
# Windows 主机 IP: 10.255.255.254
# 代理端口: 7897

# 设置 Windows 主机 IP 和代理端口
export WINDOWS_HOST=10.255.255.254
export PROXY_PORT=7897

echo "=== WSL 代理配置 ==="
echo "Windows Host IP: $WINDOWS_HOST"
echo "Proxy Port: $PROXY_PORT"
echo ""

# 1. 设置环境变量代理（当前会话）
export http_proxy=http://$WINDOWS_HOST:$PROXY_PORT
export https_proxy=http://$WINDOWS_HOST:$PROXY_PORT
export HTTP_PROXY=http://$WINDOWS_HOST:$PROXY_PORT
export HTTPS_PROXY=http://$WINDOWS_HOST:$PROXY_PORT

echo "✓ 环境变量代理已设置"
echo "  http_proxy=$http_proxy"
echo "  https_proxy=$https_proxy"
echo ""

# 2. 配置 apt 代理
sudo tee /etc/apt/apt.conf.d/95proxies > /dev/null <<EOF
Acquire::http::Proxy "http://${WINDOWS_HOST}:${PROXY_PORT}";
Acquire::https::Proxy "http://${WINDOWS_HOST}:${PROXY_PORT}";
EOF

echo "✓ apt 代理已配置"
echo ""

# 3. 配置 Git 代理
git config --global http.proxy http://${WINDOWS_HOST}:${PROXY_PORT}
git config --global https.proxy http://${WINDOWS_HOST}:${PROXY_PORT}

echo "✓ Git 代理已配置"
echo ""

# 4. 添加到 .bashrc（持久化）
if ! grep -q "# WSL 代理配置" ~/.bashrc; then
    cat >> ~/.bashrc <<BASHRC

# WSL 代理配置
export WINDOWS_HOST=10.255.255.254
export PROXY_PORT=7897
export http_proxy=http://\$WINDOWS_HOST:\$PROXY_PORT
export https_proxy=http://\$WINDOWS_HOST:\$PROXY_PORT
export HTTP_PROXY=http://\$WINDOWS_HOST:\$PROXY_PORT
export HTTPS_PROXY=http://\$WINDOWS_HOST:\$PROXY_PORT
BASHRC
    echo "✓ 代理配置已添加到 ~/.bashrc（永久生效）"
else
    echo "✓ 代理配置已存在于 ~/.bashrc"
fi

echo ""
echo "=== 配置完成 ==="
echo ""
echo "下一步："
echo "1. 确保 Clash Verge 中'局域网连接'已开启"
echo "2. 测试代理连接：curl -I https://www.google.com"
echo "3. 测试 apt：sudo apt update"
