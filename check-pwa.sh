#!/bin/bash
# Script to check if a website has Progressive Web App (PWA) support
#
# Usage: ./check-pwa.sh [url]
#
# Examples:
#   ./check-pwa.sh https://kvcache.io/
#   ./check-pwa.sh  # defaults to https://kvcache.io/
#
# Checks for:
#   - Web app manifest (manifest.json)
#   - Theme color meta tag
#   - Apple mobile web app support
#   - Service worker registration
#   - App icons (standard and Apple)
#   - Viewport meta tag

set -eo pipefail

URL="${1:-https://kvcache.io/}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== PWA Support Checker ===${NC}"
echo -e "Checking: ${YELLOW}${URL}${NC}\n"

# Fetch the HTML content
HTML=$(curl -sL "$URL")

# Check for manifest link
echo -n "1. Manifest link: "
if echo "$HTML" | grep -q '<link[^>]*rel="manifest"'; then
    MANIFEST_PATH=$(echo "$HTML" | grep -o '<link[^>]*rel="manifest"[^>]*href="[^"]*"' | grep -o 'href="[^"]*"' | cut -d'"' -f2 | head -1)
    echo -e "${GREEN}✓ Found${NC} ($MANIFEST_PATH)"

    # Try to fetch manifest.json
    if [[ "$MANIFEST_PATH" == http* ]]; then
        MANIFEST_URL="$MANIFEST_PATH"
    elif [[ "$MANIFEST_PATH" == /* ]]; then
        BASE_URL=$(echo "$URL" | sed -E 's|(https?://[^/]+).*|\1|')
        MANIFEST_URL="${BASE_URL}${MANIFEST_PATH}"
    else
        MANIFEST_URL="${URL%/}/${MANIFEST_PATH}"
    fi

    echo -n "   Manifest accessible: "
    if curl -sL -o /dev/null -w "%{http_code}" "$MANIFEST_URL" | grep -q "200"; then
        echo -e "${GREEN}✓ Yes${NC}"
        MANIFEST_CONTENT=$(curl -sL "$MANIFEST_URL")
        echo "   Preview:"
        echo "$MANIFEST_CONTENT" | jq -r '"\(.name) - \(.description)"' 2>/dev/null || echo "   (Unable to parse JSON)"
    else
        echo -e "${RED}✗ No (404 or error)${NC}"
    fi
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check for theme-color
echo -n "2. Theme color: "
if echo "$HTML" | grep -q '<meta[^>]*name="theme-color"'; then
    THEME_COLOR=$(echo "$HTML" | grep -o '<meta[^>]*name="theme-color"[^>]*content="[^"]*"' | grep -o 'content="[^"]*"' | cut -d'"' -f2 | head -1)
    echo -e "${GREEN}✓ Found${NC} ($THEME_COLOR)"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check for Apple mobile web app capable
echo -n "3. Apple mobile web app: "
if echo "$HTML" | grep -q '<meta[^>]*name="apple-mobile-web-app-capable"'; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check for service worker registration
echo -n "4. Service worker: "
if echo "$HTML" | grep -q "serviceWorker"; then
    if echo "$HTML" | grep -q "\.register"; then
        SW_PATH=$(echo "$HTML" | grep -oP "register\(['\"]([^'\"]+)['\"]" | head -1 | grep -oP "['\"]([^'\"]+)['\"]" | tr -d "'" | tr -d '"' | head -1)
        if [ -n "$SW_PATH" ]; then
            echo -e "${GREEN}✓ Found${NC} ($SW_PATH)"
        else
            echo -e "${GREEN}✓ Found${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Detected but not registered${NC}"
    fi
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check for app icons
echo -n "5. App icons: "
ICON_COUNT=$(echo "$HTML" | grep -c '<link[^>]*rel="icon"' || true)
APPLE_ICON_COUNT=$(echo "$HTML" | grep -c '<link[^>]*rel="apple-touch-icon"' || true)
TOTAL_ICONS=$((ICON_COUNT + APPLE_ICON_COUNT))
if [ "$TOTAL_ICONS" -gt 0 ]; then
    echo -e "${GREEN}✓ Found${NC} (${ICON_COUNT} standard, ${APPLE_ICON_COUNT} Apple)"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check for viewport meta tag
echo -n "6. Viewport meta: "
if echo "$HTML" | grep -q '<meta[^>]*name="viewport"'; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Summary
echo -e "\n${BLUE}=== Summary ===${NC}"
CHECKS=0
PASSED=0

# Count checks
((CHECKS++)) || true
echo "$HTML" | grep -q '<link[^>]*rel="manifest"' && ((PASSED++)) || true

((CHECKS++)) || true
echo "$HTML" | grep -q '<meta[^>]*name="theme-color"' && ((PASSED++)) || true

((CHECKS++)) || true
echo "$HTML" | grep -q '<meta[^>]*name="apple-mobile-web-app-capable"' && ((PASSED++)) || true

((CHECKS++)) || true
(echo "$HTML" | grep -q "serviceWorker" && echo "$HTML" | grep -q "\.register") && ((PASSED++)) || true

((CHECKS++)) || true
[ "$TOTAL_ICONS" -gt 0 ] && ((PASSED++)) || true

((CHECKS++)) || true
echo "$HTML" | grep -q '<meta[^>]*name="viewport"' && ((PASSED++)) || true

PERCENTAGE=$((PASSED * 100 / CHECKS))

if [ "$PERCENTAGE" -ge 80 ]; then
    echo -e "${GREEN}PWA Support: ${PASSED}/${CHECKS} checks passed (${PERCENTAGE}%)${NC}"
    echo -e "${GREEN}✓ Good PWA support detected!${NC}"
elif [ "$PERCENTAGE" -ge 50 ]; then
    echo -e "${YELLOW}PWA Support: ${PASSED}/${CHECKS} checks passed (${PERCENTAGE}%)${NC}"
    echo -e "${YELLOW}⚠ Partial PWA support${NC}"
else
    echo -e "${RED}PWA Support: ${PASSED}/${CHECKS} checks passed (${PERCENTAGE}%)${NC}"
    echo -e "${RED}✗ Limited or no PWA support${NC}"
fi
