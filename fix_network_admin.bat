@echo off
:: ============================================================
::  NETWORK FIX SCRIPT — Run as Administrator
::  Fixes: ICMP Ping, DNS Servers, Network Profile
:: ============================================================
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] This script must be run as Administrator.
    echo Right-click this file and select "Run as administrator".
    pause
    exit /b 1
)

echo ============================================================
echo   APPLYING ADMIN-LEVEL NETWORK FIXES
echo ============================================================
echo.

:: FIX 1 - Enable outbound ICMP ping (ICMPv4 + ICMPv6)
echo [FIX 1] Enabling ICMP Echo Request Outbound (Ping)...
netsh advfirewall firewall set rule name="Core Networking Diagnostics - ICMP Echo Request (ICMPv4-Out)" new enable=yes
netsh advfirewall firewall set rule name="Core Networking Diagnostics - ICMP Echo Request (ICMPv6-Out)" new enable=yes
echo        Done.

:: FIX 2 - Enable inbound ICMP ping
echo [FIX 2] Enabling ICMP Echo Request Inbound...
netsh advfirewall firewall set rule name="Core Networking Diagnostics - ICMP Echo Request (ICMPv4-In)"  new enable=yes
netsh advfirewall firewall set rule name="Core Networking Diagnostics - ICMP Echo Request (ICMPv6-In)"  new enable=yes
echo        Done.

:: FIX 3 - Set DNS to Google 8.8.8.8 / 8.8.4.4
echo [FIX 3] Setting DNS servers to Google (8.8.8.8 / 8.8.4.4)...
netsh interface ip set dns name="Ethernet" static 8.8.8.8
netsh interface ip add dns name="Ethernet" 8.8.4.4 index=2
echo        Done.

:: FIX 4 - Flush DNS cache
echo [FIX 4] Flushing DNS cache...
ipconfig /flushdns
echo        Done.

:: FIX 5 - Change network profile to Private via registry
echo [FIX 5] Setting network profile to Private...
powershell -Command "Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private" 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo        [INFO] Could not change via PowerShell (Group Policy may prevent this).
    echo        This is cosmetic only - your connection still works.
) ELSE (
    echo        Done.
)

:: FIX 6 - Re-run Winsock + TCP-IP reset as admin for full effect
echo [FIX 6] Resetting Winsock (admin)...
netsh winsock reset
echo [FIX 7] Resetting TCP/IP stack (admin)...
netsh int ip reset

echo.
echo ============================================================
echo   VERIFICATION
echo ============================================================
echo.
echo [TEST] Pinging 8.8.8.8...
ping -n 4 8.8.8.8

echo.
echo [TEST] Pinging google.com...
ping -n 4 google.com

echo.
echo [INFO] Current DNS servers:
netsh interface ip show dns "Ethernet"

echo.
echo ============================================================
echo  ALL FIXES APPLIED. Please RESTART YOUR PC for full effect.
echo ============================================================
echo.
pause
