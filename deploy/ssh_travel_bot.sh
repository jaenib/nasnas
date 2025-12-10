#!/usr/bin/env expect

# Quick helper to SSH into the travel-bot server without typing the password.
# Usage: ./ssh_travel_bot.sh [optional command]

set timeout -1
set host "root@82.165.45.100"
set password "wnp5su9Q"

if {$argc == 0} {
    set cmd ""
} else {
    set cmd [join $argv " "]
}

if {$cmd eq ""} {
    spawn ssh -o StrictHostKeyChecking=no $host
} else {
    spawn ssh -o StrictHostKeyChecking=no $host $cmd
}
expect "*?assword:*"
send "$password\r"
interact
