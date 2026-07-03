#!/bin/bash
#
# monitor_load.sh
# Records load average, top 5 CPU-consuming processes, and iostat disk I/O
# every INTERVAL seconds for a total of DURATION seconds (default 15 min).
#
# Usage:
#   ./monitor_load.sh [duration_seconds] [interval_seconds] [output_file]
#
# Example (default 15 min, sampling every 30s):
#   ./monitor_load.sh
#
# Example (custom 10 min run, sampling every 10s, custom log path):
#   ./monitor_load.sh 600 10 /var/log/loadcheck.log

DURATION="${1:-900}"        # total run time in seconds (default: 900s = 15 min)
INTERVAL="${2:-30}"         # sampling interval in seconds (default: 30s)
OUTFILE="${3:-./load_report_$(date +%Y%m%d_%H%M%S).log}"

# Check for iostat (part of sysstat package)
if ! command -v iostat >/dev/null 2>&1; then
    echo "WARNING: 'iostat' not found. Install with: sudo apt-get install sysstat" | tee -a "$OUTFILE"
    HAVE_IOSTAT=0
else
    HAVE_IOSTAT=1
fi

END_TIME=$(( $(date +%s) + DURATION ))

echo "==========================================================" | tee -a "$OUTFILE"
echo "Load/CPU/IO monitoring started: $(date)"                     | tee -a "$OUTFILE"
echo "Duration: ${DURATION}s | Interval: ${INTERVAL}s"             | tee -a "$OUTFILE"
echo "Output file: $OUTFILE"                                       | tee -a "$OUTFILE"
echo "==========================================================" | tee -a "$OUTFILE"

while [ "$(date +%s)" -lt "$END_TIME" ]; do
    {
        echo ""
        echo "----- $(date '+%Y-%m-%d %H:%M:%S') -----"

        echo "--- Load Average ---"
        uptime

        echo ""
        echo "--- Top 5 CPU-consuming processes ---"
        # Header + top 5 processes sorted by CPU usage
        ps -eo pid,ppid,user,pcpu,pmem,comm --sort=-pcpu | head -n 6

        if [ "$HAVE_IOSTAT" -eq 1 ]; then
            echo ""
            echo "--- Disk I/O (iostat -xz, 1 sample) ---"
            iostat -xz 1 1 | grep -v '^$'
        fi

    } >> "$OUTFILE" 2>&1

    sleep "$INTERVAL"
done

echo "" | tee -a "$OUTFILE"
echo "Monitoring finished: $(date)" | tee -a "$OUTFILE"
echo "Full report saved to: $OUTFILE"
