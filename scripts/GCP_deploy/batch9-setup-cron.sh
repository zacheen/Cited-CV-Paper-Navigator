#!/bin/bash
# Batch 9 [SSH] - Set up daily cron job for paper ingestion (5 PM local time)

# Set VM timezone to Los Angeles so cron handles daylight saving automatically
sudo timedatectl set-timezone America/Los_Angeles

# Add cron job: run daily at 6:30 PM PST/PDT
(crontab -l 2>/dev/null | grep -v "get_today_trend"; echo "30 18 * * * /usr/bin/docker exec -w /root cv-rag python3 update/get_today_trend.py >> /home/User/daily_papers.log 2>&1") | crontab -

# Verify
echo "Timezone set to:"
timedatectl | grep "Time zone"
echo ""
echo "Current cron jobs:"
crontab -l
