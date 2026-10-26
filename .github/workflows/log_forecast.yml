# Logs the official Monte Carlo forecast every weekday after the US market closes,
# then saves data/forecast_log.csv back to this repository.
#
# 21:30 UTC = 5:30pm ET in daylight time (until Nov 1) and 4:30pm ET after that,
# so it always runs after the 4pm close. GitHub can start scheduled runs a few
# minutes late; that's fine.
#
# To run it by hand: GitHub -> Actions tab -> "Log daily forecast" -> Run workflow.

name: Log daily forecast

on:
  schedule:
    - cron: "30 21 * * 1-5"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  log:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install packages
        run: pip install -r requirements.txt

      - name: Check the math
        run: |
          python tests/test_portfolio.py
          python tests/test_analytics.py
          python tests/test_forecast_log.py

      - name: Log today's forecast (retries if Yahoo is busy)
        run: |
          for attempt in 1 2 3; do
            python log_forecast.py && exit 0
            echo "Attempt $attempt failed. Waiting 60 seconds..."
            sleep 60
          done
          exit 1

      - name: Save the log to the repository
        run: |
          git config user.name "forecast-bot"
          git config user.email "forecast-bot@users.noreply.github.com"
          if [ ! -f data/forecast_log.csv ]; then
            echo "No forecast log yet (weekend, or nothing logged today). Nothing to save."
            exit 0
          fi
          git add data/forecast_log.csv
          if git diff --cached --quiet; then
            echo "No new forecast to save."
          else
            git commit -m "Forecast log: $(date -u +%Y-%m-%d)"
            git push
          fi
