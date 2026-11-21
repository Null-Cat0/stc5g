# Dataset construction: raw captures (.pcap) -> windowed, labelled CSV files.
#
#   make            convert every capture for every window length in WINDOWS
#   make csv-1      only the 1 s windows (any value works: csv-0.5, csv-5, ...)
#   make help       list the targets and options
#
# Input layout (one sub-directory per class, names as in CLASS_MAP of src/stc5g/config.py):
#   $(PCAP_DIR)/youtube/*.pcap   $(PCAP_DIR)/twitch/*.pcap   $(PCAP_DIR)/tiktok/*.pcap
#
# Output layout (one sub-directory per window length, same class sub-directories):
#   $(CSV_DIR)/1s/youtube/<capture>_1s_tshark.csv   ...
#
# Each CSV row is one time window: frame.time (window start), frame.len (bytes),
# the mean inter-packet time, frame.time_epoch (packet count), throughput_Mbps and class.
#
# Requirements: tshark and editcap (Wireshark command-line tools) on PATH, and uv
# (or set STC5G to another way of running the package, e.g. `make STC5G=stc5g`).
#
# The conversion is incremental: existing CSV files are not regenerated and files that
# already have the class column are not relabelled. Use `make clean` to start over.

SHELL := /bin/bash

# ---- options (override on the command line: make PCAP_DIR=/data/pcap WINDOWS=1) ----
PCAP_DIR ?= data/pcap
CSV_DIR  ?= data/csv_throughput
WINDOWS  ?= 1
UE_ONLY  ?= 1
STC5G    ?= uv run stc5g

.DEFAULT_GOAL := csv
.PHONY: help setup check-tools repair repair-dry-run csv clean

help: ## Show this help
	@echo 'usage: make [target] [PCAP_DIR=...] [CSV_DIR=...] [WINDOWS="0.5 1 5 10"] [UE_ONLY=0|1]'
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_%-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the package and its dependencies with uv
	uv sync

check-tools: ## Check that tshark/editcap are installed and that PCAP_DIR exists
	@command -v tshark  >/dev/null || { echo "tshark not found: install the Wireshark CLI tools"; exit 1; }
	@command -v editcap >/dev/null || { echo "editcap not found: install the Wireshark CLI tools"; exit 1; }
	@test -d "$(PCAP_DIR)" || { echo "PCAP_DIR not found: $(PCAP_DIR)"; exit 1; }
	@echo "$$(tshark --version | head -n1)"
	@echo "pcap files in $(PCAP_DIR): $$(find "$(PCAP_DIR)" -name '*.pcap' | wc -l | tr -d ' ')"

repair: check-tools ## Repair truncated captures in place with editcap (keeps *.corrupt.backup)
	$(STC5G) repair --dir "$(PCAP_DIR)"

repair-dry-run: check-tools ## List the captures repair would process, without modifying them
	$(STC5G) repair --dir "$(PCAP_DIR)" --dry-run

csv: $(addprefix csv-,$(WINDOWS)) ## Convert all captures for every window in WINDOWS (default)

csv-%: check-tools ## Convert all captures with a window of % seconds (e.g. csv-1)
	@echo "==> $*s windows: $(PCAP_DIR) -> $(CSV_DIR)/$*s"
	$(STC5G) extract --secs $* --input-dir "$(PCAP_DIR)" --output-dir "$(CSV_DIR)/$*s" --ue-only $(UE_ONLY)
	$(STC5G) label --root "$(CSV_DIR)/$*s"

clean: ## Delete the generated CSV files (CSV_DIR)
	rm -rf "$(CSV_DIR)"
