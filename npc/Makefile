# npc/Makefile — convenience targets for common pipeline runs

DND_CFG  := run_dnd_npc.yaml
DND_THEME := themes/dnd_npc

SA_CFG   := run_stock_analyst.yaml
SA_THEME  := themes/stock_analyst

PYTHON   := python3

# D&D NPC — full pipeline
dnd:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME)

# D&D NPC — regen only
dnd-gen:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME) --stages generate validate

# D&D NPC — train only (use existing dataset)
dnd-train:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME) --stages train

# Stock analyst — full pipeline
stock:
	$(PYTHON) pipeline/run.py --config $(SA_CFG) --theme $(SA_THEME)

# Validate configs only
validate-configs:
	$(PYTHON) pipeline/run.py --config $(DND_CFG) --theme $(DND_THEME) --validate-config
	$(PYTHON) pipeline/run.py --config $(SA_CFG) --theme $(SA_THEME) --validate-config
