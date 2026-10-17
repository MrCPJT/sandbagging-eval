PY := uv run python
MODEL ?= anthropic/claude-haiku-5-5
TIER ?= 6
EPOCHS ?=
MODE ?= skeleton
APPROVE ?=
TIERS ?=
KEYCHAIN_ITEM ?= ntl-anthropic-api

# Approved runs read the API key from the macOS Keychain for that command only
KEY = key="$$(security find-generic-password -s $(KEYCHAIN_ITEM) -w)" && ANTHROPIC_API_KEY="$$key"
RUN = $(if $(APPROVE),$(KEY)) $(PY) -m pipeline.runner $(1) --model $(MODEL) --tier $(TIER) $(if $(TIERS),--tiers $(TIERS)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(APPROVE),--approve)

.PHONY: items items-replication skeleton calibrate dry run replicate analyse report test

items:
	$(PY) -c "from behaviours.sandbagging.items import main; main()"

items-replication:
	$(PY) -c "from behaviours.sandbagging.items import REPLICATION_SEED, main; main(REPLICATION_SEED, (6,))"

skeleton:
	$(call RUN,skeleton)

calibrate:
	$(call RUN,calibrate)

dry:
	$(call RUN,dry)

run:
	$(call RUN,full)

# Confirmatory rerun of paraphrase 8, neutral vs incentive, on fresh items
replicate:
	$(call RUN,replicate) --conditions neutral,incentive --paraphrases 8 --item-seed 20261018

analyse:
	$(PY) -m pipeline.analysis logs/$(MODE) results/$(MODE)

# Every number quoted in the README
report:
	$(PY) -m pipeline.report

test:
	uv run pytest -q
