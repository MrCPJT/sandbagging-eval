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

.PHONY: items skeleton calibrate dry run analyse test

items:
	$(PY) -c "from behaviours.sandbagging.items import main; main()"

skeleton:
	$(call RUN,skeleton)

calibrate:
	$(call RUN,calibrate)

dry:
	$(call RUN,dry)

run:
	$(call RUN,full)

analyse:
	$(PY) -m pipeline.analysis logs/$(MODE) results/$(MODE)

test:
	uv run pytest -q
