.DEFAULT_GOAL := all

.PHONY: all clean libbine pico_core force_rebuild

# --- NEW: allow submit_wrapper to isolate stamp files per library ---
STAMP_DIR ?= obj        # changed from hard-coded "obj"

$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)

CFLAGS_COMMON = -O3 -Wall -I$(BINE_DIR)/include -MMD -MP

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif
export CFLAGS_COMMON

all: force_rebuild libbine pico_core

# --- UPDATED: use $(STAMP_DIR) instead of ./obj for the build stamps ---
PREV_DEBUG      := $(shell [ -f $(STAMP_DIR)/.debug_flag ] && cat $(STAMP_DIR)/.debug_flag)
PREV_LIB        := $(shell [ -f $(STAMP_DIR)/.last_lib ] && cat $(STAMP_DIR)/.last_lib)
PREV_CUDA_AWARE := $(shell [ -f $(STAMP_DIR)/.cuda_aware ] && cat $(STAMP_DIR)/.cuda_aware)

force_rebuild: $(STAMP_DIR)
	@if [[ ! -f $(STAMP_DIR)/.debug_flag || ! -f $(STAMP_DIR)/.last_lib || ! -f $(STAMP_DIR)/.cuda_aware || \
	     "$(PREV_DEBUG)" != "$(DEBUG)" || "$(PREV_LIB)" != "$(MPI_LIB)" || "$(PREV_CUDA_AWARE)" != "$(CUDA_AWARE)" ]]; then \
		echo -e "$(RED)[BUILD] LIB, DEBUG or CUDA flag changed. Cleaning subdirectories...$(NC)"; \
		$(MAKE) -C libbine clean; \
		$(MAKE) -C pico_core clean; \
		echo "$(DEBUG)"       > $(STAMP_DIR)/.debug_flag; \
		echo "$(MPI_LIB)"     > $(STAMP_DIR)/.last_lib; \
		echo "$(CUDA_AWARE)"  > $(STAMP_DIR)/.cuda_aware; \
	else \
		echo -e "$(BLUE)[BUILD] LIB, DEBUG or CUDA flag unchanged...$(NC)"; \
	fi

libbine:
	@echo -e "$(BLUE)[BUILD] Compiling libbine static library...$(NC)"
	$(MAKE) -C libbine $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(CUDA_AWARE),CUDA_AWARE=$(CUDA_AWARE))

pico_core: libbine
	@echo -e "$(BLUE)[BUILD] Compiling pico_core executable...$(NC)"
	$(MAKE) -C pico_core $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(CUDA_AWARE),CUDA_AWARE=$(CUDA_AWARE))

clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libbine clean
	@$(MAKE) -C pico_core clean
	@rm -rf $(STAMP_DIR)
