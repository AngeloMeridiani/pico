.DEFAULT_GOAL := all

.PHONY: all clean libbine pico_core

CFLAGS_COMMON = -O3 -Wall -I$(PICO_DIR)/include -MMD -MP

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif
export CFLAGS_COMMON

all: libbine pico_core

libbine:
	@echo -e "$(BLUE)[BUILD] Compiling libbine static library...$(NC)"
	$(MAKE) -C libbine $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PICO_MPI_CUDA_AWARE),PICO_MPI_CUDA_AWARE=$(PICO_MPI_CUDA_AWARE))

pico_core: libbine
	@echo -e "$(BLUE)[BUILD] Compiling pico_core executable...$(NC)"
	$(MAKE) -C pico_core $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PICO_MPI_CUDA_AWARE),PICO_MPI_CUDA_AWARE=$(PICO_MPI_CUDA_AWARE))

clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@rm -rf bin/ obj/ lib/
