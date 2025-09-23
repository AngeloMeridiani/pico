CFLAGS = $(CFLAGS_COMMON)

LIB_DIR        ?= ../lib
LIB_SRC        := $(wildcard *.c)

LIB_OBJ_DIR    ?= ../obj/lib
LIB_OBJS       := $(patsubst %.c, $(LIB_OBJ_DIR)/%.o, $(LIB_SRC))
LIB_DEPS       := $(patsubst %.c, $(LIB_OBJ_DIR)/%.d, $(LIB_SRC))
PICOLIB        := $(LIB_DIR)/picolib.a

LIB_OBJ_DIR_CUDA ?= ../obj/lib_cuda
LIB_OBJS_CUDA    := $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.o, $(LIB_SRC))
LIB_DEPS_CUDA    := $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.d, $(LIB_SRC))
PICOLIB_CUDA     := $(LIB_DIR)/picolib_cuda.a

ifeq ($(PICO_MPI_CUDA_AWARE),1)
all: $(PICOLIB) $(PICOLIB_CUDA)
else
all: $(PICOLIB)
endif

######################################################################
######################STANDARD MAKEFILE TARGETS#######################
######################################################################
$(PICOLIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(PICOLIB)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) -c $< -o $@

-include $(LIB_DEPS)

######################################################################
######################CUDA-AWARE MAKEFILE TARGETS#####################
######################################################################
$(PICOLIB_CUDA): $(LIB_OBJS_CUDA)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(PICOLIB_CUDA)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) -DPICO_MPI_CUDA_AWARE -ldcudart -c $< -o $@

-include $(LIB_DEPS_CUDA)

clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(LIB_OBJ_DIR) $(PICOLIB) $(LIB_OBJ_DIR_CUDA) $(PICOLIB_CUDA)
